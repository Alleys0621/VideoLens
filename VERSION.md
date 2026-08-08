# 更新日志

完整版本历史。最新版本说明同时见 [README](./README.md)。

## V1.4.4（2026-08-08）

**用户画像双轨分层：内核层稳定 + 表现层即时**

按"字段描述的是用户是谁（内核）还是用户现在怎样（表现）"重构 L1 画像。

**分层字段**
- 内核层（稳定，不易变）：interaction_style / interaction_initiative / engagement_motivation / humor_level / teasing_tolerance / spoiler_tolerance / pet_peeves，`conf_stable >= 0.6` 才注入 prompt
- 表现层（易变，用户当下指令）：`alleys_attitude` / `alleys_response_preference`，每轮即时覆盖，渲染时始终注入并排第一位（标【即时】）

**双轨采集**
- 轻量轨道（每轮，qwen-turbo）：`maybe_update_user_profile_instant` 更新表现层，保证用户"别损了/太啰嗦"这类指令即时生效
- 完整轨道（每 5 轮，qwen3.7-flash）：更新内核层，`conf_stable` 用 EWMA（α=0.3）维护 + `_FIELD_GATE=0.5` 字段门控，防低置信度噪声污染稳定画像

**DB / 字段**
- `0008_l1_profile_extension.sql`：加 interaction_initiative / alleys_response_preference / teasing_tolerance / pet_peeves
- `0009_l1_profile_dual_track.sql`：`confidence` 列改名 `conf_stable`
- 删除 `render_profile_overlay()`，画像统一走 `companion.py::_format_l1_l2` 分层注入

**prompt 结构**
- 新增 `l1_instant_system`（轻量轨道 prompt），`l1_system` 改为输出 `conf_stable`
- user_blocks 块名统一：`profile→user_profile`、`history→short_term_memory`、`long_term→episode_memory`、`kb→video_kb`、`web→web_search`
- `l1_instant_system` 判断标准：只看用户这句话的**对象**是剧情还是 Alleys，区分吐槽剧情 vs 对 Alleys 的指令

**迁移**
- 协作者 pull 后跑 `python -m scripts.apply_migrations`，重启 `.\start.bat`

## V1.4.3（2026-08-04）

**Alleys 视角重建：工作记忆 + 人设重构 + Mem0 质量优化**

基于真实用户对话（jinyao_chat）的问题分析，从"塞 context 的聊天 LLM"回归"有感知的陪看者"。针对陪看场景三个根本错位（没有视角 / 记忆是标签不是认知 / 人设是性格形容词不是行为契约）做架构性修复。

**工作记忆机制（感知层）**
- watching_state 的 video_time → 按 KB event 时间范围（action.evidence.timestamp 派生）找当前事件
- 注入"你现在看到的"段（事件级 summary），框成"Alleys 的视角，不是客观事实"
- 间隙（>5s 无剧情）自动跳过，Alleys 自然闲聊不硬扯画面
- system prompt 加元指令：这是你的工作记忆，细节没给就是没看清不要补
- 修复：剧情幻觉（当前画面）、假装看到、主语错乱（把用户当剧情角色）

**人设 prompt 重构（对话层）**
- companion_alleys_system 并入 companion_prompts（单 key 统一，去掉 system/user 嵌套）
- 三条铁律：情绪优先 / 说错就认 / 不知道就说不知道
- few-shot 示例（含纠错、"没细讲过"、长度控制）
- user_blocks 第一视角：你记得的关于用户的事 / 你对这位用户的理解 / 你现在看到的
- 删除"代词指代优先推断"（jinyao 会话 4 过度解读用户情绪的元凶）

**Mem0 长期记忆质量优化（记忆层）**
- custom_instructions 重写：记身份/生活/观影偏好/角色立场/与 Alleys 的关系
- 明确"不记"：剧情梗概（KB 有）、助手发言、一次性情绪、会话内临时状态
- update 语义明确：ADD/UPDATE/DELETE/NOOP，冲突时 UPDATE 不新增重复
- 修复：之前 60%+ 记忆是剧情梗概（和 KB 重复），Mem0 变成了第二个 KB

**画像更新修复**
- counter 原子归零（CASE...RETURNING 同一 SQL）：修复 async 并发导致的"每轮都更新"
- 修复调用方：increment 改返回 bool 后，video_utils 还用 `n >= 阈值`（bool>=2 永远 False，画像静默失效）→ 直接用 bool
- L1/L2 严格同步：show 门控前置，show 为空两者都不更新
- 阈值 5→2，预筛 4→2（更新更及时）
- 删废弃常量 PROFILE_INJECT_MIN_CONFIDENCE + 死代码 reset_message_counter

**会话标题恢复**
- V1.4 重构误删 companion_node 的 maybe_set_thread_title 异步调用 → 标题生成成死代码（V1.4 后新会话标题全是前端首条消息截断）
- 恢复首轮异步生成；thread_title.py model 写死 qwen3-flash → 走 config（model_text_flash）

**Pipeline 清理**
- TransNetV2 移除：Stage 2 唯一路径 speaker_anchor，无声纹 segments 直接 RuntimeError（不再回退 SBD）
- 删 src/scene/ 目录 + stage2_visual 死代码（extract_keyframes / filter_subtitle_frames）

**基础设施**
- cf tunnel 文件名兼容（cloudflared.exe / cloudflared-windows-amd64.exe 都识别）
- start.bat 后端显式 set PYTHONUTF8=1（修 langgraph_api 在 Windows GBK 下崩溃）
- log 全部归集到 logs/（之前散落在 data/output/）
- 目录文档体系：所有 __init__.py 加 docstring + frontend/config/db/scripts 加 README.md
- VERSION.md 从 README 拆出（README 聚焦部署/更新）

## V1.4.2（2026-08-03）

**观看状态（watching state）上报机制**
- 前端播放器定时上报视频坐标（video_dir / video_time / is_playing），后端存 `watching_state` 表
- agent 检索优先用 watching_state 的 video_time（持续坐标），fallback configurable（提交瞬间快照）
- 上报频率：播放中 5s 节流 + pause/seeked/ended/切集 即时上报
- 为 agent 化（`retrieve_kb` 时间锚点）和后续主动式服务铺路
- 新增 `/api/watching` POST 端点（JWT 鉴权，复用 `/api/playback` 模式）
- DB migration: `db/migrations/0007_watching_state.sql`

**看完一集的记忆沉淀 hook**
- `/api/playback` 在 `completed=true` 时留 hook 点（空实现 + TODO），后续填 episodic memory 总结

## V1.4（2026-08-01）

**对话为核心架构（v2_dialog_first）**
- 去掉意图路由：删除 `intent_router.py` / `context_builder.py`，不再按 task 裁剪上下文
- 每轮全量注入：L1+L2 画像 + 最近 5 轮 + Mem0 长期记忆 + KB 检索，由 prompt 引导"按需引用"（用户问剧情才参考 KB，闲聊不复述）
- 主 LLM 固定 qwen3.7-flash（`COMPANION_MAIN_MODEL=plus` 可切），不再按置信度切模型
- 对话 prompt 全部 yaml 驱动（`companion_prompts.user_blocks`），改 yaml 即可热编辑
- 抽取 `video_utils.py` 统一数据加载 + 异步副作用

**L1 画像新增 `alleys_attitude`**
- 记录用户对 Alleys 的态度/印象/期望，注入 L1 overlay 影响 Alleys 回应方式
- 画像 prompt 迁到 yaml（`profile_prompts.l1_system / l2_system`），代码内保留 fallback
- DB migration: `db/migrations/0005_user_profiles_alleys_attitude.sql`

**数据源统一 + 清理**
- 检索/加载统一改读 `stage3_kb.json`，删除冗余的 `stage3_dryrun.json`
- 移除误跟踪的 `data/_mem0_qdrant.bak/`（本地向量库备份，进 `.gitignore`）

## V1.3.1（2026-07-30）

**用户画像升级（L1 + L2）**
- L1 新增 `engagement_motivation`（推理探索型 / 情绪共鸣型 / 角色陪伴型 / 剧情消费型）
- L2 新增 `attention_characters`（关注 ≠ 喜欢）、`theme_preferences`、`disliked_elements`
- L1/L2 prompt 重写：详细 confidence 标尺 + 严格区分用户/Alleys 发言
- L2 模型 qwen-plus → qwen3.7-flash（统一）
- 去掉 L1/L2 注入门控（confidence < 阈值不注入），允许 agent 边聊边猜、像人一样逐步了解用户
- DB migration: `db/migrations/002_add_profile_fields.sql`

## V1.3（2026-07-29）

**延迟优化：localhost → 127.0.0.1**
- Windows 上 `localhost` 先试 IPv6 `::1`，超时 ~2s 后 fallback IPv4，每条消息白白多等 2s
- 全站 `localhost:2024` → `127.0.0.1:2024`（前端 `.env`、代理默认值、SDK 默认值）
- TTFB 从 ~3s 降到 ~1s

**统一启动脚本**
- 删除 `start-prod.bat` + `scripts/_langgraph_prod.py`（dev/prod 后端延迟无差异，不再维护两套）
- 日常一律用 `start.bat`

## V1.2.1（2026-07-29）

**意图路由 hybrid 化**
- 新增 `hybrid_route_intent`：embedding 优先，cosine < 阈值时回退 LLM
- 实测 50 条混合数据：准确率 96%（vs 纯 LLM 87.5%、纯 emb 84%），70% query 走 emb 直通，平均 485ms（vs 纯 LLM 1976ms，快 4 倍）
- 默认阈值 0.65（基于数据扫描甜点位）

**主 LLM 按置信度切 flash/plus**
- intent 置信度 ≥ 0.75 → qwen3.7-flash（首 token 快 2-4 倍）
- 否则 → qwen3.7-plus（防幻觉）
- 26 条 task×cosine 矩阵测试定位 flash 崩塌边界：低置信区间 flash 易幻觉/越界，plus 更稳

**Embedding 升级**
- `text-embedding-v3` → `qwen3.7-text-embedding`
- 100 条数据标定：两模型 cosine 分布几乎一致（中位漂移 0.010），所有阈值上 qwen3.7 准确率都 ≥ v3
- 阈值不需要重新校准

**配置统一进 yaml**
- 路由/模型相关配置全部从 `.env` 迁到 `config/pipeline.yaml`
- `.env` 只留 `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`

## V1.2（2026-06）

陪看智能体心智层升级：语义意图理解、上下文白名单、分层用户画像、中文会话标题。
