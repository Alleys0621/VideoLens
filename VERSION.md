# 更新日志

完整版本历史。最新版本说明同时见 [README](./README.md)。

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
