-- 0003: L1 用户长期画像
-- 存结构化、可注入、可审计的人设级偏好 (不是事实记忆).
-- 字段值用中文, DBeaver 直接可读.
--
-- 设计原则:
--   - 只存"这个人什么样" (聊法/剧透/接梗), 不存事实 (事实归 Mem0).
--   - confidence 低时不注入 system prompt (冷启动用默认人设).
--   - 不每轮写, 由 messages_since_update 累计到阈值才触发 LLM 增量更新.

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id              UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    interaction_style    TEXT,       -- 吐槽型 / 分析型 / 陪伴型 / 提问型 / 混合
    spoiler_tolerance    TEXT,       -- 接受 / 谨慎 / 拒绝
    humor_level          TEXT,       -- 高 / 中 / 低
    confidence           DOUBLE PRECISION NOT NULL DEFAULT 0,
    messages_since_update INTEGER    NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_confidence
    ON user_profiles (confidence);
