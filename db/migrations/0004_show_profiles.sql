-- 0004: L2 作品画像 (per user × show)
-- 存"Alleys 和这个用户在这部剧里"的关系记忆: 喜欢谁、对谁什么态度.
-- 事实类 → 作为 context 段注入 (companion/knowledge), 不进 system prompt.
--
-- 设计原则:
--   - 按 (user_id, show) 一行, show = video_dir 顶层目录 (如 "家有儿女").
--   - 与 L1 不重叠: L1 是人设级偏好, L2 是剧内角色评价.
--   - 与 playback_progress 不重叠: 那个存进度, 这个存情感态度.

CREATE TABLE IF NOT EXISTS show_profiles (
    user_id            UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show               TEXT        NOT NULL,
    favorite_characters TEXT[]     NOT NULL DEFAULT '{}',
    character_opinions JSONB       NOT NULL DEFAULT '[]'::jsonb,
    confidence         DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, show)
);
