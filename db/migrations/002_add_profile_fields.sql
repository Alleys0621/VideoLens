-- 002: 画像字段扩充 (L1 + L2)
-- 已有库执行此迁移; 新库直接用 init.sql

-- L1: 加 engagement_motivation
ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS engagement_motivation TEXT;

-- L2: 加 attention_characters / theme_preferences / disliked_elements
ALTER TABLE show_profiles
    ADD COLUMN IF NOT EXISTS attention_characters TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS theme_preferences TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS disliked_elements TEXT[] NOT NULL DEFAULT '{}';
