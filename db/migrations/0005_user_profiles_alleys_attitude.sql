-- 0005: L1 画像新增 alleys_attitude (用户对 Alleys 的态度/印象/期望)
-- 已有库执行此迁移; 新库直接用 init.sql

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS alleys_attitude TEXT;
