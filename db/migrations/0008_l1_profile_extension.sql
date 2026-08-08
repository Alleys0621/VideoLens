-- 0008: L1 画像扩展 — 引入行为指令式字段
-- 借鉴 fix.txt 设计: 字段值 → 行为指令, 配合 companion.py 的 _format_l1_l2 翻译逻辑
-- 新字段:
--   interaction_initiative     主动型/被动型/混合, 决定平淡段是否主动起话题
--   alleys_response_preference 反问引导/直接表态/拱火加码/冷静降温, 决定回应策略
--   teasing_tolerance          能被吐槽/只能吐槽剧情/完全不能, 配合 humor_level
--   pet_peeves                 雷区 list, 遇到时主动吐槽

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS interaction_initiative     TEXT,
    ADD COLUMN IF NOT EXISTS alleys_response_preference TEXT,
    ADD COLUMN IF NOT EXISTS teasing_tolerance          TEXT,
    ADD COLUMN IF NOT EXISTS pet_peeves                 TEXT[] NOT NULL DEFAULT '{}';
