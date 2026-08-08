-- 0009: L1 画像双轨分层改造
--
-- 画像字段按稳定性分两层:
--   内核层 (稳定, 不易变): interaction_style / interaction_initiative /
--                        engagement_motivation / humor_level / teasing_tolerance /
--                        spoiler_tolerance / pet_peeves
--      - 由完整轨道每 N 轮更新, EWMA 维护 conf_stable, 渲染时 conf_stable>=0.6 才注入
--   表现层 (易变, 用户当下指令): alleys_attitude / alleys_response_preference
--      - 由轻量轨道每轮更新, 直接覆盖无置信度, 渲染时始终注入并排第一位
--
-- confidence 列改名为 conf_stable, 只代表内核层的稳定置信度.
-- 表现层无置信度 (用户明说就是真理).

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS conf_stable DOUBLE PRECISION NOT NULL DEFAULT 0,
    DROP COLUMN IF EXISTS confidence;
