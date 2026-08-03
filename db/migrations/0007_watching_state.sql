-- 0007: 观看状态表 (实时坐标, 每用户一条)
-- 已有库执行此迁移; 新库直接用 init.sql

CREATE TABLE IF NOT EXISTS watching_state (
    user_id     UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    video_dir   TEXT        NOT NULL,
    video_time  DOUBLE PRECISION NOT NULL DEFAULT 0,
    is_playing  BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
