-- 0008: 人设卡（默认搭子 + thread 人设）

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id            UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    default_persona_id TEXT        NOT NULL DEFAULT 'alleys',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE threads ADD COLUMN IF NOT EXISTS persona_id TEXT NOT NULL DEFAULT 'alleys';
CREATE INDEX IF NOT EXISTS idx_threads_persona_id ON threads (persona_id);
