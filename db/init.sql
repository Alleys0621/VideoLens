-- VideoLens 数据库初始化
-- 由 docker-entrypoint-initdb.d 在首次创建数据库时自动执行
-- (后续重启不会重复执行, 除非 down -v 删数据卷)
--
-- 注意: LangGraph 的 checkpoints / checkpoint_writes / checkpoint_blobs 表
-- 由 AsyncPostgresSaver.setup() 在首次连接时自动创建, 不在这里建.

-- ============================================================
-- 用户表
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            UUID         PRIMARY KEY,
    phone         VARCHAR(11)  NOT NULL UNIQUE,
    password_hash TEXT         NOT NULL,
    display_name  VARCHAR(64)  NOT NULL,
    -- status: 账号状态. active=正常, disabled=停用 (后台封禁/注销预留).
    -- 不在这里放认证方式字段 (provider/external_id 等), 避免未来扩展污染本表.
    status        TEXT         NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'disabled')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone);

-- ============================================================
-- 用户长期画像 (L1)
-- 结构化、可注入、可审计的人设级偏好. 字段值用中文.
-- confidence 低时不注入 system prompt. 不每轮写, 由阈值触发更新.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id               UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    interaction_style     TEXT,       -- 吐槽型 / 分析型 / 陪伴型 / 提问型 / 混合
    spoiler_tolerance     TEXT,       -- 接受 / 谨慎 / 拒绝
    humor_level           TEXT,       -- 高 / 中 / 低
    confidence            DOUBLE PRECISION NOT NULL DEFAULT 0,
    messages_since_update INTEGER     NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_confidence ON user_profiles (confidence);

-- ============================================================
-- 作品画像 (L2) — per user × show
-- 事实类 (喜欢/讨厌的角色、角色评价), 作为 context 段注入 companion/knowledge.
-- ============================================================
CREATE TABLE IF NOT EXISTS show_profiles (
    user_id             UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show                TEXT        NOT NULL,
    favorite_characters TEXT[]      NOT NULL DEFAULT '{}',
    character_opinions  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    confidence          DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, show)
);

-- ============================================================
-- 播放进度表 (用户 × 视频 = 一条记录)
-- ============================================================
CREATE TABLE IF NOT EXISTS playback_progress (
    user_id    UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_dir  TEXT        NOT NULL,
    position   DOUBLE PRECISION NOT NULL DEFAULT 0,
    duration   DOUBLE PRECISION,
    completed  BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, video_dir)
);

CREATE INDEX IF NOT EXISTS idx_playback_user ON playback_progress (user_id);

-- ============================================================
-- 会话(thread)元数据表
-- - thread_id 是 LangGraph 生成的 UUID, 这里复用作为 PK
-- - 删 thread 时, 应用层先调 LangGraph /threads/{id} DELETE, 再删本表
--   (CASCADE 不会触发 LangGraph 那边, 必须应用层双删)
-- ============================================================
CREATE TABLE IF NOT EXISTS threads (
    thread_id     UUID        PRIMARY KEY,
    user_id       UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    custom_title  TEXT,
    pinned        BOOLEAN     NOT NULL DEFAULT FALSE,
    -- created_at/updated_at 在 LangGraph 那边维护, 这里冗余存一份便于排序/分组
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON threads (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_user_pinned ON threads (user_id, pinned);

-- updated_at 自动更新触发器
CREATE OR REPLACE FUNCTION trg_threads_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS threads_set_updated_at ON threads;
CREATE TRIGGER threads_set_updated_at
    BEFORE UPDATE ON threads
    FOR EACH ROW
    EXECUTE FUNCTION trg_threads_set_updated_at();

-- users.updated_at 自动更新触发器
CREATE OR REPLACE FUNCTION trg_users_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS users_set_updated_at ON users;
CREATE TRIGGER users_set_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION trg_users_set_updated_at();
