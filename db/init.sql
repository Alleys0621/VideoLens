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
    -- 内核层 (稳定, 不易变): 由完整轨道每 N 轮更新, conf_stable 门控注入
    user_id                   UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    interaction_style         TEXT,       -- 吐槽型 / 分析型 / 陪伴型 / 提问型 / 混合
    spoiler_tolerance         TEXT,       -- 接受 / 谨慎 / 拒绝
    humor_level               TEXT,       -- 高 / 中 / 低
    engagement_motivation     TEXT,       -- 推理探索型 / 情绪共鸣型 / 角色陪伴型 / 剧情消费型
    interaction_initiative    TEXT,       -- 主动型 / 被动型 / 混合 (决定平淡段是否主动起话题)
    teasing_tolerance         TEXT,       -- 能被吐槽 / 只能吐槽剧情 / 完全不能
    pet_peeves                TEXT[]      NOT NULL DEFAULT '{}',  -- 雷区 list, 遇到主动吐槽
    conf_stable               DOUBLE PRECISION NOT NULL DEFAULT 0,  -- 内核层稳定置信度 (EWMA)
    -- 表现层 (易变, 用户当下指令): 由轻量轨道每轮更新, 直接覆盖, 无置信度
    alleys_attitude           TEXT,       -- 用户对 Alleys 的当下态度/指令 (自由文本, 渲染排第一位)
    alleys_response_preference TEXT,      -- 当下回应策略: 反问引导 / 直接表态 / 拱火加码 / 冷静降温
    messages_since_update     INTEGER     NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
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
    attention_characters TEXT[]     NOT NULL DEFAULT '{}',
    character_opinions  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    theme_preferences   TEXT[]      NOT NULL DEFAULT '{}',
    disliked_elements   TEXT[]      NOT NULL DEFAULT '{}',
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
-- 观看状态表 (实时坐标, 每用户一条)
-- 给 agent 的 retrieve_kb tool 做 video_time 锚点; 为主动式服务铺路.
-- 与 playback_progress 区别: 这里存"此刻", 那里存"上次到哪了".
-- ============================================================
CREATE TABLE IF NOT EXISTS watching_state (
    user_id     UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    video_dir   TEXT        NOT NULL,
    video_time  DOUBLE PRECISION NOT NULL DEFAULT 0,
    is_playing  BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
