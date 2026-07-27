-- 0002: users 表加 updated_at + status
-- 目的:
--   - updated_at: 与其他业务表对齐, 便于审计/同步.
--   - status: 账号状态 (active/disabled), 为未来停用/封禁预留.
--     不引入 provider/external_id 等认证字段, 避免单 provider 阶段过度抽象.
--
-- 幂等: 可重复执行, 已存在的列/触发器不会报错.

-- 1. 加列 (如果不存在)
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE users ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';

-- 2. 加约束 (如果不存在). DO 块避免重复创建报错.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_status_check'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_status_check CHECK (status IN ('active', 'disabled'));
    END IF;
END$$;

-- 3. updated_at 触发器 (与 threads 一致)
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
