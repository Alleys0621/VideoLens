# DB

- `docker-compose.yml` Postgres 容器 (端口 25432)
- `init.sql` 完整 schema (users / user_profiles / show_profiles / playback_progress / threads)
- `migrations/` 增量迁移 (0001-0006); 新库直接用 init.sql, 已有库跑 migrations

LangGraph 的 `checkpoints` / `checkpoint_writes` / `checkpoint_blobs` 由 `AsyncPostgresSaver.setup()` 自动建, 不在 init.sql 里.
