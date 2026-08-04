# Scripts

## 运维
- `_video_server.py` 视频静态服务 (端口 9802, 独立于 Next.js 避免影响视频流)
- `apply_migrations.py` 应用 db/migrations
- `migrate_to_postgres.py` 迁移到 Postgres
- `cleanup_orphan_threads.py` 清理孤立会话
- `kill-videolens-ports.ps1` 杀端口 3000/2024/9800/9801
- `install-cloudflared.bat` 装 cloudflared
- `transcode_videos.bat` 视频转码

## Pipeline 子阶段
- `stage3_p1p2.py` Action + Event 抽取
- `stage3_p345.py` PlotArc + 摘要 + Global
- `stage3_p6.py` 角色深度画像
- `stage3_eval.py` Stage 3 评估

## 测试
- `latency_test.py` 端到端延迟测试
- `export_for_gt.py` 导出 GT 数据
