"""路径与配置解析工具。

从 orchestrator.py 抽出, 供所有 stage / eval / 脚本共享,
避免 stage → orchestrator 的反向依赖 (循环导入)。
"""

import os

import yaml


def resolve_video_path(video_dir: str) -> str:
    """解析视频文件路径, 支持 data/videos/ 下的子目录结构.

    支持的视频格式 (按优先级): .mp4, .mkv, .mov, .avi
    支持嵌套子目录, 如 "家有儿女/第一季/第01集".

    例如:
      "052 鸟蛋之争"               → data/videos/喜羊羊与灰太狼/052 鸟蛋之争.mp4
      "家有儿女/第001集"            → data/videos/家有儿女/第二季/第001集.mp4
      "家有儿女/第一季/第01集"       → data/videos/家有儿女/第一季/第01集.mkv
    """
    VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")
    videos_root = "data/videos"

    # 1. 直接路径 (含子目录)
    for ext in VIDEO_EXTS:
        direct = os.path.join(videos_root, f"{video_dir}{ext}")
        if os.path.isfile(direct):
            return direct

    # 2. 在子目录中扁平搜索 (兼容老式 "第001集" 不带季路径)
    if os.path.isdir(videos_root):
        for subdir in os.listdir(videos_root):
            subdir_path = os.path.join(videos_root, subdir)
            if os.path.isdir(subdir_path):
                # 一级子目录
                for ext in VIDEO_EXTS:
                    candidate = os.path.join(subdir_path, f"{video_dir}{ext}")
                    if os.path.isfile(candidate):
                        return candidate
                # 二级子目录 (季)
                for sub2 in os.listdir(subdir_path):
                    sub2_path = os.path.join(subdir_path, sub2)
                    if os.path.isdir(sub2_path):
                        for ext in VIDEO_EXTS:
                            candidate = os.path.join(sub2_path, f"{video_dir}{ext}")
                            if os.path.isfile(candidate):
                                return candidate

    # 3. 返回默认路径 (后续会报错)
    return os.path.join(videos_root, f"{video_dir}.mp4")


def get_show_name(video_dir: str) -> str:
    """从 video_dir 推断所属影视作品名

    Returns:
        如 "喜羊羊与灰太狼", "家有儿女", 或 ""
    """
    videos_root = "data/videos"

    # 子目录格式: "家有儿女/第001集"
    if "/" in video_dir or "\\" in video_dir:
        parts = video_dir.replace("\\", "/").split("/")
        return parts[0] if parts else ""

    # 平铺格式: 搜索哪个子目录包含此文件
    if os.path.isdir(videos_root):
        for subdir in os.listdir(videos_root):
            subdir_path = os.path.join(videos_root, subdir)
            if os.path.isdir(subdir_path):
                candidate = os.path.join(subdir_path, f"{video_dir}.mp4")
                if os.path.isfile(candidate):
                    return subdir

    return ""


def load_voiceprint_config(show_name: str):
    """从 pipeline.yaml 加载对应影视作品的声纹配置

    Returns:
        (group_id, name_map) 或 ("", None) 如果未配置
    """
    config_path = os.path.join("config", "pipeline.yaml")
    if not os.path.isfile(config_path):
        return "", None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    groups = cfg.get("voiceprint_groups", {})
    show_cfg = groups.get(show_name, {})
    if not show_cfg:
        return "", None

    return show_cfg.get("group_id", ""), show_cfg.get("name_mapping", {})
