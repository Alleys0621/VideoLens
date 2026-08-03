"""日志管理 - 彩色控制台 + 文件输出"""

import logging
import os
import sys

import coloredlogs


def setup_logger(name: str = "videolens", level: str = "INFO") -> logging.Logger:
    """创建并配置 logger（彩色控制台 + 纯文本文件）"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- 控制台输出（带颜色 + 毫秒时间戳，由 coloredlogs 管理）---
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        field_styles={
            "asctime": {"color": "cyan"},
            "levelname": {"bold": True, "color": "white"},
            "name": {"color": "blue"},
        },
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "green"},
            "warning": {"color": "yellow"},
            "error": {"color": "red", "bold": True},
            "critical": {"background": "red", "color": "white", "bold": True},
        },
    )

    # --- 文件 Handler（纯文本，无颜色）---
    # 日志放项目根 logs/ 目录 (运行时数据, 不进 git)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "videolens.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "videolens") -> logging.Logger:
    """获取 logger 实例"""
    return setup_logger(name)
