"""日志管理 - 彩色控制台 + 文件输出"""

import logging
import sys

import coloredlogs


def setup_logger(name: str = "videolens", level: str = "INFO") -> logging.Logger:
    """创建并配置 logger"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        field_styles={
            "asctime": {"color": "cyan"},
            "levelname": {"bold": True, "color": "white"},
            "name": {"color": "blue"},
        },
        level_styles={
            "debug": {"color": "green"},
            "info": {"color": "white"},
            "warning": {"color": "yellow"},
            "error": {"color": "red", "bold": True},
            "critical": {"background": "red", "color": "white", "bold": True},
        },
    )

    return logger


def get_logger(name: str = "videolens") -> logging.Logger:
    """获取 logger 实例"""
    return setup_logger(name)
