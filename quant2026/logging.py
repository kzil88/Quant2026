"""Quant2026 统一日志配置。

基于 loguru，提供统一格式、级别、文件输出、脱敏、回测专用日志。
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# ── 脱敏正则 ────────────────────────────────────────────────────

_API_KEY_PATTERN = re.compile(
    r"""(?i)"""
    r"""(?:api[_-]?key|token|secret|password|authorization|bearer)"""
    r"""[\s:='"]*"""
    r"""([A-Za-z0-9_\-]{8,})""",
)
_IP_PATTERN = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")


def sanitize_log(message: str) -> str:
    """敏感信息脱敏。

    - 股票代码保留（6 位纯数字）
    - API key/token/secret/password 用 ****** 替代
    - IP 地址脱敏为 x.x.x.*
    """
    # API key / token
    def _mask_key(m: re.Match) -> str:
        full = m.group(0)
        secret = m.group(1)
        return full.replace(secret, "******")

    result = _API_KEY_PATTERN.sub(_mask_key, message)

    # IP 地址脱敏（保留前三段，最后一段替换为 *）
    def _mask_ip(m: re.Match) -> str:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}.*"

    result = _IP_PATTERN.sub(_mask_ip, result)
    return result


def _sanitize_filter(record: dict) -> bool:
    """loguru filter：对 message 做脱敏处理。"""
    record["message"] = sanitize_log(record["message"])
    return True


# ── 日志配置 ────────────────────────────────────────────────────

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)

_LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | "
    "{name}:{function}:{line} | {message}"
)

_BACKTEST_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}"


def setup_logging(
    level: str = "INFO",
    log_dir: str | None = None,
    log_file: str = "quant2026.log",
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize: bool = False,
    colorize: bool = True,
    backtrace: bool = True,
    diagnose: bool = False,
) -> None:
    """统一配置 loguru。

    1. 清除默认 handler
    2. 添加 stderr handler（带颜色、配置级别）
    3. 如果 log_dir 不为空，添加文件 handler
    4. 过滤敏感信息
    """
    logger.remove()

    # stderr handler
    logger.add(
        sys.stderr,
        level=level,
        format=_LOG_FORMAT,
        colorize=colorize,
        backtrace=backtrace,
        diagnose=diagnose,
        filter=_sanitize_filter,
    )

    # 文件 handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path / log_file),
            level=level,
            format=_LOG_FORMAT_FILE,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            backtrace=backtrace,
            diagnose=diagnose,
            filter=_sanitize_filter,
            encoding="utf-8",
        )


def setup_backtest_logging(log_dir: str, run_id: str) -> int:
    """回测专用日志。

    - 文件名: {log_dir}/backtest_{run_id}_{date}.log
    - 只记录 module=backtest/trade 的日志
    - 格式: {time} | {level} | {message}

    Returns:
        handler id，方便回测结束后 logger.remove(id)
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"backtest_{run_id}_{date_str}.log"

    def _backtest_filter(record: dict) -> bool:
        module = record.get("extra", {}).get("module", "")
        return module in ("backtest", "trade")

    handler_id = logger.add(
        str(log_path / filename),
        level="DEBUG",
        format=_BACKTEST_FORMAT,
        filter=_backtest_filter,
        encoding="utf-8",
    )
    return handler_id


# ── 预定义 logger 实例 ──────────────────────────────────────────

backtest_logger = logger.bind(module="backtest")
trade_logger = logger.bind(module="trade")
data_logger = logger.bind(module="data")
factor_logger = logger.bind(module="factor")
risk_logger = logger.bind(module="risk")
