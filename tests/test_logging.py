"""Tests for quant2026.logging module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from quant2026.logging import (
    sanitize_log,
    setup_logging,
    setup_backtest_logging,
    backtest_logger,
    trade_logger,
    data_logger,
)


class TestSanitizeLog:
    """敏感信息脱敏测试。"""

    def test_api_key_masked(self):
        msg = "Using api_key=sk_abcdefgh12345678 for request"
        result = sanitize_log(msg)
        assert "sk_abcdefgh12345678" not in result
        assert "******" in result

    def test_token_masked(self):
        msg = "token: ghp_1234567890abcdef"
        result = sanitize_log(msg)
        assert "ghp_1234567890abcdef" not in result
        assert "******" in result

    def test_password_masked(self):
        msg = "password=MySecretPass123"
        result = sanitize_log(msg)
        assert "MySecretPass123" not in result
        assert "******" in result

    def test_ip_masked(self):
        msg = "Connecting to 192.168.1.100 on port 8080"
        result = sanitize_log(msg)
        assert "192.168.1.100" not in result
        assert "192.168.1.*" in result

    def test_stock_code_preserved(self):
        msg = "买入 600519 100股"
        result = sanitize_log(msg)
        assert "600519" in result

    def test_no_sensitive_info(self):
        msg = "回测完成，收益率 15.3%"
        result = sanitize_log(msg)
        assert result == msg


class TestSetupLogging:
    """setup_logging 基础测试。"""

    def test_setup_no_error(self):
        """调用 setup_logging 不报错。"""
        setup_logging(level="DEBUG")

    def test_setup_with_file(self, tmp_path):
        """文件日志写入。"""
        log_dir = str(tmp_path / "logs")
        setup_logging(level="DEBUG", log_dir=log_dir, log_file="test.log")
        logger.info("test file logging")
        # Force flush
        logger.complete()

        log_file = tmp_path / "logs" / "test.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "test file logging" in content

    def test_setup_level_filter(self, tmp_path):
        """INFO 级别不输出 DEBUG。"""
        log_dir = str(tmp_path / "logs")
        setup_logging(level="INFO", log_dir=log_dir, log_file="level.log")
        logger.debug("this is debug")
        logger.info("this is info")
        logger.complete()

        content = (tmp_path / "logs" / "level.log").read_text(encoding="utf-8")
        assert "this is debug" not in content
        assert "this is info" in content


class TestBacktestLogging:
    """回测专用日志测试。"""

    def test_backtest_log_created(self, tmp_path):
        """setup_backtest_logging 创建文件。"""
        setup_logging(level="DEBUG")
        log_dir = str(tmp_path / "bt_logs")
        handler_id = setup_backtest_logging(log_dir, run_id="test001")

        # 用 bound logger 写入
        backtest_logger.info("开始回测 test001")
        trade_logger.info("买入 600519 100股 价格 1800.00")
        logger.info("普通日志不应出现在回测文件中")
        logger.complete()

        # 检查文件存在
        files = list(Path(log_dir).glob("backtest_test001_*.log"))
        assert len(files) == 1

        content = files[0].read_text(encoding="utf-8")
        assert "开始回测 test001" in content
        assert "买入 600519" in content
        # 普通日志不应出现
        assert "普通日志不应出现" not in content

        logger.remove(handler_id)

    def test_backtest_log_trade_details(self, tmp_path):
        """回测日志包含交易细节。"""
        setup_logging(level="DEBUG")
        handler_id = setup_backtest_logging(str(tmp_path), run_id="trade_test")

        trade_logger.info("买入 | 600519 | 100股 | 价格 1800.00")
        trade_logger.info("卖出 | 000858 | 200股 | 价格 150.50")
        trade_logger.info("调仓 | 2024-06-01 | 持仓 5 只")
        logger.complete()

        files = list(tmp_path.glob("backtest_trade_test_*.log"))
        content = files[0].read_text(encoding="utf-8")
        assert "买入" in content
        assert "卖出" in content
        assert "调仓" in content

        logger.remove(handler_id)
