"""CLI 测试。"""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from quant2026.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def default_config():
    """Return path to default config relative to project root."""
    return str(Path(__file__).resolve().parent.parent / "config" / "default.yaml")


class TestHelp:
    """Test --help for all commands."""

    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Quant2026" in result.output

    def test_backtest_help(self, runner):
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_optimize_help(self, runner):
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output

    def test_walkforward_help(self, runner):
        result = runner.invoke(cli, ["walkforward", "--help"])
        assert result.exit_code == 0

    def test_validate_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_init_help(self, runner):
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0


class TestVersion:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "quant2026" in result.output.lower()


class TestValidate:
    def test_valid_config(self, runner, default_config):
        result = runner.invoke(cli, ["validate", "-c", default_config])
        assert result.exit_code == 0
        assert "有效" in result.output

    def test_missing_config(self, runner):
        result = runner.invoke(cli, ["validate", "-c", "/nonexistent/file.yaml"])
        assert result.exit_code != 0

    def test_invalid_config(self, runner):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("data:\n  start_date: '2025-12-31'\n  end_date: '2024-01-01'\n  stock_pool: []\n")
            f.flush()
            result = runner.invoke(cli, ["validate", "-c", f.name])
            assert result.exit_code != 0
        os.unlink(f.name)


class TestInit:
    def test_init_creates_file(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_config.yaml")
            result = runner.invoke(cli, ["init", "-o", out])
            assert result.exit_code == 0
            assert Path(out).exists()
            content = Path(out).read_text()
            assert "data" in content

    def test_init_overwrite_confirm(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_config.yaml")
            # Create first
            runner.invoke(cli, ["init", "-o", out])
            # Try overwrite, say no
            result = runner.invoke(cli, ["init", "-o", out], input="n\n")
            assert "取消" in result.output
