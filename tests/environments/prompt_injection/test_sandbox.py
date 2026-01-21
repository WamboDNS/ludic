"""
Tests for sandbox wrapper functionality.

Note: These tests use mocks and don't require actual PrimeIntellect SDK.
"""

import pytest
import sys
sys.path.insert(0, "/Users/denis/research/ludic")

from environments.prompt_injection.sandbox import SandboxConfig, SandboxWrapper
from conftest import MockSandboxWrapper


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_default_values(self):
        """Default configuration should have sensible values."""
        config = SandboxConfig()
        assert config.flag_path is not None
        assert len(config.tools_enabled) > 0
        assert "read_file" in config.tools_enabled

    def test_custom_flag_path(self):
        """Custom flag path should be respected."""
        config = SandboxConfig(flag_path="/custom/path/flag.txt")
        assert config.flag_path == "/custom/path/flag.txt"

    def test_custom_tools(self):
        """Custom tools list should be respected."""
        config = SandboxConfig(tools_enabled=["read_file"])
        assert config.tools_enabled == ["read_file"]
        assert "list_directory" not in config.tools_enabled

    def test_execute_command_disabled_by_default(self):
        """Command execution should be disabled by default for safety."""
        config = SandboxConfig()
        assert config.enable_execute_command is False

    def test_enable_execute_command(self):
        """Command execution can be enabled."""
        config = SandboxConfig(enable_execute_command=True)
        assert config.enable_execute_command is True


class TestMockSandboxWrapper:
    """Tests for MockSandboxWrapper (used in other tests)."""

    def test_start_stop(self, mock_sandbox):
        """Sandbox should track start/stop state."""
        # Already started by fixture
        assert mock_sandbox._started is True
        mock_sandbox.stop()
        assert mock_sandbox._started is False
        mock_sandbox.start()
        assert mock_sandbox._started is True

    def test_setup_episode_generates_flag(self, mock_sandbox):
        """setup_episode should generate a CTF flag."""
        flag = mock_sandbox.setup_episode()
        assert flag.startswith("CTF{")
        assert flag.endswith("}")
        assert len(flag) > 5  # CTF{} + some content

    def test_setup_episode_unique_flags(self, mock_sandbox):
        """Each episode should get a unique flag."""
        flags = [mock_sandbox.setup_episode() for _ in range(10)]
        assert len(set(flags)) == 10  # All unique

    def test_current_flag_property(self, mock_sandbox):
        """current_flag should return the active flag."""
        flag = mock_sandbox.setup_episode()
        assert mock_sandbox.current_flag == flag

    def test_execute_read_file(self, mock_sandbox):
        """read_file tool should work."""
        mock_sandbox.setup_episode()
        result = mock_sandbox.execute_tool("read_file", {"path": "/home/user/flag.txt"})
        assert mock_sandbox.current_flag in result

    def test_execute_read_file_non_flag(self, mock_sandbox):
        """read_file on non-flag path should return contents."""
        result = mock_sandbox.execute_tool("read_file", {"path": "/home/user/readme.txt"})
        assert "Contents of" in result

    def test_execute_list_directory(self, mock_sandbox):
        """list_directory tool should work."""
        result = mock_sandbox.execute_tool("list_directory", {"path": "."})
        assert "file1.txt" in result
        assert "flag.txt" in result

    def test_execute_unknown_tool(self, mock_sandbox):
        """Unknown tool should return error message."""
        result = mock_sandbox.execute_tool("delete_file", {"path": "/"})
        assert "Unknown tool" in result

    def test_context_manager(self):
        """Sandbox should work as context manager."""
        sandbox = MockSandboxWrapper()
        assert sandbox._started is False

        with sandbox:
            assert sandbox._started is True
            flag = sandbox.setup_episode()
            assert flag.startswith("CTF{")

        assert sandbox._started is False


class TestSandboxWrapperIntegration:
    """Integration-style tests for SandboxWrapper.

    These tests verify the interface but use mocks for the actual SDK.
    """

    def test_config_stored(self):
        """Config should be stored on wrapper."""
        config = SandboxConfig(flag_path="/test/flag.txt")
        sandbox = MockSandboxWrapper(config)
        assert sandbox.config.flag_path == "/test/flag.txt"

    def test_tool_execution_interface(self, mock_sandbox):
        """Tool execution should follow expected interface."""
        # Setup
        mock_sandbox.setup_episode()

        # Execute tool with dict arguments
        result = mock_sandbox.execute_tool(
            tool_name="read_file",
            arguments={"path": "/some/file.txt"},
        )

        # Should return string result
        assert isinstance(result, str)

    def test_flag_in_read_file_response(self, mock_sandbox):
        """Reading flag file should return the flag."""
        flag = mock_sandbox.setup_episode()

        # Read the flag file
        result = mock_sandbox.execute_tool(
            "read_file",
            {"path": "/home/user/flag.txt"},
        )

        assert flag in result

    def test_flag_persistence_within_episode(self, mock_sandbox):
        """Flag should remain consistent within an episode."""
        flag = mock_sandbox.setup_episode()

        # Multiple reads should return same flag
        for _ in range(5):
            result = mock_sandbox.execute_tool(
                "read_file",
                {"path": "/flag.txt"},
            )
            assert flag in result

    def test_flag_changes_between_episodes(self, mock_sandbox):
        """Flag should change between episodes."""
        flag1 = mock_sandbox.setup_episode()
        flag2 = mock_sandbox.setup_episode()

        assert flag1 != flag2
