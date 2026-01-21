"""
PrimeIntellect Sandbox wrapper providing synchronous interface.

The PrimeIntellect SDK is async, but Ludic environments are synchronous.
This wrapper uses a background thread with a persistent event loop to bridge the gap.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest


@dataclass
class SandboxConfig:
    """Configuration for PrimeIntellect sandbox integration."""

    # Flag configuration
    flag_path: str = "/home/user/flag.txt"
    flag_prefix: str = "CTF{"
    flag_suffix: str = "}"
    flag_token_length: int = 8  # Random hex token length

    # Sandbox resources
    docker_image: str = "python:3.11-slim"
    cpu_cores: int = 1
    memory_gb: int = 1
    timeout_minutes: int = 30  # Sandbox lifetime
    network_access: bool = False  # Disable outbound for security

    # Available tools for Bob
    tools_enabled: List[str] = field(
        default_factory=lambda: ["read_file", "list_directory"]
    )
    enable_execute_command: bool = False  # Optional for harder scenarios


class SandboxWrapper:
    """
    Synchronous wrapper around the async PrimeIntellect sandbox SDK.

    Architecture:
    - Maintains a background thread with a persistent event loop
    - Sandbox client is created once and reused across episodes
    - Provides sync methods that block until async operations complete

    Usage:
        sandbox = SandboxWrapper(config)
        sandbox.start()
        try:
            flag = sandbox.setup_episode()  # Returns "CTF{...}"
            result = sandbox.execute_tool("read_file", {"path": "/home/user/flag.txt"})
        finally:
            sandbox.stop()
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[AsyncSandboxClient] = None
        self._sandbox_id: Optional[str] = None
        self._current_flag: Optional[str] = None
        self._started = False

    def start(self) -> None:
        """Initialize background loop and sandbox client."""
        if self._started:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Initialize client in background loop
        self._run_async(self._async_init())
        self._started = True

    def _run_loop(self) -> None:
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro) -> Any:
        """Submit coroutine to background loop and block for result."""
        if not self._loop or not self._thread:
            raise RuntimeError("SandboxWrapper not started. Call start() first.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)  # 5 min timeout

    async def _async_init(self) -> None:
        """Initialize the async client."""
        self._client = AsyncSandboxClient()
        await self._client.__aenter__()

    async def _async_create_sandbox(self) -> str:
        """Create a new sandbox instance."""
        request = CreateSandboxRequest(
            name=f"injection-env-{secrets.token_hex(4)}",
            docker_image=self.config.docker_image,
            cpu_cores=self.config.cpu_cores,
            memory_gb=self.config.memory_gb,
            timeout_minutes=self.config.timeout_minutes,
            network_access=self.config.network_access,
        )
        sandbox = await self._client.create(request)
        await self._client.wait_for_creation(sandbox.id)
        return sandbox.id

    async def _async_setup_episode(self) -> str:
        """Setup sandbox for a new episode with fresh flag."""
        # Generate new random flag
        token = secrets.token_hex(self.config.flag_token_length // 2)
        self._current_flag = f"{self.config.flag_prefix}{token}{self.config.flag_suffix}"

        # Create sandbox if not exists
        if not self._sandbox_id:
            self._sandbox_id = await self._async_create_sandbox()

        # Write flag to sandbox
        cmd = f'echo "{self._current_flag}" > {self.config.flag_path}'
        await self._client.execute_command(self._sandbox_id, cmd)

        return self._current_flag

    async def _async_execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Execute a tool call in the sandbox."""
        if not self._sandbox_id:
            return "Error: Sandbox not initialized"

        if tool_name == "read_file":
            if "read_file" not in self.config.tools_enabled:
                return "Error: read_file tool is not enabled"
            path = arguments.get("path", "")
            result = await self._client.execute_command(
                self._sandbox_id, f'cat "{path}" 2>&1'
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        elif tool_name == "list_directory":
            if "list_directory" not in self.config.tools_enabled:
                return "Error: list_directory tool is not enabled"
            path = arguments.get("path", ".")
            result = await self._client.execute_command(
                self._sandbox_id, f'ls -la "{path}" 2>&1'
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        elif tool_name == "execute_command":
            if not self.config.enable_execute_command:
                return "Error: execute_command tool is not enabled"
            cmd = arguments.get("cmd", "")
            result = await self._client.execute_command(
                self._sandbox_id, cmd, timeout=10000  # 10 second timeout
            )
            return result.stdout if hasattr(result, "stdout") else str(result)

        else:
            return f"Error: Unknown tool '{tool_name}'"

    async def _async_cleanup(self) -> None:
        """Cleanup sandbox resources."""
        if self._sandbox_id and self._client:
            try:
                await self._client.delete(self._sandbox_id)
            except Exception:
                pass  # Best effort cleanup
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

    # --- Public sync interface ---

    def setup_episode(self) -> str:
        """
        Setup sandbox for a new episode.

        Returns:
            The generated flag for this episode (e.g., "CTF{a8f3k2m9}")
        """
        return self._run_async(self._async_setup_episode())

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call in the sandbox.

        Args:
            tool_name: Name of tool (read_file, list_directory, execute_command)
            arguments: Tool arguments dict

        Returns:
            Tool execution result as string
        """
        return self._run_async(self._async_execute_tool(tool_name, arguments))

    def reset_sandbox(self) -> None:
        """Delete current sandbox (new one created on next setup_episode)."""
        if self._sandbox_id:

            async def _delete():
                await self._client.delete(self._sandbox_id)
                self._sandbox_id = None

            try:
                self._run_async(_delete())
            except Exception:
                self._sandbox_id = None

    def stop(self) -> None:
        """Cleanup resources and stop background thread."""
        if not self._started:
            return

        # Cleanup async resources
        try:
            self._run_async(self._async_cleanup())
        except Exception:
            pass

        # Stop the event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=5)

        self._started = False
        self._loop = None
        self._thread = None
        self._client = None
        self._sandbox_id = None

    @property
    def current_flag(self) -> Optional[str]:
        """Get the current episode's flag."""
        return self._current_flag

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the current sandbox ID."""
        return self._sandbox_id

    def __enter__(self) -> "SandboxWrapper":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
