"""WebSocket server for broadcasting Qwen inference results."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional, Set

import websockets

LOGGER = logging.getLogger(__name__)


class ResultsServer:
    """WebSocket server that broadcasts Qwen inference results to connected clients."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._stopped = threading.Event()

    async def _broadcast(self, message: dict) -> None:
        """Broadcast a result to all connected clients."""
        if not self.clients:
            LOGGER.debug("No connected clients; discarding result")
            return

        message_json = json.dumps(message)
        dead_clients = set()

        for client in list(self.clients):
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                dead_clients.add(client)
            except OSError:
                LOGGER.debug("Error sending to client")
                dead_clients.add(client)

        # Clean up dead connections
        for client in dead_clients:
            self.clients.discard(client)

    async def _handle_client(self, websocket, _path: str | None = None) -> None:
        """Handle a new client connection."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        LOGGER.info("Client connected: %s (total: %d)", client_addr, len(self.clients))

        try:
            async for message in websocket:
                # Echo or handle client messages if needed
                LOGGER.debug("Received from client %s: %s", client_addr, message)
        except websockets.exceptions.ConnectionClosed:
            LOGGER.info("Client disconnected: %s", client_addr)
        except OSError:
            LOGGER.debug("Error with client %s", client_addr)
        finally:
            self.clients.discard(websocket)
            LOGGER.info("Client removed: %s (total: %d)", client_addr, len(self.clients))

    async def _serve_forever(self) -> None:
        """Create the websocket server and keep the loop alive."""
        LOGGER.info("Starting results server on ws://%s:%d", self.host, self.port)
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=None,
        )
        LOGGER.info("Results server listening on ws://%s:%d", self.host, self.port)
        self._started.set()

        await asyncio.Event().wait()

    def start_background(self) -> None:
        """Start the server on a dedicated background thread."""
        if self._thread and self._thread.is_alive():
            return

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.create_task(self._serve_forever())
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
                self._stopped.set()

        self._thread = threading.Thread(target=_run, name="results-server", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)

    def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._loop:
            return

        async def _shutdown() -> None:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                LOGGER.info("Results server stopped")

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        future.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def broadcast_result(self, result: dict) -> None:
        """Broadcast a result (non-async wrapper for sync code).
        
        Call this from sync code to queue the result for broadcast.
        """
        self.broadcast_result_async(result)

    def broadcast_result_async(self, result: dict) -> None:
        """Broadcast a result safely from any thread."""
        if not self._loop:
            LOGGER.debug("Results server loop not ready; dropping result")
            return

        asyncio.run_coroutine_threadsafe(self._broadcast(result), self._loop)


# Global server instance
_global_server: ResultsServer | None = None


def get_results_server() -> ResultsServer:
    """Get or create the global results server instance."""
    # pylint: disable=global-statement
    global _global_server
    if _global_server is None:
        _global_server = ResultsServer()
    return _global_server


def start_results_server() -> None:
    """Start the global results server in a background thread."""
    server = get_results_server()
    LOGGER.info("Starting results server...")
    server.start_background()
