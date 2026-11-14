"""Lightweight, production-grade local web server for StressSense Interactive Dashboard.

Serves the pre-compiled standalone HTML5/Tailwind dashboard from results/dashboard/
with automatic port-finding, CORS support, and clean terminal shutdown.
"""

from functools import partial
import http.server
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Optional, Tuple, Union
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.generator import DashboardGenerator


class DashboardServer:
    """HTTP Server manager for the StressSense interactive dashboard."""

    DEFAULT_PORT = 8050
    PORT_SEARCH_RANGE = 20

    def __init__(
        self,
        dashboard_dir: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        """Initialize dashboard server."""
        self.dashboard_dir = Path(dashboard_dir) if dashboard_dir else PROJECT_ROOT / "results" / "dashboard"
        self.host = host
        self.preferred_port = port
        self.actual_port: Optional[int] = None
        self.httpd: Optional[http.server.HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def ensure_dashboard_built(self) -> Path:
        """Ensure that index.html exists, generating it if missing."""
        index_file = self.dashboard_dir / "index.html"
        if not index_file.is_file():
            print("Dashboard index.html not found. Building dashboard payload and HTML...")
            generator = DashboardGenerator()
            generator.generate_html(index_file)
        return index_file

    def find_available_port(self) -> int:
        """Find an available port starting from preferred_port."""
        for port in range(self.preferred_port, self.preferred_port + self.PORT_SEARCH_RANGE):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                result = sock.connect_ex((self.host, port))
                if result != 0:
                    # Port is free
                    return port
        raise RuntimeError(f"No available port found in range {self.preferred_port} - {self.preferred_port + self.PORT_SEARCH_RANGE}")

    def start(self, background: bool = False, open_browser: bool = False) -> str:
        """Start the dashboard HTTP server.

        Args:
            background: If True, runs the server on a daemon thread.
            open_browser: If True, opens the default web browser.

        Returns:
            The URL string where the dashboard is being served.
        """
        self.ensure_dashboard_built()
        self.actual_port = self.find_available_port()

        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(self.dashboard_dir))
        self.httpd = http.server.HTTPServer((self.host, self.actual_port), handler)
        url = f"http://{self.host}:{self.actual_port}/index.html"

        if background:
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            print(f"StressSense Dashboard server running in background at: {url}")
        else:
            print(f"===============================================================")
            print(f"🌾 StressSense Interactive Dashboard Server")
            print(f"Serving at: {url}")
            print(f"Press Ctrl+C to terminate.")
            print(f"===============================================================")
            if open_browser:
                webbrowser.open(url)
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down StressSense dashboard server...")
            finally:
                self.stop()

        if open_browser and background:
            webbrowser.open(url)

        return url

    def stop(self) -> None:
        """Stop the HTTP server cleanly."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
            print("Dashboard server stopped.")


def main():
    """CLI launcher entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch StressSense Interactive Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="Preferred port (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of dashboard HTML before serving")

    args = parser.parse_args()

    server = DashboardServer(host=args.host, port=args.port)
    if args.rebuild:
        print("Force rebuilding dashboard HTML...")
        DashboardGenerator().generate_html()

    server.start(background=False, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
