"""
Pengu CLI entry point.

Usage:
    pengu                  — start Pengu backend
    pengu --port 8420      — custom port
    pengu --debug          — debug mode
    pengu hw               — hardware detection
    pengu config           — show configuration
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pengu",
        description="Pengu — ₹0-cost local-first autonomous desktop assistant",
    )
    sub = parser.add_subparsers(dest="command")

    # Default: serve
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", default=None, help="Path to pengu.yaml config file")

    # Sub-commands
    hw_parser = sub.add_parser("hw", help="Run hardware detection")
    config_parser = sub.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "hw":
        from pengu.hardware.cli import main as hw_main
        hw_main()
    elif args.command == "config":
        from pengu.config import load_config
        config = load_config(args.config)
        import json
        print(json.dumps(config.summary(), indent=2))
    else:
        # Start server
        import uvicorn
        from pengu.config import load_config
        from pengu.logging import setup_logging

        config = load_config(args.config)
        if args.debug:
            config.debug = True

        setup_logging(
            level="DEBUG" if config.debug else "INFO",
            json_output=False,
        )

        print(f"""
  +------------------------------------------+
  |           PENGU  v{config.version}              |
  |   0-cost Local-First Desktop Assistant   |
  +------------------------------------------+
  |  Cost Mode:  {config.cost_mode.value:<24s} |
  |  Host:       {config.api.host:<24s} |
  |  Port:       {config.api.port:<24d} |
  |  Debug:      {str(config.debug):<24s} |
  +------------------------------------------+
        """)

        uvicorn.run(
            "pengu.api:app",
            host=config.api.host,
            port=config.api.port,
            reload=args.debug or config.api.reload,
            log_level="debug" if config.debug else "info",
        )


if __name__ == "__main__":
    main()
