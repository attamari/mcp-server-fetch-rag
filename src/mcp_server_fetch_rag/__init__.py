import argparse
import asyncio
from .server import serve


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="MCP server for RAG-based web fetching")
    parser.add_argument(
        "--ignore-robots-txt",
        action="store_true",
        help="Ignore robots.txt restrictions",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        help="Custom User-Agent string",
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        help="Proxy URL for requests",
    )
    args = parser.parse_args()

    asyncio.run(serve(
        custom_user_agent=args.user_agent,
        ignore_robots_txt=args.ignore_robots_txt,
        proxy_url=args.proxy_url,
    ))


__all__ = ["main", "serve"]
