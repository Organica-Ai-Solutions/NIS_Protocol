#!/usr/bin/env python3
"""
NIS Protocol Server
==================

Start and manage the NIS Protocol platform server.
"""

import asyncio
import uvicorn
import click
from typing import Optional


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    verbose: bool = False
):
    """
    Start the NIS Protocol platform server.
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        workers: Number of worker processes
        reload: Enable auto-reload for development
        verbose: Enable verbose logging
    """
    
    # Configure logging level
    log_level = "debug" if verbose else "info"
    
    # Server configuration
    config = uvicorn.Config(
        app="nis_protocol.server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload only works with single worker
        reload=reload,
        log_level=log_level,
        access_log=verbose
    )
    
    # Start server
    server = uvicorn.Server(config)
    
    click.echo(f"🚀 Starting NIS Protocol server...")
    click.echo(f"🌐 Server: http://{host}:{port}")
    click.echo(f"📊 API Docs: http://{host}:{port}/docs")
    click.echo(f"👥 Workers: {workers}")
    click.echo(f"🔄 Reload: {'enabled' if reload else 'disabled'}")
    
    if reload:
        click.echo("⚠️  Development mode - use single worker for reload")
    
    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped by user")
    except Exception as e:
        click.echo(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    @click.command()
    @click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
    @click.option('--port', '-p', default=8000, help='Port to bind to')
    @click.option('--workers', '-w', default=1, help='Number of worker processes')
    @click.option('--reload', is_flag=True, help='Enable auto-reload for development')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
    def main(host: str, port: int, workers: int, reload: bool, verbose: bool):
        """Start the NIS Protocol platform server."""
        start_server(host, port, workers, reload, verbose)
    
    main()
