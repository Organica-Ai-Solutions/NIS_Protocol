#!/usr/bin/env python3
"""
NIS Protocol CLI - Main Entry Point
===================================

Central command-line interface for NIS Protocol platform management.

Usage:
    nis --help
    nis init my-project
    nis deploy edge --device raspberry-pi
    nis serve --port 8000
"""

import click
import sys
from typing import Optional

from .. import __version__, welcome, platform_info, health_check


@click.group()
@click.version_option(version=__version__, prog_name="NIS Protocol")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]):
    """
    NIS Protocol - AI Development Platform & SDK
    
    The foundational AI operating system for edge devices, autonomous systems, and smart infrastructure.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        click.echo(f"🧠 NIS Protocol v{__version__}")


@cli.command()
@click.argument('project_name')
@click.option('--template', '-t', default='basic', 
              type=click.Choice(['basic', 'edge', 'drone', 'robot', 'city', 'industrial']),
              help='Project template to use')
@click.option('--path', '-p', default='.', help='Project directory path')
@click.pass_context
def init(ctx: click.Context, project_name: str, template: str, path: str):
    """Initialize a new NIS Protocol project."""
    from .init import init_project
    
    click.echo(f"🚀 Initializing NIS Protocol project: {project_name}")
    click.echo(f"📁 Template: {template}")
    click.echo(f"📂 Path: {path}")
    
    try:
        success = init_project(project_name, template, path, verbose=ctx.obj.get('verbose', False))
        if success:
            click.echo(f"✅ Project '{project_name}' created successfully!")
            click.echo(f"\nNext steps:")
            click.echo(f"  cd {project_name}")
            click.echo(f"  nis serve")
        else:
            click.echo("❌ Failed to create project")
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', default=8000, help='Port to bind to')
@click.option('--workers', '-w', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, workers: int, reload: bool):
    """Start the NIS Protocol platform server."""
    from .serve import start_server
    
    click.echo(f"🚀 Starting NIS Protocol server...")
    click.echo(f"🌐 Host: {host}:{port}")
    click.echo(f"👥 Workers: {workers}")
    
    try:
        start_server(
            host=host, 
            port=port, 
            workers=workers, 
            reload=reload,
            verbose=ctx.obj.get('verbose', False)
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Server error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('target', type=click.Choice(['edge', 'cloud', 'hybrid', 'docker', 'k8s']))
@click.option('--device', help='Target device type (for edge deployment)')
@click.option('--provider', help='Cloud provider (for cloud deployment)')
@click.option('--config-file', help='Deployment configuration file')
@click.pass_context
def deploy(ctx: click.Context, target: str, device: Optional[str], provider: Optional[str], config_file: Optional[str]):
    """Deploy NIS Protocol to target environment."""
    from .deploy import deploy_platform
    
    click.echo(f"🚀 Deploying to {target}...")
    
    deploy_config = {}
    if device:
        deploy_config['device_type'] = device
    if provider:
        deploy_config['provider'] = provider
    if config_file:
        deploy_config['config_file'] = config_file
    
    try:
        success = deploy_platform(
            target=target,
            config=deploy_config,
            verbose=ctx.obj.get('verbose', False)
        )
        if success:
            click.echo(f"✅ Deployment to {target} successful!")
        else:
            click.echo(f"❌ Deployment to {target} failed")
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Deployment error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--format', '-f', default='table', 
              type=click.Choice(['table', 'json', 'yaml']),
              help='Output format')
def status(format: str):
    """Show platform status and health information."""
    import json as json_lib
    import yaml
    from tabulate import tabulate
    
    try:
        health = health_check()
        platform = platform_info()
        
        if format == 'json':
            click.echo(json_lib.dumps({"health": health, "platform": platform}, indent=2))
        elif format == 'yaml':
            click.echo(yaml.dump({"health": health, "platform": platform}, default_flow_style=False))
        else:
            # Table format
            click.echo(f"🧠 NIS Protocol v{platform['version']} Status")
            click.echo("=" * 50)
            
            # Platform info
            info_data = [
                ["Name", platform['name']],
                ["Version", platform['version']],
                ["License", platform['license']],
                ["Status", "✅ Healthy" if health['platform_available'] else "❌ Issues"],
            ]
            click.echo("\n📊 Platform Information:")
            click.echo(tabulate(info_data, headers=["Property", "Value"], tablefmt="grid"))
            
            # Core agents status
            agents_data = []
            for agent, available in health['core_agents'].items():
                status_icon = "✅" if available else "❌"
                agents_data.append([agent.title(), f"{status_icon} {'Available' if available else 'Missing'}"])
            
            click.echo("\n🤖 Core Agents:")
            click.echo(tabulate(agents_data, headers=["Agent", "Status"], tablefmt="grid"))
            
            # Protocols status
            protocols_data = []
            for protocol, available in health['protocols'].items():
                status_icon = "✅" if available else "❌"
                protocols_data.append([protocol.upper(), f"{status_icon} {'Available' if available else 'Missing'}"])
            
            click.echo("\n🔌 Protocol Support:")
            click.echo(tabulate(protocols_data, headers=["Protocol", "Status"], tablefmt="grid"))
            
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        sys.exit(1)


@cli.command()
def info():
    """Show comprehensive platform information."""
    platform = platform_info()
    
    click.echo(f"🧠 {platform['name']} v{platform['version']}")
    click.echo("=" * 60)
    click.echo(f"📝 {platform['description']}")
    click.echo(f"👤 Author: {platform['author']}")
    click.echo(f"📄 License: {platform['license']}")
    
    click.echo("\n🎯 Capabilities:")
    for capability, enabled in platform['capabilities'].items():
        icon = "✅" if enabled else "❌"
        click.echo(f"  {icon} {capability.replace('_', ' ').title()}")
    
    click.echo("\n📱 Supported Devices:")
    for device in platform['supported_devices']:
        click.echo(f"  • {device}")
    
    click.echo("\n🏭 Use Cases:")
    for use_case in platform['use_cases']:
        click.echo(f"  • {use_case}")


@cli.command()
def welcome_msg():
    """Show welcome message for new users."""
    welcome()


# Add subcommands from other modules
@cli.group()
def agent():
    """Agent management commands."""
    pass


@cli.group()
def edge():
    """Edge device management commands."""
    pass


@cli.group()
def config():
    """Configuration management commands."""
    pass


def main():
    """Main CLI entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
