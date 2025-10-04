#!/usr/bin/env python3
"""
NIS Protocol Command Line Interface
"""

import click
import sys
from pathlib import Path

# Add src to path
_src_path = Path(__file__).parent.parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


@click.group()
@click.version_option(version="3.2.1", prog_name="nis-protocol")
def main():
    """
    NIS Protocol - Neuro-Inspired System Protocol
    
    A comprehensive autonomous AI framework with physics validation,
    multi-agent orchestration, and LLM integration.
    """
    pass


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def server(host, port, reload):
    """Start the NIS Protocol server"""
    click.echo(f"🚀 Starting NIS Protocol server on {host}:{port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        click.echo("❌ Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error starting server: {e}", err=True)
        sys.exit(1)


@main.command()
def info():
    """Display NIS Protocol information"""
    from nis_protocol import get_info
    
    info_data = get_info()
    
    click.echo("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║                      NIS Protocol Info                         ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    for key, value in info_data.items():
        click.echo(f"  {key.capitalize()}: {value}")
    
    click.echo("")


@main.command()
def quickstart():
    """Display quick start guide"""
    from nis_protocol import quick_start
    quick_start()


@main.command()
@click.argument('message')
@click.option('--provider', default=None, help='LLM provider to use')
def chat(message, provider):
    """Send a chat message (synchronous)"""
    try:
        from nis_protocol import NISCore
        
        nis = NISCore()
        response = nis.get_llm_response(message, provider=provider)
        
        click.echo(f"\n🤖 Response: {response.get('content', 'No response')}\n")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('message')
def autonomous(message):
    """Process message autonomously"""
    import asyncio
    
    try:
        from nis_protocol import NISCore
        
        async def process():
            nis = NISCore()
            result = await nis.process_autonomously(message)
            
            click.echo(f"\n🎯 Intent: {result['intent']}")
            click.echo(f"🔧 Tools Used: {', '.join(result['tools_used'])}")
            click.echo(f"💭 Reasoning: {result['reasoning']}")
            click.echo(f"\n🤖 Response: {result['response']}\n")
        
        asyncio.run(process())
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
def test():
    """Run quick system test"""
    click.echo("🧪 Running NIS Protocol system test...\n")
    
    try:
        from nis_protocol import NISCore
        import asyncio
        
        async def run_test():
            nis = NISCore()
            
            # Test 1: LLM response
            click.echo("1️⃣ Testing LLM response...")
            response = nis.get_llm_response("Hello, NIS!")
            if response and 'content' in response:
                click.echo("   ✅ LLM working")
            else:
                click.echo("   ⚠️  LLM response incomplete")
            
            # Test 2: Autonomous processing
            click.echo("\n2️⃣ Testing autonomous processing...")
            result = await nis.process_autonomously("What is 2+2?")
            if result and result.get('success'):
                click.echo("   ✅ Autonomous mode working")
                click.echo(f"   Intent detected: {result['intent']}")
                click.echo(f"   Tools used: {', '.join(result['tools_used'])}")
            else:
                click.echo("   ⚠️  Autonomous mode incomplete")
            
            click.echo("\n✅ System test complete!")
        
        asyncio.run(run_test())
        
    except Exception as e:
        click.echo(f"\n❌ Test failed: {e}", err=True)
        sys.exit(1)


@main.command()
def verify():
    """Verify installation and dependencies"""
    click.echo("🔍 Verifying NIS Protocol installation...\n")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "openai",
        "anthropic",
        "torch",
        "langchain",
        "langgraph"
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
            click.echo(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            click.echo(f"  ❌ {package} (missing)")
    
    click.echo(f"\n📊 Status: {len(installed)}/{len(required_packages)} packages installed")
    
    if missing:
        click.echo(f"\n⚠️  Missing packages: {', '.join(missing)}")
        click.echo("Run: pip install nis-protocol[full]")
        sys.exit(1)
    else:
        click.echo("\n✅ All dependencies installed!")


if __name__ == '__main__':
    main()

