#!/usr/bin/env python3
"""
NIS Protocol Server Runner
"""

import sys
from pathlib import Path

# Add src to path
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the NIS Protocol server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    try:
        import uvicorn
        
        print(f"""
        ╔════════════════════════════════════════════════════════════════╗
        ║                                                                ║
        ║                  NIS Protocol Server                           ║
        ║                                                                ║
        ╚════════════════════════════════════════════════════════════════╝
        
        🚀 Starting server on http://{host}:{port}
        
        Endpoints:
          - http://{host}:{port}/health
          - http://{host}:{port}/docs
          - http://{host}:{port}/console (Classic Chat)
          - http://{host}:{port}/modern-chat (Modern Chat)
          - http://{host}:{port}/chat/autonomous (Autonomous AI)
        
        Press CTRL+C to stop
        """)
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except ImportError:
        print("❌ Error: uvicorn not installed")
        print("Run: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()

