#!/usr/bin/env python3
"""
NIS Protocol Tool Runner HTTP Server
Persistent HTTP service for secure execution of Python scripts and shell commands
"""

import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import logging

from exec import SecureRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunnerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.runner = SecureRunner()
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handle POST requests for tool execution"""
        try:
            # Parse the request path
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/execute':
                self._handle_execute()
            elif parsed_path.path == '/health':
                self._handle_health()
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            self._send_error(500, str(e))

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self._handle_health()
        elif parsed_path.path == '/audit':
            self._handle_audit()
        else:
            self._send_error(404, "Endpoint not found")

    def _handle_execute(self):
        """Handle tool execution requests"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return
                
            body = self.rfile.read(content_length).decode('utf-8')
            request_data = json.loads(body)
            
            tool = request_data.get('tool')
            args = request_data.get('args', {})
            
            if not tool:
                self._send_error(400, "Tool type required")
                return
            
            # Execute the requested tool
            if tool == 'run_shell':
                cmd = args.get('cmd', '')
                timeout = args.get('timeout', 30)
                result = self.runner.run_shell(cmd, timeout)
                
            elif tool == 'run_python':
                filepath = args.get('filepath', '')
                script_args = args.get('args', [])
                timeout = args.get('timeout', 60)
                result = self.runner.run_python(filepath, script_args, timeout)
                
            elif tool == 'write_script':
                filename = args.get('filename', '')
                content = args.get('content', '')
                result = self.runner.write_script(filename, content)
                
            elif tool == 'read_file':
                filepath = args.get('filepath', '')
                result = self.runner.read_file(filepath)
                
            elif tool == 'list_files':
                path = args.get('path', '.')
                result = self.runner.list_files(path)
                
            elif tool == 'create_directory':
                path = args.get('path', '')
                result = self.runner.create_directory(path)
                
            elif tool == 'delete_file':
                filepath = args.get('filepath', '')
                result = self.runner.delete_file(filepath)
                
            elif tool == 'system_info':
                result = self.runner.get_system_info()
                
            elif tool == 'disk_usage':
                result = self.runner.get_disk_usage()
                
            elif tool == 'process_list':
                result = self.runner.get_process_list()
                
            elif tool == 'network_test':
                host = args.get('host', 'google.com')
                result = self.runner.test_network(host)
                
            elif tool == 'generate_report':
                result = self.runner.generate_system_report()
                
            elif tool == 'install_package':
                package = args.get('package', '')
                result = self.runner.install_python_package(package)
                
            elif tool == 'git_status':
                result = self.runner.git_status()
                
            elif tool == 'backup_workspace':
                result = self.runner.backup_workspace()
            
            # Advanced NIS Protocol tools
            elif tool == 'signal_analysis':
                data = args.get('data', '')
                analysis_type = args.get('analysis_type', 'frequency')
                result = self.runner.signal_analysis(data, analysis_type)
                
            elif tool == 'physics_validation':
                equation = args.get('equation', '')
                constraints = args.get('constraints', '')
                result = self.runner.physics_validation(equation, constraints)
                
            elif tool == 'document_analysis':
                filepath = args.get('filepath', '')
                analysis_mode = args.get('analysis_mode', 'structure')
                result = self.runner.document_analysis(filepath, analysis_mode)
                
            elif tool == 'performance_profiling':
                command = args.get('command', '')
                iterations = args.get('iterations', 1)
                result = self.runner.performance_profiling(command, iterations)
                
            elif tool == 'data_processing':
                operation = args.get('operation', '')
                # Pass all args to data_processing
                result = self.runner.data_processing(operation, **args)
                
            elif tool == 'code_analysis':
                filepath = args.get('filepath', '')
                analysis_type = args.get('analysis_type', 'quality')
                result = self.runner.code_analysis(filepath, analysis_type)
                
            else:
                self._send_error(400, f"Unknown tool: {tool}")
                return
            
            # Send response
            self._send_json_response(result)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Execute error: {e}")
            self._send_error(500, str(e))

    def _handle_health(self):
        """Handle health check requests"""
        health_data = {
            "status": "healthy",
            "service": "nis-runner",
            "version": "1.0",
            "timestamp": time.time(),
            "workspace": str(self.runner.workspace),
            "executions_count": len(self.runner.execution_log)
        }
        self._send_json_response(health_data)

    def _handle_audit(self):
        """Handle audit log requests"""
        audit_data = {
            "audit_log": self.runner.get_audit_log(),
            "total_executions": len(self.runner.execution_log),
            "timestamp": time.time()
        }
        self._send_json_response(audit_data)

    def _send_json_response(self, data, status_code=200):
        """Send JSON response"""
        response = json.dumps(data, indent=2)
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

    def _send_error(self, status_code, message):
        """Send error response"""
        error_data = {
            "error": message,
            "status_code": status_code,
            "timestamp": time.time()
        }
        self._send_json_response(error_data, status_code)

    def log_message(self, format, *args):
        """Override to use proper logging"""
        logger.info(f"{self.address_string()} - {format % args}")

def create_runner_server():
    """Create a shared runner instance for the server"""
    runner = SecureRunner()
    
    class SharedRunnerHandler(RunnerHandler):
        def __init__(self, *args, **kwargs):
            # Use shared runner instance
            self.runner = runner
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)
    
    return SharedRunnerHandler

def main():
    """Main server function"""
    host = '0.0.0.0'
    port = 8001
    
    handler_class = create_runner_server()
    server = HTTPServer((host, port), handler_class)
    
    logger.info(f"🔧 NIS Runner Server starting on {host}:{port}")
    logger.info("📊 Available endpoints:")
    logger.info("  POST /execute - Execute tools")
    logger.info("  GET  /health  - Health check")
    logger.info("  GET  /audit   - Audit log")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server shutting down...")
        server.shutdown()

if __name__ == "__main__":
    main()
