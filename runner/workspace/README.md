# NIS Protocol Secure Code Execution Workspace

This is a secure sandbox environment for running untrusted code safely.

## Features

- **Restricted Python execution** using RestrictedPython
- **Resource limits** (memory, CPU, execution time)
- **Import restrictions** - dangerous modules blocked
- **File system isolation** - limited to workspace directory
- **Network isolation** - no external network access
- **Process isolation** - no subprocess creation

## Security Measures

1. **Import Filtering**: Dangerous modules like `os`, `sys`, `subprocess` are blocked
2. **Builtin Restrictions**: Only safe builtin functions are available
3. **Resource Monitoring**: Memory and CPU usage monitored and limited
4. **Execution Timeout**: Code execution has strict time limits
5. **File Type Validation**: Only safe file types can be uploaded
6. **Code Pattern Analysis**: Dangerous code patterns are detected

## Usage

Send code execution requests to the runner API:

```python
{
    "code": "print('Hello, secure world!')",
    "language": "python",
    "timeout": 30,
    "memory_limit": 512
}
```

## Workspace

This directory is where uploaded files are stored and where code execution takes place.
Files are isolated from the main system and other executions.
