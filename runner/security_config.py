"""
Security Configuration for NIS Protocol Secure Runner
Defines security policies and restrictions for safe code execution
"""

import os
from typing import Set, Dict, List

class SecurityConfig:
    """Comprehensive security configuration for the code runner"""
    
    # === IMPORT RESTRICTIONS ===
    BLOCKED_IMPORTS: Set[str] = {
        # System access
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        
        # Network access (requests allowed for web scraping)
        'socket', 'urllib', 'http', 'https',
        'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib',
        'xmlrpc', 'email', 'mimetypes',
        
        # File system
        'tempfile', 'glob', 'fnmatch', 'linecache',
        'fileinput', 'filecmp', 'tarfile', 'zipfile',
        
        # Process control
        'multiprocessing', 'threading', 'asyncio', 'concurrent',
        'queue', 'sched', 'signal', 'atexit',
        
        # Dynamic execution
        'ctypes', 'marshal', 'pickle', 'dill', 'joblib',
        'importlib', 'pkgutil', 'modulefinder',
        
        # Reflection/introspection
        'inspect', 'gc', 'weakref', 'copy', 'types',
        
        # Database access
        'sqlite3', 'dbm', 'shelve',
        
        # Cryptography (potentially dangerous)
        'hashlib', 'hmac', 'secrets', 'ssl', 'crypt',
        
        # Platform specific
        'platform', 'getpass', 'pwd', 'grp', 'termios',
        'tty', 'pty', 'fcntl', 'pipes', 'resource',
        
        # Development tools
        'pdb', 'trace', 'traceback', 'warnings',
        'unittest', 'doctest', 'profile', 'pstats',
        
        # External packages that could be dangerous
        'numpy.ctypeslib', 'scipy.weave', 'IPython',
        'jupyter', 'notebook', 'tornado'
    }
    
    # === ALLOWED BUILTINS ===
    SAFE_BUILTINS: Set[str] = {
        # Basic types
        'bool', 'int', 'float', 'complex', 'str', 'bytes', 'bytearray',
        
        # Collections
        'list', 'tuple', 'dict', 'set', 'frozenset',
        
        # Iteration
        'range', 'enumerate', 'zip', 'map', 'filter', 'reversed',
        
        # Math
        'abs', 'round', 'pow', 'divmod', 'sum', 'min', 'max',
        
        # Type checking
        'type', 'isinstance', 'issubclass',
        
        # Attribute access (restricted)
        'hasattr', 'getattr', 'setattr', 'delattr',
        
        # String operations
        'repr', 'str', 'format', 'ascii', 'ord', 'chr',
        
        # Container operations
        'len', 'sorted', 'all', 'any',
        
        # Object creation
        'object', 'property', 'classmethod', 'staticmethod',
        
        # Safe I/O
        'print',  # Output only, no input functions
        
        # Exceptions
        'Exception', 'ValueError', 'TypeError', 'IndexError',
        'KeyError', 'AttributeError', 'RuntimeError'
    }
    
    # === ALLOWED MODULES ===
    SAFE_MODULES: Dict[str, List[str]] = {
        'math': ['*'],  # All math functions are safe
        'random': ['random', 'randint', 'choice', 'shuffle', 'sample'],
        'datetime': ['datetime', 'date', 'time', 'timedelta'],
        'json': ['loads', 'dumps', 'load', 'dump'],
        'string': ['ascii_letters', 'digits', 'punctuation'],
        'itertools': ['*'],  # Iterator tools are generally safe
        'functools': ['reduce', 'partial', 'wraps'],
        'operator': ['*'],  # Operator functions are safe
        'collections': ['Counter', 'defaultdict', 'deque', 'namedtuple'],
        're': ['compile', 'search', 'match', 'findall', 'split', 'sub'],
        # Browser automation (restricted)
        'browser_security': ['get_secure_browser', 'SecureBrowser'],
        'beautifulsoup4': ['BeautifulSoup'],
        'bs4': ['BeautifulSoup'],
        'requests': ['get', 'post', 'put', 'delete', 'head', 'options'],
    }
    
    # === FILE SYSTEM RESTRICTIONS ===
    ALLOWED_EXTENSIONS: Set[str] = {
        '.py', '.txt', '.json', '.csv', '.md', '.yaml', '.yml',
        '.xml', '.html', '.css', '.js', '.sql'
    }
    
    BLOCKED_EXTENSIONS: Set[str] = {
        '.exe', '.bat', '.sh', '.cmd', '.ps1', '.vbs',
        '.dll', '.so', '.dylib', '.bin', '.com'
    }
    
    # File size limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_TOTAL_FILES: int = 100
    
    # === EXECUTION LIMITS ===
    MAX_EXECUTION_TIME: int = 30  # seconds
    MAX_MEMORY_MB: int = 512  # MB
    MAX_CPU_PERCENT: int = 80  # CPU usage limit
    MAX_CONCURRENT_EXECUTIONS: int = 5
    
    # === BROWSER SECURITY ===
    BROWSER_ENABLED: bool = True
    MAX_BROWSER_SESSIONS: int = 2
    MAX_PAGES_PER_SESSION: int = 10
    BROWSER_TIMEOUT: int = 30  # seconds
    ALLOWED_BROWSER_DOMAINS: Set[str] = {
        'httpbin.org', 'example.com', 'jsonplaceholder.typicode.com',
        'httpstat.us', 'reqres.in', 'postman-echo.com'
    }
    
    # === DANGEROUS PATTERNS ===
    DANGEROUS_PATTERNS: List[str] = [
        # Direct function calls
        'eval(', 'exec(', 'compile(', '__import__(',
        
        # File operations
        'open(', 'file(', 'input(', 'raw_input(',
        
        # Introspection
        'globals(', 'locals(', 'vars(', 'dir(',
        
        # Magic methods
        '__class__', '__bases__', '__subclasses__',
        '__import__', '__builtins__',
        
        # System access
        'system(', 'popen(', 'spawn(',
        
        # Network patterns
        'connect(', 'bind(', 'listen(', 'accept(',
        'send(', 'recv(', 'sendto(', 'recvfrom(',
    ]
    
    # === RESOURCE MONITORING ===
    MONITOR_INTERVALS: Dict[str, int] = {
        'memory_check': 1,    # seconds
        'cpu_check': 1,       # seconds
        'timeout_check': 0.1, # seconds
    }
    
    # === LOGGING CONFIGURATION ===
    LOG_EXECUTION_DETAILS: bool = True
    LOG_SECURITY_VIOLATIONS: bool = True
    LOG_RESOURCE_USAGE: bool = True
    
    # === SANDBOX POLICIES ===
    ENABLE_NETWORK: bool = False
    ENABLE_FILE_WRITE: bool = True  # Only to workspace
    ENABLE_SUBPROCESS: bool = False
    ENABLE_IMPORT_RESTRICTION: bool = True
    
    @classmethod
    def is_import_allowed(cls, module_name: str) -> bool:
        """Check if a module import is allowed"""
        return module_name not in cls.BLOCKED_IMPORTS
    
    @classmethod
    def is_builtin_allowed(cls, builtin_name: str) -> bool:
        """Check if a builtin function is allowed"""
        return builtin_name in cls.SAFE_BUILTINS
    
    @classmethod
    def is_file_extension_allowed(cls, extension: str) -> bool:
        """Check if a file extension is allowed"""
        return (extension.lower() in cls.ALLOWED_EXTENSIONS and 
                extension.lower() not in cls.BLOCKED_EXTENSIONS)
    
    @classmethod
    def get_security_violations(cls, code: str) -> List[str]:
        """Analyze code for security violations"""
        violations = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check imports
            if line_stripped.startswith(('import ', 'from ')):
                # Extract module names more precisely
                if line_stripped.startswith('import '):
                    # Handle: import module, import module as alias
                    import_part = line_stripped[7:].split(' as ')[0].strip()
                    modules = [m.strip() for m in import_part.split(',')]
                elif line_stripped.startswith('from '):
                    # Handle: from module import ...
                    from_part = line_stripped[5:].split(' import ')[0].strip()
                    modules = [from_part]
                else:
                    modules = []
                
                for module in modules:
                    if module in cls.BLOCKED_IMPORTS:
                        violations.append(
                            f"Line {line_num}: Blocked import '{module}'"
                        )
            
            # Check dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                if pattern in line_stripped:
                    violations.append(
                        f"Line {line_num}: Dangerous pattern '{pattern}'"
                    )
        
        return violations
    
    @classmethod
    def get_safe_builtins_dict(cls) -> Dict[str, any]:
        """Get a dictionary of safe builtins for restricted execution"""
        import builtins
        safe_dict = {}
        
        for name in cls.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_dict[name] = getattr(builtins, name)
        
        return safe_dict
