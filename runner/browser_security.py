"""
Browser Security Module for NIS Protocol Secure Runner
Provides secure browser automation with strict security controls
"""

import os
import tempfile
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Playwright removed for now - focus on Selenium

class SecureBrowserConfig:
    """Secure browser configuration for sandboxed execution"""
    
    # Security restrictions
    MAX_PAGE_LOAD_TIME = 30  # seconds
    MAX_EXECUTION_TIME = 60  # seconds
    MAX_PAGES_PER_SESSION = 10
    MAX_DOWNLOADS = 0  # No downloads allowed
    MAX_MEMORY_MB = 256  # Browser memory limit
    
    # Allowed domains (whitelist)
    ALLOWED_DOMAINS = {
        'httpbin.org',  # Testing
        'example.com',  # Testing
        'jsonplaceholder.typicode.com',  # API testing
        'httpstat.us',  # HTTP status testing
        'reqres.in',  # API testing
        'postman-echo.com'  # API testing
    }
    
    # Blocked domains (blacklist)
    BLOCKED_DOMAINS = {
        'localhost', '127.0.0.1', '0.0.0.0',  # Local access
        'internal', 'intranet',  # Internal networks
        'admin', 'administrator',  # Admin interfaces
        'file://', 'ftp://', 'ssh://'  # Non-HTTP protocols
    }
    
    # Security headers to enforce
    SECURITY_HEADERS = {
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'no-referrer'
    }

class SecureBrowser:
    """Secure browser wrapper with safety controls"""
    
    def __init__(self, browser_type: str = "chromium"):
        self.browser_type = browser_type
        self.session_start_time = None
        self.pages_visited = 0
        self.driver = None
        self.temp_dir = None
        
    def _create_secure_chrome_options(self) -> ChromeOptions:
        """Create secure Chrome options"""
        options = ChromeOptions()
        
        # Security options
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=NetworkService")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-field-trial-config")
        options.add_argument("--disable-ipc-flooding-protection")
        
        # Privacy and security
        options.add_argument("--incognito")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--disable-background-networking")
        
        # Performance limits
        options.add_argument(f"--memory-pressure-off")
        options.add_argument("--max_old_space_size=256")
        
        # Disable potentially dangerous features
        options.add_argument("--disable-file-system")
        options.add_argument("--disable-local-storage")
        options.add_argument("--disable-databases")
        options.add_argument("--disable-geolocation")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-camera")
        options.add_argument("--disable-microphone")
        
        # Create temporary profile directory
        self.temp_dir = tempfile.mkdtemp(prefix="secure_browser_")
        options.add_argument(f"--user-data-dir={self.temp_dir}")
        
        return options
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL against security policies"""
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Extract domain
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check blocked domains
        for blocked in SecureBrowserConfig.BLOCKED_DOMAINS:
            if blocked in domain:
                return False
        
        # If whitelist is enabled, check allowed domains
        if SecureBrowserConfig.ALLOWED_DOMAINS:
            domain_allowed = any(
                allowed in domain 
                for allowed in SecureBrowserConfig.ALLOWED_DOMAINS
            )
            if not domain_allowed:
                return False
        
        return True
    
    def start_session(self) -> bool:
        """Start a secure browser session"""
        try:
            if self.browser_type == "chrome":
                options = self._create_secure_chrome_options()
                service = Service()  # Use system chromedriver
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
            
            # Set timeouts
            self.driver.set_page_load_timeout(SecureBrowserConfig.MAX_PAGE_LOAD_TIME)
            self.driver.implicitly_wait(10)
            
            self.session_start_time = asyncio.get_event_loop().time()
            self.pages_visited = 0
            
            return True
            
        except Exception as e:
            print(f"Failed to start browser session: {e}")
            return False
    
    def navigate_to(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL with security checks"""
        if not self.driver:
            return {"success": False, "error": "No active browser session"}
        
        # Security validation
        if not self._validate_url(url):
            return {"success": False, "error": f"URL blocked by security policy: {url}"}
        
        # Check session limits
        if self.pages_visited >= SecureBrowserConfig.MAX_PAGES_PER_SESSION:
            return {"success": False, "error": "Maximum pages per session exceeded"}
        
        try:
            self.driver.get(url)
            self.pages_visited += 1
            
            # Get page info
            title = self.driver.title
            current_url = self.driver.current_url
            
            return {
                "success": True,
                "title": title,
                "url": current_url,
                "pages_visited": self.pages_visited
            }
            
        except TimeoutException:
            return {"success": False, "error": "Page load timeout"}
        except WebDriverException as e:
            return {"success": False, "error": f"Navigation failed: {str(e)}"}
    
    def get_page_source(self) -> Dict[str, Any]:
        """Get page source with size limits"""
        if not self.driver:
            return {"success": False, "error": "No active browser session"}
        
        try:
            source = self.driver.page_source
            
            # Limit source size for security
            max_size = 1024 * 1024  # 1MB
            if len(source) > max_size:
                source = source[:max_size] + "\n... [TRUNCATED FOR SECURITY]"
            
            return {
                "success": True,
                "source": source,
                "length": len(source)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to get page source: {str(e)}"}
    
    def find_elements(self, selector: str, by_type: str = "css") -> Dict[str, Any]:
        """Find elements with safety limits"""
        if not self.driver:
            return {"success": False, "error": "No active browser session"}
        
        try:
            # Map selector types
            by_map = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME
            }
            
            if by_type not in by_map:
                return {"success": False, "error": f"Unsupported selector type: {by_type}"}
            
            elements = self.driver.find_elements(by_map[by_type], selector)
            
            # Limit number of elements returned
            max_elements = 100
            if len(elements) > max_elements:
                elements = elements[:max_elements]
            
            element_data = []
            for elem in elements:
                try:
                    element_data.append({
                        "tag": elem.tag_name,
                        "text": elem.text[:500],  # Limit text length
                        "visible": elem.is_displayed(),
                        "enabled": elem.is_enabled()
                    })
                except:
                    continue
            
            return {
                "success": True,
                "elements": element_data,
                "count": len(element_data)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Element search failed: {str(e)}"}
    
    def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot with security controls"""
        if not self.driver:
            return {"success": False, "error": "No active browser session"}
        
        try:
            # Take screenshot as base64
            screenshot_b64 = self.driver.get_screenshot_as_base64()
            
            # Limit screenshot size
            max_size = 512 * 1024  # 512KB
            if len(screenshot_b64) > max_size:
                return {"success": False, "error": "Screenshot too large"}
            
            return {
                "success": True,
                "screenshot": screenshot_b64,
                "format": "base64_png"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Screenshot failed: {str(e)}"}
    
    def execute_script(self, script: str) -> Dict[str, Any]:
        """Execute JavaScript with restrictions"""
        if not self.driver:
            return {"success": False, "error": "No active browser session"}
        
        # Security validation of script
        dangerous_patterns = [
            'fetch(', 'XMLHttpRequest', 'eval(', 'Function(',
            'import(', 'require(', 'localStorage', 'sessionStorage',
            'document.cookie', 'navigator.', 'location.href',
            'window.open', 'alert(', 'confirm(', 'prompt('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in script:
                return {"success": False, "error": f"Script contains dangerous pattern: {pattern}"}
        
        # Limit script length
        if len(script) > 1000:
            return {"success": False, "error": "Script too long"}
        
        try:
            result = self.driver.execute_script(script)
            
            # Serialize result safely
            if result is None:
                result_data = None
            elif isinstance(result, (str, int, float, bool)):
                result_data = result
            elif isinstance(result, (list, dict)):
                result_data = json.dumps(result)[:1000]  # Limit size
            else:
                result_data = str(result)[:500]
            
            return {
                "success": True,
                "result": result_data
            }
            
        except Exception as e:
            return {"success": False, "error": f"Script execution failed: {str(e)}"}
    
    def close_session(self):
        """Close browser session and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        
        # Cleanup temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
            self.temp_dir = None
        
        self.session_start_time = None
        self.pages_visited = 0

# Global browser instance for reuse
_browser_instance = None

def get_secure_browser() -> SecureBrowser:
    """Get or create a secure browser instance"""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = SecureBrowser("chrome")
    return _browser_instance

def cleanup_browser():
    """Cleanup global browser instance"""
    global _browser_instance
    if _browser_instance:
        _browser_instance.close_session()
        _browser_instance = None
