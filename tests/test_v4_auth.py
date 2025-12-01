"""
Test V4.0 Authentication & User Management
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.security.user_management import UserManager, DEV_MASTER_KEY


class TestAuthentication:
    """Test authentication endpoints"""

    def setup_method(self):
        """Create fresh user manager for each test"""
        self.manager = UserManager(storage_path="data/test_users")

    def teardown_method(self):
        """Cleanup test data"""
        import shutil
        if os.path.exists("data/test_users"):
            shutil.rmtree("data/test_users")

    def test_signup_success(self):
        """Test successful user signup"""
        result = self.manager.signup(
            email="test@example.com",
            password="securepass123",
            name="Test User"
        )
        assert result["success"] is True
        assert "token" in result
        assert result["user"]["email"] == "test@example.com"
        assert result["user"]["name"] == "Test User"
        assert result["user"]["role"] == "user"

    def test_signup_duplicate_email(self):
        """Test signup with existing email fails"""
        self.manager.signup("test@example.com", "pass1", "User 1")
        result = self.manager.signup("test@example.com", "pass2", "User 2")
        assert result["success"] is False
        assert "already registered" in result["error"].lower()

    def test_login_success(self):
        """Test successful login"""
        self.manager.signup("login@test.com", "mypassword", "Login User")
        result = self.manager.login("login@test.com", "mypassword")
        assert result["success"] is True
        assert "token" in result

    def test_login_wrong_password(self):
        """Test login with wrong password fails"""
        self.manager.signup("wrong@test.com", "correctpass", "Wrong Pass User")
        result = self.manager.login("wrong@test.com", "wrongpass")
        assert result["success"] is False

    def test_login_nonexistent_user(self):
        """Test login with non-existent user fails"""
        result = self.manager.login("nobody@test.com", "anypass")
        assert result["success"] is False

    def test_dev_bypass(self):
        """Test developer bypass authentication"""
        result = self.manager.login("any@email.com", DEV_MASTER_KEY)
        assert result["success"] is True
        assert result["user"]["role"] == "admin"
        assert result["user"]["id"] == "dev_master"

    def test_token_verification(self):
        """Test token verification"""
        signup_result = self.manager.signup("verify@test.com", "pass", "Verify User")
        token = signup_result["token"]
        
        user = self.manager.verify_token(token)
        assert user is not None
        assert user["email"] == "verify@test.com"

    def test_invalid_token(self):
        """Test invalid token returns None"""
        user = self.manager.verify_token("invalid_token_12345")
        assert user is None

    def test_logout(self):
        """Test logout invalidates token"""
        signup_result = self.manager.signup("logout@test.com", "pass", "Logout User")
        token = signup_result["token"]
        
        # Token valid before logout
        assert self.manager.verify_token(token) is not None
        
        # Logout
        self.manager.logout(token)
        
        # Token invalid after logout
        assert self.manager.verify_token(token) is None

    def test_password_recovery(self):
        """Test password recovery request"""
        self.manager.signup("recover@test.com", "pass", "Recover User")
        result = self.manager.recover_password("recover@test.com")
        assert result["success"] is True

    def test_token_refresh(self):
        """Test token refresh"""
        signup_result = self.manager.signup("refresh@test.com", "pass", "Refresh User")
        old_token = signup_result["token"]
        
        result = self.manager.refresh_token(old_token)
        assert result["success"] is True
        assert result["token"] != old_token
        
        # Old token should be invalid
        assert self.manager.verify_token(old_token) is None
        # New token should be valid
        assert self.manager.verify_token(result["token"]) is not None


class TestAPIKeys:
    """Test API key management"""

    def setup_method(self):
        self.manager = UserManager(storage_path="data/test_users")
        result = self.manager.signup("apikey@test.com", "pass", "API Key User")
        self.user_id = result["user"]["id"]
        self.token = result["token"]

    def teardown_method(self):
        import shutil
        if os.path.exists("data/test_users"):
            shutil.rmtree("data/test_users")

    def test_create_api_key(self):
        """Test API key creation"""
        result = self.manager.create_api_key(self.user_id, "Test Key")
        assert result["success"] is True
        assert "key" in result
        assert result["key"].startswith("nis_")
        assert result["name"] == "Test Key"

    def test_verify_api_key(self):
        """Test API key verification"""
        create_result = self.manager.create_api_key(self.user_id, "Verify Key")
        api_key = create_result["key"]
        
        user = self.manager.verify_api_key(api_key)
        assert user is not None
        assert user["email"] == "apikey@test.com"

    def test_delete_api_key(self):
        """Test API key deletion"""
        create_result = self.manager.create_api_key(self.user_id, "Delete Key")
        key_id = create_result["id"]
        api_key = create_result["key"]
        
        # Key works before deletion
        assert self.manager.verify_api_key(api_key) is not None
        
        # Delete key
        self.manager.delete_api_key(self.user_id, key_id)
        
        # Key doesn't work after deletion
        assert self.manager.verify_api_key(api_key) is None

    def test_invalid_api_key(self):
        """Test invalid API key returns None"""
        user = self.manager.verify_api_key("nis_invalid_key_12345")
        assert user is None


class TestUserProfile:
    """Test user profile management"""

    def setup_method(self):
        self.manager = UserManager(storage_path="data/test_users")
        result = self.manager.signup("profile@test.com", "pass", "Profile User")
        self.user_id = result["user"]["id"]

    def teardown_method(self):
        import shutil
        if os.path.exists("data/test_users"):
            shutil.rmtree("data/test_users")

    def test_get_user(self):
        """Test getting user profile"""
        user = self.manager.get_user(self.user_id)
        assert user is not None
        assert user["email"] == "profile@test.com"
        assert user["name"] == "Profile User"

    def test_update_user_name(self):
        """Test updating user name"""
        result = self.manager.update_user(self.user_id, {"name": "New Name"})
        assert result["success"] is True
        assert result["user"]["name"] == "New Name"

    def test_update_user_settings(self):
        """Test updating user settings"""
        result = self.manager.update_user(self.user_id, {
            "settings": {"theme": "dark", "notifications": True}
        })
        assert result["success"] is True

    def test_get_usage(self):
        """Test getting user usage stats"""
        usage = self.manager.get_usage(self.user_id)
        assert "api_calls" in usage
        assert "tokens_used" in usage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
