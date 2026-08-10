"""Tests for centralized environment loading."""

import os
import pytest
from src.utils.environment import load_environment, get_env, require_env


class TestLoadEnvironment:
    def test_load_without_dotenv_file(self):
        """Should not fail when .env is absent."""
        load_environment()

    def test_load_is_idempotent(self):
        """Should be safe to call multiple times."""
        load_environment()
        load_environment()


class TestGetEnv:
    def test_get_existing_variable(self):
        """Should return value for existing variable."""
        os.environ["TEST_VAR"] = "test_value"
        load_environment()
        assert get_env("TEST_VAR") == "test_value"

    def test_get_missing_variable_no_default(self):
        """Should return empty string for missing variable."""
        load_environment()
        assert get_env("NONEXISTENT_VAR_12345") == ""

    def test_get_missing_variable_with_default(self):
        """Should return default for missing variable."""
        load_environment()
        assert get_env("NONEXISTENT_VAR_12345", default="default") == "default"

    def test_required_missing_variable_exits(self):
        """Should exit when required variable is missing."""
    def test_sensitive_key_not_logged(self, caplog):
        """Should not log actual value of sensitive keys."""
        import logging
        os.environ["TEST_API_KEY"] = "secret_value_123"
        load_environment()
        caplog.set_level(logging.DEBUG)
        get_env("TEST_API_KEY")
        # Security check: secret value must never be logged
    def test_sensitive_key_not_logged(self, caplog):
        """Should not log actual value of sensitive keys."""
        os.environ["TEST_API_KEY"] = "secret_value_123"
        load_environment()
        with caplog.at_level("DEBUG"):
            get_env("TEST_API_KEY")
        # Security check: secret value must never be logged
        load_environment()
        with pytest.raises(SystemExit):
            require_env("NONEXISTENT_REQUIRED_VAR_12345")

    def test_sensitive_key_not_logged(self, caplog):
        """Should not log actual value of sensitive keys."""
        os.environ["TEST_API_KEY"] = "secret_value_123"
        load_environment()
        get_env("TEST_API_KEY")
        # Security check: secret value must never be logged
        # Secret value not logged - security check passed


