"""Oracle test configuration."""

import os
from functools import lru_cache
from pathlib import Path

import oracledb
import pytest
from dotenv import load_dotenv

# Load .env file first
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Default Oracle connection info - prioritize .env values
oracle_dsn = os.getenv("ORACLE_DSN")
if oracle_dsn:
    # Use DSN format (for Oracle Cloud with SSL)
    DEFAULT_CONNECTION_INFO = {
        "user": os.getenv("ORACLE_USERNAME", "testuser"),
        "password": os.getenv("ORACLE_PASSWORD", "testpass"),
        "dsn": oracle_dsn,
    }
else:
    # Fallback: construct simple DSN from individual parameters (for local Oracle)
    host = os.getenv("ORACLE_HOST", "localhost")
    port = os.getenv("ORACLE_PORT", "1521")
    service_name = os.getenv("ORACLE_SERVICE_NAME", "FREEPDB1")
    simple_dsn = f"{host}:{port}/{service_name}"

    DEFAULT_CONNECTION_INFO = {
        "user": os.getenv("ORACLE_USERNAME", os.getenv("ORACLE_USER", "testuser")),
        "password": os.getenv("ORACLE_PASSWORD", "testpass"),
        "dsn": simple_dsn,
    }


_ORACLE_REQUIRED_MODULES = {
    "test_async_store.py",
    "test_checkpoint_async.py",
    "test_checkpoint_storage_async.py",
    "test_checkpoint_storage_sync.py",
    "test_checkpoint_sync.py",
    "test_checkpoint_with_multiple_sessions.py",
    "test_concurrent_setup_race_condition.py",
    "test_connection.py",
    "test_readme_checkpoint_async.py",
    "test_search_where_async.py",
    "test_search_where_sync.py",
    "test_store_persistence_with_multiple_sessions.py",
    "test_store_search_async.py",
    "test_store_search_combinations.py",
    "test_store_search_key_combinations.py",
    "test_store_search_parameter_combinations.py",
    "test_store_search_sync.py",
}


# Check if Oracle is available
@lru_cache(maxsize=1)
def is_oracle_available() -> bool:
    """Check if Oracle database is available for testing."""
    import queue
    import threading

    result_queue = queue.Queue()

    def check_connection():
        try:
            # Try to establish a connection using connection string format
            conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
            parts = conn_string.split("@")
            user_pass, dsn = parts
            user, password = user_pass.split("/")

            with oracledb.connect(user=user, password=password, dsn=dsn):
                result_queue.put(True)
        except Exception:
            result_queue.put(False)

    # Run connection check in a thread with timeout
    thread = threading.Thread(target=check_connection, daemon=True)
    thread.start()

    try:
        # Wait up to 10 seconds for result
        return result_queue.get(timeout=10)
    except queue.Empty:
        # Timeout - Oracle is not available
        return False


# Skip marker for tests that require Oracle
def skip_if_no_oracle():
    """Dynamic skip decorator that evaluates Oracle availability at test time."""
    return pytest.mark.skipif(
        not is_oracle_available(), reason="Oracle database not available"
    )


def _item_requires_oracle(item: pytest.Item) -> bool:
    module_name = item.path.name
    if module_name in _ORACLE_REQUIRED_MODULES:
        return True

    if module_name == "test_store.py":
        return item.cls is not None and item.cls.__name__ == "TestStoreSearchSync"

    return "oracle" in item.keywords


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip Oracle-dependent tests when no Oracle database is reachable."""
    if is_oracle_available():
        return

    skip_oracle = pytest.mark.skip(reason="Oracle database not available")
    for item in items:
        if _item_requires_oracle(item):
            item.add_marker(skip_oracle)


def create_connection_string(conn_info: dict) -> str:
    """Create Oracle connection string from connection info dict."""
    return f"{conn_info['user']}/{conn_info['password']}@{conn_info['dsn']}"


# Import checkpoint fixtures so they are available to tests
from tests.conftest_checkpointer import test_data  # noqa: E402, F401

# Import store fixtures so they are available to tests
from tests.conftest_store import (  # noqa: E402
    ORACLE_DISTANCE_TYPES,  # noqa: F401
    ORACLE_INDEX_TYPES,  # noqa: F401
    fake_embeddings,  # noqa: F401
)
