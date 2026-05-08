"""Oracle test configuration."""

import os
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


# Check if Oracle is available
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


def create_connection_string(conn_info: dict) -> str:
    """Create Oracle connection string from connection info dict."""
    return f"{conn_info['user']}/{conn_info['password']}@{conn_info['dsn']}"


# Import checkpoint fixtures so they are available to tests
from tests.conftest_checkpointer import test_data  # noqa: E402, F401

# Import store fixtures so they are available to tests
from tests.conftest_store import (
    ORACLE_DISTANCE_TYPES,
    ORACLE_INDEX_TYPES,
    fake_embeddings,
)  # noqa: E402, F401
