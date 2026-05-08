#!/usr/bin/env python3
"""Test basic connection."""

import asyncio
import sys

import oracledb

from langgraph_oracledb.store.oracle import AsyncOracleStore
from tests.conftest import DEFAULT_CONNECTION_INFO

print("Testing connection...", file=sys.stderr)


def test_direct_connection():
    """Test 1: Direct oracledb connection"""
    print("\n=== Test 1: Direct oracledb connection ===", file=sys.stderr)

    # DEFAULT_CONNECTION_INFO always has dsn format now
    dsn = DEFAULT_CONNECTION_INFO["dsn"]
    user = DEFAULT_CONNECTION_INFO["user"]
    password = DEFAULT_CONNECTION_INFO["password"]

    print(f"DSN: {dsn[:50]}...", file=sys.stderr)
    print(f"User: {user}", file=sys.stderr)

    try:
        print("Attempting direct connection...", file=sys.stderr)
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        print("Direct connection successful!", file=sys.stderr)
        conn.close()
        print("Direct connection closed", file=sys.stderr)
    except Exception as e:
        print(f"Direct connection failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


async def test_single_from_conn_string():
    """Test 2: AsyncOracleStore.from_conn_string (single connection)"""
    print(
        "\n=== Test 2: AsyncOracleStore.from_conn_string (single connection) ===",
        file=sys.stderr,
    )

    # DEFAULT_CONNECTION_INFO always has dsn format now
    conn_string = f"{DEFAULT_CONNECTION_INFO['user']}/{DEFAULT_CONNECTION_INFO['password']}@{DEFAULT_CONNECTION_INFO['dsn']}"

    try:
        async with AsyncOracleStore.from_conn_string(conn_string) as store:
            await store.setup()
            print("Single from_conn_string connection successful!", file=sys.stderr)

            # Test a basic operation
            await store.aput(("test",), "key1", {"value": "test"})
            result = await store.aget(("test",), "key1")
            print(f"Basic operation test: {result is not None}", file=sys.stderr)
            await store.ateardown()
            async with oracledb.connect_async(conn_string) as conn:
                await conn.execute("drop table if exists STORE_CONFIGS purge")

    except Exception as e:
        print(f"Single from_conn_string connection failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


async def test_pooled_from_conn_string():
    """Test 3: AsyncOracleStore.from_conn_string with connection pooling"""
    print(
        "\n=== Test 3: AsyncOracleStore.from_conn_string (pooled connection) ===",
        file=sys.stderr,
    )

    # DEFAULT_CONNECTION_INFO always has dsn format now
    conn_string = f"{DEFAULT_CONNECTION_INFO['user']}/{DEFAULT_CONNECTION_INFO['password']}@{DEFAULT_CONNECTION_INFO['dsn']}"

    pool_config = {"min_size": 2, "max_size": 5}

    try:
        async with AsyncOracleStore.from_conn_string(
            conn_string, pool_config=pool_config
        ) as store:
            await store.setup()
            print("Pooled from_conn_string connection successful!", file=sys.stderr)

            # Test a basic operation
            await store.aput(("test_pool",), "key1", {"value": "test_pool"})
            result = await store.aget(("test_pool",), "key1")
            print(
                f"Basic operation test with pool: {result is not None}", file=sys.stderr
            )
            await store.ateardown()
            async with oracledb.connect_async(conn_string) as conn:
                await conn.execute("drop table if exists STORE_CONFIGS purge")

    except Exception as e:
        print(f"Pooled from_conn_string connection failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


async def run_all_tests():
    """Run all connection tests"""
    test_direct_connection()
    await test_single_from_conn_string()
    await test_pooled_from_conn_string()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
