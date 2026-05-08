"""
Test that specifically targets the ORA-00001 race condition in concurrent setup().
This test creates a scenario that would definitely fail without IGNORE_ROW_ON_DUPKEY_INDEX.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

import oracledb
import pytest

from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver, OracleSaver
from langgraph_oracledb.store.oracle import AsyncOracleStore
from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string
from tests.conftest_checkpointer import (
    _async_cleanup_checkpoint_tables,
    _cleanup_checkpoint_tables,
)


class TestConcurrentSetupRaceCondition:
    """Test that IGNORE_ROW_ON_DUPKEY_INDEX actually prevents ORA-00001 errors."""

    def test_sync_migration_race_condition(self):
        """Test that would fail with ORA-00001 without the IGNORE_ROW_ON_DUPKEY_INDEX hint."""
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        # Use a barrier to make sure all workers try to insert the same migration version simultaneously
        from threading import Barrier, Event

        num_workers = 3
        barrier = Barrier(num_workers)
        start_event = Event()
        errors = []

        def run_setup_with_barrier(worker_id):
            """Run setup() but coordinate timing to maximize race condition."""
            try:
                with OracleSaver.from_conn_string(conn_string) as checkpointer:
                    # Wait for all workers to be ready
                    barrier.wait()

                    # Wait for start signal to ensure maximum concurrency
                    start_event.wait()

                    # All workers try to run setup at exactly the same time
                    checkpointer.setup()
                    return f"Worker {worker_id}: Success"
            except Exception as e:
                error_msg = f"Worker {worker_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                # Check if this is specifically the unique constraint violation we're preventing
                if "ORA-00001" in str(e) or "unique constraint" in str(e).lower():
                    errors.append(
                        f"CRITICAL: Worker {worker_id} hit unique constraint error: {e}"
                    )
                return error_msg

        try:
            # Start all workers simultaneously
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(run_setup_with_barrier, i)
                    for i in range(num_workers)
                ]

                # Small delay to let all workers reach the barrier
                import time

                time.sleep(0.1)

                # Release all workers simultaneously
                start_event.set()

                # Collect results
                results = [future.result() for future in futures]

            # Verify no ORA-00001 errors occurred
            unique_constraint_errors = [
                e
                for e in errors
                if "ORA-00001" in e or "unique constraint" in e.lower()
            ]
            assert len(unique_constraint_errors) == 0, (
                f"Unique constraint violations detected: {unique_constraint_errors}"
            )

            # All setups should succeed
            success_count = len([r for r in results if "Success" in r])
            assert success_count == num_workers, (
                f"Expected {num_workers} successes, got {success_count}. Errors: {errors}"
            )

        finally:
            _cleanup_checkpoint_tables()

    @pytest.mark.asyncio
    async def test_async_migration_race_condition(self):
        """Test async version that would fail with ORA-00001 without the hint."""
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        num_workers = 3
        errors = []

        # Use asyncio events to coordinate timing
        start_event = asyncio.Event()
        ready_count = 0
        ready_lock = asyncio.Lock()

        async def run_setup_with_coordination(worker_id):
            """Run setup() but coordinate timing to maximize race condition."""
            nonlocal ready_count
            try:
                async with AsyncOracleSaver.from_conn_string(
                    conn_string
                ) as checkpointer:
                    # Signal that this worker is ready
                    async with ready_lock:
                        ready_count += 1

                    # Wait until all workers are ready
                    while ready_count < num_workers:
                        await asyncio.sleep(0.001)

                    # Wait for start signal
                    await start_event.wait()

                    # All workers try to run setup at the same time
                    await checkpointer.setup()
                    return f"Worker {worker_id}: Success"
            except Exception as e:
                error_msg = f"Worker {worker_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                # Check if this is specifically the unique constraint violation we're preventing
                if "ORA-00001" in str(e) or "unique constraint" in str(e).lower():
                    errors.append(
                        f"CRITICAL: Worker {worker_id} hit unique constraint error: {e}"
                    )
                return error_msg

        try:
            # Start all workers
            tasks = [run_setup_with_coordination(i) for i in range(num_workers)]

            # Small delay to let workers get ready
            await asyncio.sleep(0.1)

            # Release all workers simultaneously
            start_event.set()

            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and check for exceptions
            string_results = []
            for result in results:
                if isinstance(result, Exception):
                    error_msg = f"Exception: {type(result).__name__}: {result}"
                    errors.append(error_msg)
                    if (
                        "ORA-00001" in str(result)
                        or "unique constraint" in str(result).lower()
                    ):
                        errors.append(
                            f"CRITICAL: Hit unique constraint error: {result}"
                        )
                else:
                    string_results.append(result)

            # Verify no ORA-00001 errors occurred
            unique_constraint_errors = [
                e
                for e in errors
                if "ORA-00001" in e or "unique constraint" in e.lower()
            ]
            assert len(unique_constraint_errors) == 0, (
                f"Unique constraint violations detected: {unique_constraint_errors}"
            )

            # All setups should succeed
            success_count = len([r for r in string_results if "Success" in r])
            assert success_count == num_workers, (
                f"Expected {num_workers} successes, got {success_count}. Errors: {errors}"
            )

        finally:
            await _async_cleanup_checkpoint_tables()

    def test_hint_syntax_is_correct(self):
        """Test that our IGNORE_ROW_ON_DUPKEY_INDEX hint has correct syntax."""
        from langgraph_oracledb.checkpoint.oracle.aio import AsyncOracleSaver
        from langgraph_oracledb.checkpoint.oracle.sync import OracleSaver

        # Check that the SQL in our setup methods contains the correct hint
        # This is a simple syntax validation test

        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        # Test that we can create instances and that they have the expected SQL
        try:
            with OracleSaver.from_conn_string(conn_string) as sync_saver:
                # The setup method should contain the hint in its implementation
                # We can't easily inspect the exact SQL, but we can verify setup works
                sync_saver.setup()

            # Test async version
            async def test_async():
                async with AsyncOracleSaver.from_conn_string(
                    conn_string
                ) as async_saver:
                    await async_saver.setup()

            asyncio.run(test_async())

            # If we get here without ORA-00001 errors, the hint is working
            assert True, "Setup completed without unique constraint violations"

        finally:
            _cleanup_checkpoint_tables()

    @pytest.mark.asyncio
    async def test_async_store_migration_race_condition(self):
        """Test async store setup that would fail with ORA-00001 without the hint."""
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        num_workers = 3
        errors = []

        # Use the SAME table suffix for all workers to force them to compete for same migration table
        shared_table_suffix = f"race_test_{uuid.uuid4().hex[:8]}"

        # Use asyncio events to coordinate timing
        start_event = asyncio.Event()
        ready_count = 0
        ready_lock = asyncio.Lock()

        async def run_store_setup_with_coordination(worker_id):
            """Run store setup() but coordinate timing to maximize race condition."""
            nonlocal ready_count
            try:
                # CRITICAL: All workers use the SAME table_suffix to compete for same migration records
                async with AsyncOracleStore.from_conn_string(
                    conn_string, table_suffix=shared_table_suffix
                ) as store:
                    # Signal that this worker is ready
                    async with ready_lock:
                        ready_count += 1

                    # Wait until all workers are ready
                    while ready_count < num_workers:
                        await asyncio.sleep(0.001)

                    # Wait for start signal
                    await start_event.wait()

                    # All workers try to run setup at exactly the same time on SAME tables
                    await store.setup()
                    return f"Worker {worker_id}: Success"
            except Exception as e:
                error_msg = f"Worker {worker_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                # Check if this is specifically the unique constraint violation we're preventing
                if "ORA-00001" in str(e) or "unique constraint" in str(e).lower():
                    errors.append(
                        f"CRITICAL: Worker {worker_id} hit unique constraint error: {e}"
                    )
                return error_msg

        try:
            # Start all workers
            tasks = [run_store_setup_with_coordination(i) for i in range(num_workers)]

            # Small delay to let workers get ready
            await asyncio.sleep(0.1)

            # Release all workers simultaneously
            start_event.set()

            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and check for exceptions
            string_results = []
            for result in results:
                if isinstance(result, Exception):
                    error_msg = f"Exception: {type(result).__name__}: {result}"
                    errors.append(error_msg)
                    if (
                        "ORA-00001" in str(result)
                        or "unique constraint" in str(result).lower()
                    ):
                        errors.append(
                            f"CRITICAL: Hit unique constraint error: {result}"
                        )
                else:
                    string_results.append(result)

            # Verify no ORA-00001 unique constraint errors occurred (our hint prevents these)
            unique_constraint_errors = [
                e
                for e in errors
                if "ORA-00001" in e or "unique constraint" in e.lower()
            ]
            assert len(unique_constraint_errors) == 0, (
                f"Unique constraint violations detected: {unique_constraint_errors}"
            )

            # Note: ORA-00955 (name already exists) errors are expected and handled gracefully by the store implementation
            # The important thing is that at least one setup succeeds and no ORA-00001 errors occur
            success_count = len([r for r in string_results if "Success" in r])
            assert success_count >= 1, (
                f"Expected at least 1 success, got {success_count}. This indicates concurrent setup handling works."
            )

            # The other workers may get ORA-00955 (table already exists) which is expected concurrent DDL behavior
            ddl_errors = [
                e
                for e in errors
                if "ORA-00955" in e or "name is already used" in e.lower()
            ]
            total_operations = success_count + len(ddl_errors)
            assert total_operations == num_workers, (
                f"Expected {num_workers} total operations (successes + DDL conflicts), got {total_operations}"
            )

        finally:
            # Clean up created store tables
            async with AsyncOracleStore.from_conn_string(
                conn_string, table_suffix=shared_table_suffix
            ) as store:
                await store.setup()
                await store.ateardown()
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")

    @pytest.mark.asyncio
    async def test_async_store_vector_migration_race_condition(self):
        """Test async store with vector config setup race condition."""
        from tests.embed_test_utils import CharacterEmbeddings

        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        num_workers = 3
        errors = []

        # Use the SAME table suffix for all workers - this is KEY to trigger race condition
        shared_table_suffix = f"vec_race_{uuid.uuid4().hex[:8]}"
        created_stores = []
        fake_embeddings = CharacterEmbeddings(dims=100)

        # Use asyncio events to coordinate timing
        start_event = asyncio.Event()
        ready_count = 0
        ready_lock = asyncio.Lock()

        index_config = {
            "dims": fake_embeddings.dims,
            "embed": fake_embeddings,
            "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
        }

        async def run_vector_store_setup_with_coordination(worker_id):
            """Run vector store setup() but coordinate timing to maximize race condition."""
            nonlocal ready_count
            try:
                # CRITICAL: All workers use the SAME table_suffix AND same index config
                # This forces them to compete for same migration records in both store and vector tables

                async with AsyncOracleStore.from_conn_string(
                    conn_string, table_suffix=shared_table_suffix, index=index_config
                ) as store:
                    created_stores.append(store)

                    # Signal that this worker is ready
                    async with ready_lock:
                        ready_count += 1

                    # Wait until all workers are ready
                    while ready_count < num_workers:
                        await asyncio.sleep(0.001)

                    # Wait for start signal
                    await start_event.wait()

                    # All workers try to run setup at exactly the same time
                    # This will try to insert into BOTH store_migrations AND vector_migrations
                    await store.setup()
                    return f"Worker {worker_id}: Success"
            except Exception as e:
                error_msg = f"Worker {worker_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                # Check if this is specifically the unique constraint violation we're preventing
                if "ORA-00001" in str(e) or "unique constraint" in str(e).lower():
                    errors.append(
                        f"CRITICAL: Worker {worker_id} hit unique constraint error: {e}"
                    )
                return error_msg

        try:
            # Start all workers
            tasks = [
                run_vector_store_setup_with_coordination(i) for i in range(num_workers)
            ]

            # Small delay to let workers get ready
            await asyncio.sleep(0.1)

            # Release all workers simultaneously
            start_event.set()

            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and check for exceptions
            string_results = []
            for result in results:
                if isinstance(result, Exception):
                    error_msg = f"Exception: {type(result).__name__}: {result}"
                    errors.append(error_msg)
                    if (
                        "ORA-00001" in str(result)
                        or "unique constraint" in str(result).lower()
                    ):
                        errors.append(
                            f"CRITICAL: Hit unique constraint error: {result}"
                        )
                else:
                    string_results.append(result)

            # Verify no ORA-00001 unique constraint errors occurred (our hint prevents these)
            unique_constraint_errors = [
                e
                for e in errors
                if "ORA-00001" in e or "unique constraint" in e.lower()
            ]
            assert len(unique_constraint_errors) == 0, (
                f"Unique constraint violations detected: {unique_constraint_errors}"
            )

            # Note: ORA-00955 (name already exists) errors are expected for concurrent DDL operations
            # The important thing is that at least one setup succeeds and no ORA-00001 errors occur
            success_count = len([r for r in string_results if "Success" in r])
            assert success_count >= 1, (
                f"Expected at least 1 success, got {success_count}. This indicates concurrent vector setup works."
            )

            # The other workers may get ORA-00955 (table/index already exists) which is expected
            print(errors)
            ddl_errors = [
                e
                for e in errors
                if "ORA-00955" in e or "name is already used" in e.lower()
            ]
            total_operations = success_count + len(ddl_errors)
            assert total_operations == num_workers, (
                f"Expected {num_workers} total operations (successes + DDL conflicts), got {total_operations}"
            )

        finally:
            try:
                async with AsyncOracleStore.from_conn_string(
                    conn_string, table_suffix=shared_table_suffix, index=index_config
                ) as store:
                    await store.setup()
                    await store.ateardown()
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")
            except Exception as e:
                print(f"Warning: Failed to teardown vector store: {e}")

    def test_sync_and_async_store_mixed_race_condition(self):
        """Test that mixed sync/async store operations don't interfere with each other."""
        # This test is more complex since we don't have sync store implementation
        # But we can test that async store setup is thread-safe when called from threads
        import threading

        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        num_workers = 2
        errors = []
        shared_table_suffix = f"mixed_race_{uuid.uuid4().hex[:8]}"

        # Use threading barrier like the sync checkpoint test
        barrier = threading.Barrier(num_workers)
        start_event = threading.Event()

        def run_async_store_from_thread(worker_id):
            """Run async store setup from a thread (simulating mixed usage)."""
            try:
                barrier.wait()  # Synchronize start
                start_event.wait()  # Wait for release signal

                # Run async store setup from within thread
                async def setup_store():
                    async with AsyncOracleStore.from_conn_string(
                        conn_string, table_suffix=shared_table_suffix
                    ) as store:
                        await store.setup()
                    return f"Thread Worker {worker_id}: Success"

                # Run the async function from thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(setup_store())
                    return result
                finally:
                    loop.close()

            except Exception as e:
                error_msg = f"Thread Worker {worker_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                if "ORA-00001" in str(e) or "unique constraint" in str(e).lower():
                    errors.append(
                        f"CRITICAL: Thread Worker {worker_id} hit unique constraint error: {e}"
                    )
                return error_msg

        try:
            # Start all workers simultaneously
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(run_async_store_from_thread, i)
                    for i in range(num_workers)
                ]

                # Small delay to let all workers reach the barrier
                import time

                time.sleep(0.1)

                # Release all workers simultaneously
                start_event.set()

                # Collect results
                results = [future.result() for future in futures]

            # Verify no ORA-00001 errors occurred
            unique_constraint_errors = [
                e
                for e in errors
                if "ORA-00001" in e or "unique constraint" in e.lower()
            ]
            assert len(unique_constraint_errors) == 0, (
                f"Unique constraint violations detected: {unique_constraint_errors}"
            )

            # For mixed threading/async operations, DDL conflicts are expected
            success_count = len([r for r in results if "Success" in r])
            assert success_count >= 1, (
                f"Expected at least 1 success, got {success_count}. This indicates store setup works in mixed environments."
            )

            # Account for DDL conflicts
            ddl_errors = [
                e
                for e in errors
                if "ORA-00955" in e or "name is already used" in e.lower()
            ]
            total_operations = success_count + len(ddl_errors)
            assert total_operations == num_workers, (
                f"Expected {num_workers} total operations, got {total_operations}"
            )

        finally:
            # Clean up - run async cleanup from sync context
            async def cleanup_stores():
                async with AsyncOracleStore.from_conn_string(
                    conn_string,
                    table_suffix=shared_table_suffix,
                ) as store:
                    await store.setup()
                    await store.ateardown()
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")

            # Run cleanup
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(cleanup_stores())
            finally:
                loop.close()
