"""Select real SQL, native VecDB, or REST backends for shared behavior tests."""

BACKENDS = ("sql", "sql-pool", "vecdb", "vecdb-pool", "rest")


def pytest_addoption(parser):
    parser.addoption(
        "--oracle-backends",
        default=",".join(BACKENDS),
        help="Comma-separated shared-test backends: " + ", ".join(BACKENDS),
    )


def pytest_generate_tests(metafunc):
    if "oracle_backend" in metafunc.fixturenames:
        names = metafunc.config.getoption("--oracle-backends").split(",")
        if not names or any(name not in BACKENDS for name in names):
            import pytest

            raise pytest.UsageError("Unknown --oracle-backends selection")
        metafunc.parametrize("oracle_backend", names)
