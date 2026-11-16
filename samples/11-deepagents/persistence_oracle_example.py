"""Deepagents persistence sample: Oracle-backed checkpoints with OracleSaver.

Runs a deep agent twice on the same ``thread_id`` with an Oracle-backed
checkpointer, proving that the second turn resumes with the first turn's
conversation state, then lists the stored checkpoints.

Connection is configured through environment variables, either as a plain
connection string (local/container databases):

    export ORACLE_CONN_STRING="USER/PASSWORD@localhost:1521/FREEPDB1"

or as wallet-based ADB settings (same variables the other samples use):

    export ADB_DSN="mydb_low"
    export ADB_USER="ADMIN"
    export ADB_PASSWORD="..."
    export ADB_WALLET_LOCATION="$HOME/wallets/mydb"
    export ADB_WALLET_PASSWORD="..."   # defaults to ADB_PASSWORD

Model settings follow the other samples in this folder:

    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_REGION="us-chicago-1"
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_AUTH_PROFILE="DEFAULT"
    export OCI_DEEPAGENTS_MODEL="google.gemini-2.5-flash"

Run from the repository root:

    PYTHONPATH=libs/oci python samples/11-deepagents/persistence_oracle_example.py \
      --thread-id persistence-demo-01
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from langchain_core.messages import HumanMessage

from langchain_oci import create_deepagents_agent


def build_checkpointer() -> tuple[Any, Any]:
    """Create an OracleSaver from environment configuration.

    Returns (checkpointer, context_manager_or_None); the caller owns cleanup.
    """
    from langgraph_oracledb.checkpoint.oracle import OracleSaver

    conn_string = os.environ.get("ORACLE_CONN_STRING")
    if conn_string:
        # from_conn_string is a context manager; keep the connection open for
        # the whole demo by entering it manually and closing in main().
        cm = OracleSaver.from_conn_string(conn_string)
        checkpointer = cm.__enter__()
        return checkpointer, cm

    import oracledb

    wallet = os.environ.get("ADB_WALLET_LOCATION")
    wallet = os.path.expanduser(wallet) if wallet else None
    password = os.environ["ADB_PASSWORD"]
    conn = oracledb.connect(
        user=os.environ.get("ADB_USER", "ADMIN"),
        password=password,
        dsn=os.environ["ADB_DSN"],
        config_dir=wallet,
        wallet_location=wallet,
        wallet_password=os.environ.get("ADB_WALLET_PASSWORD", password),
    )
    return OracleSaver(conn), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="persistence-demo-01")
    args = parser.parse_args()

    checkpointer, cm = build_checkpointer()
    try:
        checkpointer.setup()  # idempotent: creates tables + migrations

        agent = create_deepagents_agent(
            model_id=os.environ.get("OCI_DEEPAGENTS_MODEL", "google.gemini-2.5-flash"),
            compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
            checkpointer=checkpointer,
        )

        cfg = {"configurable": {"thread_id": args.thread_id}}

        print(f"--- turn 1 (thread_id={args.thread_id}) ---")
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Remember this fact for later: the launch codename "
                            "is BLUE HERON. Reply with a short confirmation."
                        )
                    )
                ]
            },
            cfg,
        )
        print(result["messages"][-1].content)

        print("--- turn 2 (same thread, new invoke) ---")
        result = agent.invoke(
            {"messages": [HumanMessage(content="What is the launch codename?")]},
            cfg,
        )
        print(result["messages"][-1].content)

        print("--- stored checkpoints ---")
        for i, checkpoint in enumerate(checkpointer.list(cfg)):
            print(f"{i}: ts={checkpoint.checkpoint['ts']}")

        state = agent.get_state(cfg)
        print(f"--- current state: {len(state.values.get('messages', []))} messages ---")
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
        else:
            checkpointer.conn.close()


if __name__ == "__main__":
    main()
