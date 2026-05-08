# LangGraph Oracle Persistence (Checkpoint + Store)

Oracle-backed implementations for:
- Checkpoints: OracleSaver (sync) and AsyncOracleSaver (async)
- Key/Value Store with optional vector search: OracleStore (sync) and AsyncOracleStore (async)

Supports:
- Oracle Database for AI Vector Search
- Python 3.10+ and `oracledb` driver

## Quickstart

### Checkpoints (Async)

```python
import os
import asyncio
from dotenv import load_dotenv
from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver

load_dotenv()

async def main():
    conn_string = f"{os.environ['ORACLE_USERNAME']}/{os.environ['ORACLE_PASSWORD']}@{os.environ['ORACLE_DSN']}"
    async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()  # Create tables & apply migrations once

        # Then pass to your graph compile (example)
        # graph = app.compile(checkpointer=checkpointer)
        # await graph.ainvoke(...)

if __name__ == "__main__":
    asyncio.run(main())
```

Sync variant:

```python
from langgraph_oracledb.checkpoint.oracle import OracleSaver

conn_string = "user/password@localhost:1521/FREEPDB1"
with OracleSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()
    # graph = app.compile(checkpointer=checkpointer)
    # graph.invoke(...)
```

### Store (Async, basic key/value)

```python
import asyncio
from langgraph_oracledb.store.oracle import AsyncOracleStore

async def main():
    conn_string = "user/password@localhost:1521/FREEPDB1"
    async with AsyncOracleStore.from_conn_string(conn_string) as store:
        await store.setup()  # Create tables & apply migrations once

        ns = ("readme", "example")
        await store.aput(ns, "doc1", {"text": "hello"})
        item = await store.aget(ns, "doc1")
        print(item.value)  # {"text": "hello"}

        # Non-vector search (lists items by namespace)
        results = await store.asearch(ns, limit=10)
        print(len(results) >= 1)

if __name__ == "__main__":
    asyncio.run(main())
```

Sync variant:

```python
from langgraph_oracledb.store.oracle import OracleStore

conn_string = "user/password@localhost:1521/FREEPDB1"
with OracleStore.from_conn_string(conn_string) as store:
    store.setup()
    ns = ("readme", "example")
    store.put(ns, "doc1", {"text": "hello"})
    item = store.get(ns, "doc1")
    print(item.value)
    results = store.search(ns, limit=10)
```

### Vector Search (optional)

Vector search is enabled by passing an `index` configuration with:
- `dims`: embedding dimension
- `embed`: a LangChain Embeddings implementation (e.g., OpenAI, HF, or your own)
- optional `fields`: which JSON fields to embed (default: whole document)
- optional `index_type`: HNSW/IVF and parameters

Example (async):

```python
# Pseudo-embeddings for illustration; use any LangChain Embeddings implementation
from langchain_core.embeddings import Embeddings
from langgraph_oracledb.store.oracle import AsyncOracleStore

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts): return [[0.0]*8 for _ in texts]
    def embed_query(self, text): return [0.0]*8

async with AsyncOracleStore.from_conn_string(
    "user/password@localhost:1521/FREEPDB1",
    index={
        "dims": 8,
        "embed": FakeEmbeddings(),
        "fields": ["text"],  # embed only the 'text' field
        "index_type": {"type": "hnsw", "neighbors": 16, "efconstruction": 200, "distance_metric": "COSINE"},
    },
) as store:
    await store.setup()
    ns = ("docs",)
    await store.aput(ns, "a", {"text": "alpha"})
    await store.aput(ns, "b", {"text": "beta"})
    results = await store.asearch(ns, query="alphabet", limit=2)
```

Notes:
- Call `setup()` once per database/schema to create/upgrade tables.
- Vector search requires Oracle 23c/23ai+ with AI Vector Search enabled.

## Testing the Examples

The repository's tests will skip automatically if an Oracle instance is not reachable.
- Place your Oracle credentials in `.env` (see Configuration)
- Run: `pytest -q`

This repository includes tests that validate the examples above:
- Async checkpoint setup works
- Async store put/get/search works
