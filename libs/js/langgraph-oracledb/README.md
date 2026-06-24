# @oracle/langgraph-oracledb

Oracle Database integrations for LangGraph JS.

This package provides two Oracle-backed persistence components:

- `OracleCheckpointSaver`: a LangGraph `BaseCheckpointSaver` implementation for durable graph checkpoints, channel blobs, pending writes, checkpoint listing, and thread deletion.
- `OracleStore`: a LangGraph `BaseStore` implementation for long-term memory with JSON values, namespace search, filter operators, pagination, and optional Oracle VECTOR search.

The implementation is designed to match LangGraph JS checkpoint and store semantics while accounting for Oracle-specific behavior such as empty string handling, byte-length limits, BLOB/CLOB binds, transaction rollback, duplicate-key races, and VECTOR literal limits.

## Requirements

- Node.js 18 or newer.
- Oracle Database connectivity through `oracledb`.
- Oracle credentials with permission to create, read, update, and delete the configured tables.
- Oracle VECTOR support is required only when using `OracleStore` with an `index` configuration.

The integration tests use these environment variables:

```sh
export ORACLE_USER="your_user"
export ORACLE_PASSWORD="your_password"
export ORACLE_CONNECT_STRING="host:port/service_name"
```

## Installation

From inside this monorepo workspace:

```sh
pnpm install
```

Install the package from npm:

```sh
pnpm add @oracle/langgraph-oracledb @langchain/core @langchain/langgraph-checkpoint
```

## Diagnostics

Both integrations expose read-only diagnostics for schema, migration, runtime,
and Oracle capability checks. These are useful as a first review step because
they do not run setup or mutate data.

```ts
const checkpointDiagnostics = await checkpointer.getDiagnostics({
  includeRowCounts: true,
});

const storeDiagnostics = await store.getDiagnostics({
  includeRowCounts: true,
});
```

Diagnostics report expected table names, migration status, missing or
mismatched columns, primary-key status, runtime node-oracledb mode, and VECTOR
availability for stores.

For stores configured with vector indexing, the top-level diagnostics status
also reflects VECTOR probe readiness: an unavailable VECTOR probe reports a
partial store status, while an inconclusive probe reports unknown. Inspect
`storeDiagnostics.vector.probe` for the database error details.

## Checkpoint Saver

`OracleCheckpointSaver` persists graph checkpoints and pending writes in Oracle tables. Call `setup()` before first use so the migration table and checkpoint tables exist.

```ts
import { OracleCheckpointSaver } from "@oracle/langgraph-oracledb";

const checkpointer = new OracleCheckpointSaver({
  connection: {
    user: process.env.ORACLE_USER,
    password: process.env.ORACLE_PASSWORD,
    connectString: process.env.ORACLE_CONNECT_STRING,
  },
});

await checkpointer.setup();
```

Use it with a LangGraph graph the same way as other LangGraph checkpointers:

```ts
const graph = workflow.compile({ checkpointer });

const config = {
  configurable: {
    thread_id: "user-123",
  },
};

await graph.invoke({ input: "remember this" }, config);

const latest = await checkpointer.getTuple(config);
```

### Checkpointer Behavior

The saver supports:

- exact checkpoint lookup by `thread_id`, `checkpoint_ns`, and `checkpoint_id`
- latest checkpoint lookup when `checkpoint_id` is omitted
- root checkpoint namespace `""`
- child checkpoint namespaces
- reversible namespace encoding to avoid Oracle empty-string/`NULL` collisions
- parent checkpoint config round trips
- pending writes via `putWrites`
- special write indexes from LangGraph's `WRITES_IDX_MAP`
- legacy pending sends hydration for checkpoints with `v < 4`
- metadata storage and list filtering
- custom LangGraph serde implementations
- `deleteThread(threadId)` cleanup across checkpoint, blob, and write tables

Current migrations store checkpoint bodies, metadata, channel values, and writes as BLOB values. Legacy CLOB checkpoint tables are supported only for text-compatible serializer payloads.

### Connection Modes

Recommended production usage is either:

- pass Oracle connection options, so the saver creates and owns an Oracle pool, or
- pass an existing Oracle pool.

Passing a raw Oracle connection object is supported for advanced callers. Saver operations on that caller-owned raw connection are serialized to avoid interleaving transaction scopes, but the connection lifecycle remains the caller's responsibility. Prefer a pool for concurrent graph execution.

## Store

`OracleStore` persists LangGraph long-term memory items as JSON values in Oracle. It supports `get`, `put`, `delete`, `search`, `batch`, and `listNamespaces`.

```ts
import { OracleStore } from "@oracle/langgraph-oracledb/store";

const store = new OracleStore({
  connection: {
    user: process.env.ORACLE_USER,
    password: process.env.ORACLE_PASSWORD,
    connectString: process.env.ORACLE_CONNECT_STRING,
  },
});

await store.put(["memories", "user-1"], "profile", {
  name: "Ada",
  score: 10,
});

const item = await store.get(["memories", "user-1"], "profile");

const results = await store.search(["memories"], {
  filter: { score: { $gte: 5 } },
  limit: 10,
});
```

Store keys are encoded internally before Oracle writes and decoded on reads. This preserves BaseStore behavior for keys such as `""` and strings that look like internal encodings.

### Store Filters

`OracleStore.search()` supports these filter operators:

- `$eq`
- `$ne`
- `$gt`
- `$gte`
- `$lt`
- `$lte`
- `$in`
- `$nin`
- `$exists`

The store safely pushes simple predicates into SQL when Oracle can preserve LangGraph semantics. It then applies strict JavaScript/BaseStore filtering before final pagination. Predicates that are unsafe for Oracle pushdown, including empty strings, strings over Oracle `JSON_VALUE` string limits, range comparisons that rely on BaseStore `Number(...)` coercion, `$ne`, and non-primitive filters, fall back to broader SQL scans followed by strict JavaScript filtering.

### Namespace Listing

`listNamespaces()` supports:

- prefix matching
- suffix matching
- `*` wildcard labels
- `maxDepth` projection
- `limit` and `offset`

Concrete prefix/suffix filters are pushed into SQL. Wildcards and `maxDepth` projection may require broader scans before in-memory projection and final pagination.

## Store With Oracle VECTOR Search

Pass an `IndexConfig` to enable vector indexing and semantic search.

```ts
import { OracleStore } from "@oracle/langgraph-oracledb/store";

const store = new OracleStore({
  connection: {
    user: process.env.ORACLE_USER,
    password: process.env.ORACLE_PASSWORD,
    connectString: process.env.ORACLE_CONNECT_STRING,
  },
  index: {
    dims: 1536,
    embeddings: myEmbeddings,
    fields: ["text", "metadata.summary", "chapters[*].content"],
  },
});

await store.put(["memories", "user-1"], "note", {
  text: "Ada likes database systems and agent memory.",
  metadata: { summary: "User memory about Ada" },
});

const results = await store.search(["memories", "user-1"], {
  query: "database memory",
  filter: { "metadata.summary": { $exists: true } },
  limit: 5,
});
```

Vector indexing supports:

- default whole-document indexing with `fields` omitted
- configured field indexing
- array paths such as `chapters[*].content`, `authors[0].name`, and `items[-1].text`
- per-put field overrides via `store.put(namespace, key, value, ["field.path"])`
- `index: false` to store the JSON row without vector rows
- dimension validation for `index.dims` before any Oracle DDL or DML runs; dimensions must be between 1 and 65535

Rows without vector entries can still appear as scoreless results after scored vector matches, matching LangGraph's in-memory store behavior.

The store validates that all embedding values are finite numbers and that embedding dimensions match `index.dims`. It uses native node-oracledb VECTOR binds when available and falls back to `TO_VECTOR(:embedding)` string binds for compatibility. The string-bind fallback validates Oracle's 32767-byte bind limit before executing SQL.

### Vector Index Management

`OracleStore` can create, list, and drop Oracle VECTOR indexes for the configured vector table.

```ts
await store.createVectorIndex({
  type: "IVF",
  name: "LG_MEMORY_IVF_IDX",
  accuracy: 90,
  neighborPartitions: 1,
});

const indexes = await store.listVectorIndexes();

await store.dropVectorIndex({
  name: "LG_MEMORY_IVF_IDX",
  ifExists: true,
});
```

Supported index types are `HNSW` and `IVF`. HNSW-specific options are `neighbors` and `efConstruction`, which must be provided together. IVF-specific options use `neighborPartitions`. Index names are validated as Oracle identifiers, and `dropVectorIndex()` refuses to drop an index unless it is on the store's vector embedding column.

## Table Names and Setup

Both integrations accept an optional `tablePrefix`. The prefix is validated as an Oracle identifier prefix and normalized to uppercase.

Checkpoint tables:

- `<PREFIX>CHECKPOINTS`
- `<PREFIX>CHECKPOINT_BLOBS`
- `<PREFIX>CHECKPOINT_WRITES`
- `<PREFIX>CHECKPOINT_MIGRATIONS`

Store tables:

- `<PREFIX>STORE`
- `<PREFIX>STORE_VECTORS`
- `<PREFIX>STORE_MIGRATIONS`

Setup is idempotent and records migrations in the migration table. Transient setup failures clear the in-memory setup promise so a later setup call can retry.

## Validation

Run the package checks with:

```sh
pnpm --filter @oracle/langgraph-oracledb lint
pnpm --filter @oracle/langgraph-oracledb exec tsc --noEmit
pnpm --filter @oracle/langgraph-oracledb test
pnpm --filter @oracle/langgraph-oracledb test:int
pnpm --filter @oracle/langgraph-oracledb run lint:dpdm
pnpm --filter @oracle/langgraph-oracledb build:internal
```

What these commands cover:

- `lint`: source linting with the repository's configured Oxlint rules.
- `tsc --noEmit`: TypeScript type-checking without writing compiled files.
- `test`: unit tests for SQL helpers and saver error/race handling.
- `test:int`: Oracle-backed integration tests, including the shared `@langchain/langgraph-checkpoint-validation` spec for `OracleCheckpointSaver`.
- `lint:dpdm`: circular dependency detection for package source files.
- `build:internal`: package compilation, type declaration generation, and package export validation.

If Oracle credentials are not present, Oracle integration tests are skipped.

## Current Limitations

- Oracle VECTOR rows are not automatically backfilled for existing JSON-only store rows when vector indexing is enabled later. Updating an item with indexing enabled creates vector rows for that item.
- VECTOR indexes are not created automatically during setup. Create them explicitly with `createVectorIndex()` when you want Oracle ANN indexing.
- Native VECTOR binds are used when node-oracledb and the connected database support them; otherwise VECTOR writes and queries fall back to `TO_VECTOR(:embedding)` string binds with a documented 32767-byte preflight limit.
- Complex filters and wildcard/depth namespace listing can require broader SQL scans followed by strict JavaScript filtering.
- Store key encoding is not backward-compatible with pre-encoding OracleStore rows. This is acceptable before first release; shipped data would require a migration or a dual-read strategy.

## Cleanup

For checkpointer data, remove all checkpoints and writes for a thread:

```ts
await checkpointer.deleteThread("thread-id");
```

For store data, delete individual keys:

```ts
await store.delete(["memories", "user-1"], "profile");
```

Close owned resources when finished:

```ts
await store.stop();
await checkpointer.end();
```
