# @oracle/langchain-oracledb

This package contains the LangChain.js integrations for Oracle Database.

## Installation

```bash npm2yarn
npm install @oracle/langchain-oracledb @langchain/core
```

This package, along with the main LangChain package, depends on [`@langchain/core`](https://npmjs.com/package/@langchain/core/).
If you are using this package with other LangChain packages, you should make sure that all of the packages depend on the same instance of @langchain/core.
You can do so by adding an appropriate field to your project's `package.json` like this:

```json
{
  "name": "your-project",
  "version": "0.1.0",
  "dependencies": {
    "@oracle/langchain-oracledb": "^0.1.0",
    "@langchain/core": "^1.0.0"
  },
  "resolutions": {
    "@langchain/core": "^1.0.0"
  },
  "overrides": {
    "@langchain/core": "^1.0.0"
  },
  "pnpm": {
    "overrides": {
      "@langchain/core": "^1.0.0"
    }
  }
}
```

The field you need depends on the package manager you're using, but we recommend adding a field for the common `yarn`, `npm`, and `pnpm` to maximize compatibility.

## Document Loaders

This package includes a document loader for loading documents from different sources and file formats.

```typescript
import {OracleDocLoader} from "@oracle/langchain-oracledb";

const loader = new OracleDocLoader(conn, loader_params);
const docs = await loader.load();
```

## Text Splitter

This package includes a text splitter for chunking documents using the database.

```typescript
import {OracleTextSplitter} from "@oracle/langchain-oracledb";

const splitter = new OracleTextSplitter(conn, splitter_params);
let chunks = await splitter.splitText(doc.pageContent);
```

## Embeddings

This package includes a class for generating embeddings either inside or outside of the database.

```typescript
import {OracleEmbeddings} from "@oracle/langchain-oracledb";

const embedder = new OracleEmbeddings(conn, embedder_params, proxy);
const embed = await embedder.embedQuery(chunk);
```

## Summary

This package includes a class for generating summaries either inside or outside of the database.

```typescript
import {OracleSummary} from "@oracle/langchain-oracledb";

const model = new OracleSummary(conn, summary_params, proxy);
const summary = await model.getSummary(doc.pageContent);
```

## Vector Store

This package includes a vector store for storing, indexing, and querying data in the database.

```typescript
import {OracleVS} from "@oracle/langchain-oracledb";

oraclevs = new OracleVS(embedder, dbConfig);
await oraclevs.initialize();

await oraclevs.addDocuments(docs);
const results = await oraclevs.similaritySearch("hello!", 3);
```

## Development

To develop the `@oracle/langchain-oracledb` package, you'll need to follow these instructions:

### Install dependencies

```bash
pnpm install
```

### Build the package

```bash
pnpm build
```

Or from the repo root:

```bash
pnpm build --filter @oracle/langchain-oracledb
```

### Run tests

Test files should live within a `tests/` file in the `src/` folder. Unit tests should end in `.test.ts` and integration tests should
end in `.int.test.ts`:

```bash
$ pnpm test
$ pnpm test:int
```

### Lint & Format

Run the linter & formatter to ensure your code is up to standard:

```bash
pnpm lint && pnpm format
```

### Adding new entrypoints

If you add a new file to be exported, either import & re-export from `src/index.ts`, or add it to the `exports` field in the `package.json` file and run `pnpm build` to generate the new entrypoint.
