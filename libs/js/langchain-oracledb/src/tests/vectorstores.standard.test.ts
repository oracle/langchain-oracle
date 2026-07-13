import { Document } from "@langchain/core/documents";
import type { DBError } from "oracledb";
import { describe, expect, test } from "vitest";

import {
  LangChainOracleError,
  OracleErrorCode,
  OracleVS,
  createIndex,
  createTable,
  dropTablePurge,
  generateWhereClause,
} from "../vectorstores.js";
import { OracleDocLoader } from "../document_loaders.js";

const embeddings = {
  embedQuery: async () => [0.1, 0.2],
  embedDocuments: async (texts: string[]) => texts.map(() => [0.1, 0.2]),
};

type OracleVSInternals = {
  ensureEmbeddingDimension(): number;
  normalizeReturnedEmbedding(value: unknown): unknown;
};

type OracleDocLoaderInternals = {
  _loadFromTable(owner: string, table: string, col: string): Promise<unknown>;
};

async function expectOracleErrorCode(
  input: Promise<unknown> | (() => unknown),
  code: OracleErrorCode,
): Promise<void> {
  try {
    if (typeof input === "function") {
      await input();
    } else {
      await input;
    }
  } catch (error) {
    const dbError = error as DBError;
    expect(dbError.code).toBe(code);
    return;
  }

  throw new Error(`Expected LangChainOracleError with code ${code}`);
}

describe("generateWhereClause", () => {
  test("binds scalar values instead of interpolating them into SQL", () => {
    const bindValues: unknown[] = [];

    const clause = generateWhereClause(
      { author: "Robert'); DROP TABLE docs; --" },
      bindValues
    );

    expect(clause).toContain("JSON_EXISTS");
    expect(clause).not.toContain("DROP TABLE");
    expect(bindValues).toEqual(["Robert'); DROP TABLE docs; --"]);
  });

  test("rejects metadata keys containing SQL injection payloads", async () => {
    await expectOracleErrorCode(
      () => generateWhereClause({ ["author') OR 1=1 --"]: "alice" }, []),
      OracleErrorCode.FILTER_INVALID_METADATA_KEY,
    );
  });

  test("covers filter validation error codes", async () => {
    await expectOracleErrorCode(
      () => generateWhereClause({ tags: { $in: "bad" } }, []),
      OracleErrorCode.FILTER_INVALID_VALUE,
    );
    await expectOracleErrorCode(
      () => generateWhereClause({ score: { $between: [1] } }, []),
      OracleErrorCode.FILTER_INVALID_VALUE,
    );
    await expectOracleErrorCode(
      () => generateWhereClause({ score: { $weird: 1 } }, []),
      OracleErrorCode.FILTER_UNSUPPORTED_OPERATOR,
    );
  });
});

describe("LangChainOracleError", () => {
  test("preserves the no rows found message while exposing a stable code", () => {
    const error = new LangChainOracleError(
      OracleErrorCode.QUERY_NO_ROWS_FOUND,
      "No rows found."
    );

    expect(error.message).toBe("No rows found.");
    expect(error.code).toBe(OracleErrorCode.QUERY_NO_ROWS_FOUND);
    expect(error.name).toBe("LangChainOracleError");
  });

  test("preserves Error subclassing semantics", () => {
    const error = new LangChainOracleError(
      OracleErrorCode.SYSTEM_ERROR,
      "system failure"
    );

    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(LangChainOracleError);
  });

  test("exposes a DBError-compatible code", () => {
    const error = new LangChainOracleError(
      OracleErrorCode.SYSTEM_ERROR,
      "system failure"
    );

    const dbError = error as DBError;
    expect(dbError.code).toBe(OracleErrorCode.SYSTEM_ERROR);
  });
});

describe("Oracle error codes", () => {
  test("covers validation identifier and missing parameter errors", async () => {
    await expectOracleErrorCode(
      () =>
        new OracleVS(embeddings as never, {
          client: {} as never,
          tableName: 'bad"name',
          query: "q",
        }),
      OracleErrorCode.VALIDATION_INVALID_IDENTIFIER,
    );

    await expectOracleErrorCode(
      OracleVS.fromDocuments([], embeddings as never, {
        tableName: "docs",
        query: "q",
      } as never),
      OracleErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER,
    );
  });

  test("covers vector configuration and index parameter errors", async () => {
    await expectOracleErrorCode(
      createTable({} as never, "docs", null),
      OracleErrorCode.VECTOR_INVALID_CONFIGURATION,
    );
    await expectOracleErrorCode(
      createTable({} as never, "docs", 7, { format: "BINARY" as never }),
      OracleErrorCode.VECTOR_INVALID_CONFIGURATION,
    );
    await expectOracleErrorCode(
      createIndex(
        {} as never,
        { tableName: '"DOCS"', distanceStrategy: "COSINE" } as OracleVS,
        { bogus: true } as never,
      ),
      OracleErrorCode.VECTOR_INVALID_INDEX_PARAMETERS,
    );
  });

  test("covers vector value, state, and unsupported representation errors", async () => {
    const store = new OracleVS(embeddings as never, {
      client: {
        executeMany: async () => undefined,
        commit: async () => undefined,
        close: async () => undefined,
      } as never,
      tableName: "docs",
      query: "q",
      format: "INT8" as never,
    });
    const storeInternals = store as unknown as OracleVSInternals;

    await expectOracleErrorCode(
      () => storeInternals.ensureEmbeddingDimension(),
      OracleErrorCode.STATE_INVALID,
    );

    store.embeddingDimension = 2;
    await expectOracleErrorCode(
      store.addVectors([[300, 1]], [new Document({ pageContent: "x", metadata: {} })]),
      OracleErrorCode.VECTOR_INVALID_VALUE,
    );
    await expectOracleErrorCode(
      () => storeInternals.normalizeReturnedEmbedding({ bad: true }),
      OracleErrorCode.VECTOR_UNSUPPORTED_REPRESENTATION,
    );
  });

  test("covers validation input errors and returns no matches for query no rows", async () => {
    const store = new OracleVS(embeddings as never, {
      client: {
        execute: async () => ({ rows: [] }),
        close: async () => undefined,
      } as never,
      tableName: "docs",
      query: "q",
    });

    await expectOracleErrorCode(
      store.addVectors([], []),
      OracleErrorCode.VALIDATION_INVALID_INPUT,
    );
    await expectOracleErrorCode(
      store.addVectors(
        [[1, 2]],
        [new Document({ pageContent: "x", metadata: {} })],
        { ids: ["a", "b"] },
      ),
      OracleErrorCode.VALIDATION_INVALID_INPUT,
    );
    await expect(
      store.similaritySearchByVectorReturningEmbeddings([1, 2], 1),
    ).resolves.toEqual([]);
  });

  test("covers metadata key and document loader validation error codes", async () => {
    await expectOracleErrorCode(
      () => generateWhereClause({ ["author') OR 1=1 --"]: "alice" }, []),
      OracleErrorCode.FILTER_INVALID_METADATA_KEY,
    );

    const loader = new OracleDocLoader({} as never, {});
    await expectOracleErrorCode(
      loader.load(),
      OracleErrorCode.VALIDATION_INVALID_INPUT,
    );

    const sqlLoader = new OracleDocLoader(
      {
        execute: async () => {
          throw new Error("bad identifier");
        },
      } as never,
      { owner: "OWNER", tablename: "DOCS", colname: "CONTENT" },
    );
    const sqlLoaderInternals = sqlLoader as unknown as OracleDocLoaderInternals;
    await expectOracleErrorCode(
      sqlLoaderInternals._loadFromTable("OWNER", "DOCS", "CONTENT"),
      OracleErrorCode.VALIDATION_INVALID_IDENTIFIER,
    );
  });

  test("covers system fallback error code", async () => {
    await expectOracleErrorCode(
      createTable(
        {
          execute: async () => {
            throw { name: "RuntimeError", message: "boom" };
          },
        } as never,
        "docs",
        2,
      ),
      OracleErrorCode.SYSTEM_ERROR,
    );

    await expectOracleErrorCode(
      dropTablePurge(
        {
          execute: async () => {
            throw { name: "ValidationError", message: "bad input" };
          },
        } as never,
        "docs",
      ),
      OracleErrorCode.SYSTEM_ERROR,
    );

    await expectOracleErrorCode(
      createIndex(
        {
          execute: async () => {
            throw { name: "OtherError", message: "surprise" };
          },
        } as never,
        { tableName: '"DOCS"', distanceStrategy: "COSINE" } as OracleVS,
      ),
      OracleErrorCode.SYSTEM_ERROR,
    );

    await expectOracleErrorCode(
      dropTablePurge(
        {
          execute: async () => {
            throw "plain failure";
          },
        } as never,
        "docs",
      ),
      OracleErrorCode.SYSTEM_ERROR,
    );
  });
});
