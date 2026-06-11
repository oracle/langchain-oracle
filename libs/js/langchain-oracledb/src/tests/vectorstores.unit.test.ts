import { describe, expect, test, vi } from "vitest";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type oracledb from "oracledb";
import {
  DistanceStrategy,
  OracleVS,
  type OracleDBVSArgs,
  type Metadata,
} from "../vectorstores.js";

async function getSimilaritySearchSql(
  tableName: string,
  filter?: Metadata
): Promise<string> {
  const execute = vi.fn().mockResolvedValue({
    rows: [
      [
        "doc-1",
        "Vector indexes with JSON metadata",
        { category: "research" },
        0.1,
        new Float32Array([0.1, 0.2, 0.3]),
      ],
    ],
  });
  const close = vi.fn().mockResolvedValue(undefined);
  const connection = { execute, close } as unknown as oracledb.Connection;
  const embeddings = {
    embedDocuments: vi.fn(),
    embedQuery: vi.fn(),
  } as unknown as EmbeddingsInterface;
  const dbConfig: OracleDBVSArgs = {
    client: connection,
    tableName,
    query: "test",
    distanceStrategy: DistanceStrategy.COSINE,
  };

  const store = new OracleVS(embeddings, dbConfig);
  await store.similaritySearchByVectorReturningEmbeddings(
    [0.1, 0.2, 0.3],
    4,
    filter
  );

  return execute.mock.calls[0][0] as string;
}

describe("OracleVS SQL generation", () => {
  test("unfiltered similarity search keeps vector index hint in executed query", async () => {
    const sql = await getSimilaritySearchSql("ORAVS_DOCUMENTS");

    expect(sql).toContain(
      'SELECT /*+ VECTOR_INDEX_TRANSFORM("ORAVS_DOCUMENTS") */'
    );
    expect(sql).toContain('FROM "ORAVS_DOCUMENTS"');
    expect(sql).not.toContain("JSON_EXISTS");
  });

  test("filtered similarity search omits vector index hint in executed query", async () => {
    const sql = await getSimilaritySearchSql("ORAVS_DOCUMENTS", {
      category: "research",
    });

    expect(sql).toContain("SELECT");
    expect(sql).not.toContain("VECTOR_INDEX_TRANSFORM");
    expect(sql).toContain('FROM "ORAVS_DOCUMENTS"');
    expect(sql).toContain("JSON_EXISTS");
    expect(sql).toMatch(/ORDER BY distance FETCH APPROX FIRST :\d+ ROWS ONLY/);
  });

  test("vector index hint uses caller supplied quoted table identifier", async () => {
    const quoted = '"My Vector Table"';
    const sql = await getSimilaritySearchSql(quoted);

    expect(sql).toContain(
      'SELECT /*+ VECTOR_INDEX_TRANSFORM("My Vector Table") */'
    );
    expect(sql).toContain(`FROM ${quoted}`);
  });
});
