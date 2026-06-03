import { describe, expect, test, vi } from "vitest";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type oracledb from "oracledb";
import {
  DistanceStrategy,
  OracleVS,
  type OracleDBVSArgs,
} from "../vectorstores.js";

describe("OracleVS SQL generation", () => {
  test("filtered similarity search keeps vector index hint in executed query", async () => {
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
      tableName: "ORAVS_DOCUMENTS",
      query: "test",
      distanceStrategy: DistanceStrategy.COSINE,
    };

    const store = new OracleVS(embeddings, dbConfig);
    await store.similaritySearchByVectorReturningEmbeddings(
      [0.1, 0.2, 0.3],
      4,
      { category: "research" }
    );

    const sql = execute.mock.calls[0][0] as string;
    expect(sql).toContain(
      'SELECT /*+ VECTOR_INDEX_TRANSFORM("ORAVS_DOCUMENTS") */'
    );
    expect(sql).toContain('FROM "ORAVS_DOCUMENTS"');
    expect(sql).toContain("JSON_EXISTS");
    expect(sql).toMatch(/ORDER BY distance FETCH APPROX FIRST :\d+ ROWS ONLY/);
  });
});
