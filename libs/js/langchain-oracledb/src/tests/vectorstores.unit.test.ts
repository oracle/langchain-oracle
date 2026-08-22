import { describe, expect, test, vi } from "vitest";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type oracledb from "oracledb";
import {
  DistanceStrategy,
  OracleVS,
  VectorElementFormat,
  VectorType,
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

describe("sparse vector element format (#297)", () => {
  // The dense-input oracledb.SparseVector constructor stores values as
  // Float64Array regardless of the typed array passed in, which made FLOAT32
  // and INT8 sparse query vectors fail with ORA-51812. These tests pin the
  // object-form construction that preserves the element format.
  const makeStore = (format: VectorElementFormat) => {
    const connection = {
      execute: vi.fn(),
      close: vi.fn(),
    } as unknown as oracledb.Connection;
    const embeddings = {
      embedDocuments: vi.fn(),
      embedQuery: vi.fn(),
    } as unknown as EmbeddingsInterface;
    return new OracleVS(embeddings, {
      client: connection,
      tableName: "SPARSE_FORMAT_TEST",
      query: "probe",
      vectorType: VectorType.SPARSE,
      format,
    });
  };

  const prepare = (store: OracleVS, vector: number[]) =>
    (
      store as unknown as {
        prepareVectorForStorage(vector: number[]): {
          values: Float32Array | Float64Array | Int8Array;
          indices: Uint32Array | number[];
          numDimensions: number;
        };
      }
    ).prepareVectorForStorage(vector);

  test("FLOAT32 sparse vectors keep Float32Array values", () => {
    const sv = prepare(makeStore(VectorElementFormat.FLOAT32), [1, 0, 2, 0.5]);

    expect(sv.values).toBeInstanceOf(Float32Array);
    expect(Array.from(sv.indices)).toEqual([0, 2, 3]);
    expect(Array.from(sv.values)).toEqual([1, 2, 0.5]);
    expect(sv.numDimensions).toBe(4);
  });

  test("FLOAT64 sparse vectors keep Float64Array values", () => {
    const sv = prepare(makeStore(VectorElementFormat.FLOAT64), [0, 1.25, 0]);

    expect(sv.values).toBeInstanceOf(Float64Array);
    expect(Array.from(sv.indices)).toEqual([1]);
    expect(Array.from(sv.values)).toEqual([1.25]);
    expect(sv.numDimensions).toBe(3);
  });

  test("INT8 sparse vectors keep Int8Array values and drop rounded zeros", () => {
    const sv = prepare(makeStore(VectorElementFormat.INT8), [1.4, 0.3, -2, 0]);

    expect(sv.values).toBeInstanceOf(Int8Array);
    // 0.3 rounds to 0 and is dropped, matching the dense representation.
    expect(Array.from(sv.indices)).toEqual([0, 2]);
    expect(Array.from(sv.values)).toEqual([1, -2]);
    expect(sv.numDimensions).toBe(4);
  });

  test("INT8 sparse vectors reject out-of-range values", () => {
    expect(() => prepare(makeStore(VectorElementFormat.INT8), [1, 300, 0])).toThrow(
      /INT8 sparse vector values/
    );
  });

  test("all-zero sparse vectors produce empty indices and values", () => {
    const sv = prepare(makeStore(VectorElementFormat.FLOAT32), [0, 0, 0]);

    expect(Array.from(sv.indices)).toEqual([]);
    expect(Array.from(sv.values)).toEqual([]);
    expect(sv.numDimensions).toBe(3);
  });
});
