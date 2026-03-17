/* eslint-disable no-process-env */
import {
  test,
  expect,
  describe,
  afterAll,
  beforeAll,
  beforeEach,
} from "vitest";
import { env } from "node:process";
import { Document } from "@langchain/core/documents";
import oracledb from "oracledb";
import {
  createIndex,
  DistanceStrategy,
  dropTablePurge,
  OracleEmbeddings,
  type OracleDBVSArgs,
} from "../index.js";
import {
  OracleVS,
  type Metadata,
  VectorElementFormat,
  VectorType,
  createTable,
} from "../vectorstores.js";

type VectorColumnMetadata = {
  vectorDimensions?: number;
  vectorFormat?: string;
  vectorType?: string;
};

function formatTableNameForMetadata(tableName: string): string {
  if (tableName.startsWith('"') && tableName.endsWith('"')) {
    return tableName.slice(1, -1);
  }
  if (/[a-z]/.test(tableName)) {
    return tableName;
  }
  return tableName.toUpperCase();
}

async function getVectorColumnMetadata(
  connection: oracledb.Connection,
  tableName: string,
): Promise<VectorColumnMetadata> {
  const normalized = formatTableNameForMetadata(tableName);
  const candidates = Array.from(
    new Set([normalized, normalized.toUpperCase()]),
  );

  let ddlResult: oracledb.Result<unknown> | undefined;
  let lastError: unknown;

  for (const candidate of candidates) {
    try {
      ddlResult = await connection.execute(
        `SELECT DBMS_METADATA.GET_DDL('TABLE', :tableName, USER) AS DDL FROM dual`,
        [candidate],
        {
          outFormat: oracledb.OUT_FORMAT_OBJECT,
          fetchInfo: {
            DDL: { type: oracledb.STRING },
          },
        },
      );
      lastError = undefined;
      break;
    } catch (error) {
      lastError = error;
    }
  }

  if (!ddlResult) {
    throw lastError ?? new Error("Unable to fetch table metadata.");
  }

  const ddlRow = ddlResult.rows?.[0] as { DDL?: string; ddl?: string } | undefined;
  const ddlText = ddlRow?.DDL ?? ddlRow?.ddl ?? "";
  const match = ddlText.match(/"?(?:EMBEDDING)"?\s+VECTOR\(([^)]+)\)/i);
  if (!match) {
    throw new Error(`Unable to parse vector definition from DDL: ${normalized}`);
  }
  const parts = match[1]
    .split(",")
    .map((part) => part.trim().toUpperCase());

  const [lengthPart, formatPart = "FLOAT32", storagePart = "DENSE"] = parts;
  const parsedLength =
    lengthPart === VectorElementFormat.FLEX ? undefined : Number(lengthPart);

  return {
    vectorDimensions: parsedLength,
    vectorFormat: formatPart,
    vectorType: storagePart,
  };
}

function unpackBinaryVector(buffer: ArrayLike<number>, dimension: number): number[] {
  const bytes = Uint8Array.from(buffer);
  const values = new Array<number>(dimension).fill(0);
  for (let i = 0; i < dimension; i += 1) {
    const byteIndex = Math.floor(i / 8);
    const bitOffset = 7 - (i % 8);
    const byte = bytes[byteIndex];
    values[i] = (byte >> bitOffset) & 1;
  }
  return values;
}

describe("OracleVectorStore", () => {
  const tableName = "testlangchain_1";
  let pool: oracledb.Pool;
  let embedder: OracleEmbeddings;
  let connection: oracledb.Connection | undefined;
  let oraclevs: OracleVS | undefined;
  let dbConfig: OracleDBVSArgs;

  beforeAll(async () => {
    pool = await oracledb.createPool({
      user: env.ORACLE_USERNAME,
      password: env.ORACLE_PASSWORD,
      connectString: env.ORACLE_DSN,
    });

    const pref = {
      provider: "database",
      model: env.DEMO_ONNX_MODEL,
    };
    connection = await pool.getConnection();
    embedder = new OracleEmbeddings(connection, pref);
    dbConfig = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };
  });

  test("createIndex builds HNSW vector index with custom parameters", async () => {
    const hnswTable = `${tableName}_hnsw_idx`;
    await dropTablePurge(connection as oracledb.Connection, hnswTable);

    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      const docs = [
        new Document({ pageContent: "Graph search tuning" }),
        new Document({ pageContent: "Vector performance tricks" }),
      ];
      const hnswStore = await OracleVS.fromDocuments(docs, embedder, {
        ...dbConfig,
        tableName: hnswTable,
      });

      let creationError: unknown;
      try {
        await createIndex(localConnection, hnswStore, {
          idxName: "hnsw_vector_idx",
          idxType: "HNSW",
          neighbors: 8,
          efConstruction: 32,
          accuracy: 80,
          parallel: 1,
        });
      } catch (error) {
        creationError = error;
      }

      if (creationError) {
        const message = (creationError as Error).message ?? String(creationError);
        expect(message).toMatch(/ORA-51962/);
        return;
      }

      const indexResult = await localConnection.execute(
        `SELECT index_name FROM user_indexes WHERE table_name = :table AND index_name = :index`,
        {
          table: hnswTable.toUpperCase(),
          index: "HNSW_VECTOR_IDX",
        },
        { outFormat: oracledb.OUT_FORMAT_OBJECT },
      );
      const indexRow =
        indexResult.rows?.[0] as { INDEX_NAME?: string; index_name?: string } | undefined;
      const indexName = indexRow?.INDEX_NAME ?? indexRow?.index_name;
      expect(indexName?.toUpperCase()).toBe("HNSW_VECTOR_IDX");
    } finally {
      await dropTablePurge(connection as oracledb.Connection, hnswTable);
      await localConnection?.close();
    }
  });

  beforeEach(async () => {
    // Drop table for the next test.
    await dropTablePurge(connection as oracledb.Connection, tableName);
  });

  afterAll(async () => {
    await dropTablePurge(connection as oracledb.Connection, tableName);
    await connection?.close();
    await pool.close();
  });

  test("Test vectorstore fromDocuments", async () => {
    let connection: oracledb.Connection | undefined;

    try {
      connection = await pool.getConnection();
      const docs = [];
      docs.push(new Document({ pageContent: "I like soccer." }));
      docs.push(new Document({ pageContent: "I love Stephen King." }));

      oraclevs = await OracleVS.fromDocuments(docs, embedder, dbConfig);

      await createIndex(connection, oraclevs, {
        idxName: "embeddings_idx",
        idxType: "IVF",
        neighborPart: 64,
        accuracy: 90,
      });

      const embedding = await embedder.embedQuery(
        "What is your favourite sport?"
      );
      const matches = await oraclevs.similaritySearchVectorWithScore(
        embedding,
        1
      );

      expect(matches).toHaveLength(1);
    } finally {
      if (connection) {
        await connection?.close();
      }
    }
  });

  test("Test vectorstore addDocuments", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const docs = [
      new Document({ pageContent: "hello", metadata: { a: 2 } }),
      new Document({ pageContent: "car", metadata: { a: 1 } }),
      new Document({ pageContent: "adjective", metadata: { a: 1 } }),
      new Document({ pageContent: "hi", metadata: { a: 1 } }),
    ];
    await oraclevs.addDocuments(docs);
    const results1 = await oraclevs.similaritySearch("hello!", 1);
    expect(results1).toHaveLength(1);
    expect(results1).toEqual([
      expect.objectContaining({ metadata: { a: 2 }, pageContent: "hello" }),
    ]);

    const dbFilter = { a: 1 };
    const results2 = await oraclevs.similaritySearchWithScore(
      "hello!",
      1,
      dbFilter
    );
    expect(results2).toHaveLength(1);
  });

  test("Test vectorstore addDocuments upsert", async () => {
    let connection: oracledb.Connection | undefined;

    try {
      connection = await pool.getConnection();
      const oraclevs = new OracleVS(embedder, dbConfig);
      await oraclevs.initialize();

      const docId = "doc-upsert-1";
      const getRow = async () => {
        if (!connection) {
          throw new Error("Connection not available");
        }
        const result = await connection.execute(
          `SELECT external_id, text, metadata FROM "${tableName}" WHERE external_id = :id`,
          [docId],
          {
            outFormat: oracledb.OUT_FORMAT_OBJECT,
            fetchInfo: {
              TEXT: { type: oracledb.STRING },
            },
          }
        );
        return (result.rows?.[0] ?? null) as {
          EXTERNAL_ID?: string;
          external_id?: string;
          TEXT?: string;
          text?: string;
          METADATA?: Metadata;
          metadata?: Metadata;
        } | null;
      };

      await oraclevs.addDocuments(
        [
          new Document({
            pageContent: "Original content",
            metadata: { version: 1 },
          }),
        ],
        { ids: [docId] }
      );

      const initialRow = await getRow();
      expect(initialRow).toBeTruthy();
      const initialId = initialRow?.EXTERNAL_ID ?? initialRow?.external_id;
      expect(initialId).toBe(docId);
      const initialText = initialRow?.TEXT ?? initialRow?.text;
      expect(initialText).toBe("Original content");
      const initialMetadata = (initialRow?.METADATA ?? initialRow?.metadata) as
        | Metadata
        | undefined;
      expect(initialMetadata?.version).toBe(1);

      await expect(
        oraclevs.addDocuments(
          [
            new Document({
              pageContent: "Updated content",
              metadata: { version: 2, updated: true },
            }),
          ],
          { ids: [docId] }
        )
      ).rejects.toThrow(/ORA-00001|unique constraint/i);

      const rowAfterFailedInsert = await getRow();
      expect(rowAfterFailedInsert?.TEXT ?? rowAfterFailedInsert?.text).toBe(
        "Original content"
      );
      const metadataAfterFailedInsert = (
        rowAfterFailedInsert?.METADATA ?? rowAfterFailedInsert?.metadata
      ) as Metadata | undefined;
      expect(metadataAfterFailedInsert?.version).toBe(1);

      await oraclevs.addDocuments(
        [
          new Document({
            pageContent: "Updated content",
            metadata: { version: 2, updated: true },
          }),
        ],
        { ids: [docId], upsert: true }
      );

      const updatedRow = await getRow();
      expect(updatedRow).toBeTruthy();
      const updatedId = updatedRow?.EXTERNAL_ID ?? updatedRow?.external_id;
      expect(updatedId).toBe(docId);
      const updatedText = updatedRow?.TEXT ?? updatedRow?.text;
      expect(updatedText).toBe("Updated content");
      const updatedMetadata = (updatedRow?.METADATA ?? updatedRow?.metadata) as
        | Metadata
        | undefined;
      expect(updatedMetadata?.version).toBe(2);
      expect(updatedMetadata?.updated).toBe(true);
    } finally {
      await connection?.close();
    }
  });

  test("initialize applies table description and column annotations", async () => {
    const annotatedTable = `${tableName}_meta`;
    const computedDimensions = (await embedder.embedQuery(dbConfig.query)).length;
    const annotatedConfig: OracleDBVSArgs = {
      ...dbConfig,
      tableName: annotatedTable,
      description: "Integration test table description",
      annotations: {
        external_id: "External identifier for documents",
        metadata: "JSON metadata payload"
      },
      vectorType: VectorType.SPARSE,
      format: VectorElementFormat.FLOAT64,
    };

    await dropTablePurge(connection as oracledb.Connection, annotatedTable);

    const annotatedStore = new OracleVS(embedder, annotatedConfig);
    await annotatedStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();

      const tableCommentResult = await metaConnection.execute(
        `SELECT comments FROM user_tab_comments WHERE table_name = :tableName`,
        [annotatedTable],
        { outFormat: oracledb.OUT_FORMAT_OBJECT }
      );
      const tableCommentRow = tableCommentResult.rows?.[0] as
        | { COMMENTS?: string; comments?: string }
        | undefined;
      const tableComment =
        tableCommentRow?.COMMENTS ?? tableCommentRow?.comments ?? null;
      expect(tableComment).toBe("Integration test table description");

      const fetchColumnComment = async (
        columnName: string
      ): Promise<string | null> => {
        const columnResult = await metaConnection!.execute(
          `SELECT comments FROM user_col_comments WHERE table_name = :tableName AND column_name = :columnName`,
          [annotatedTable, columnName],
          { outFormat: oracledb.OUT_FORMAT_OBJECT }
        );
        const columnRow = columnResult.rows?.[0] as
          | { COMMENTS?: string; comments?: string }
          | undefined;
        return columnRow?.COMMENTS ?? columnRow?.comments ?? null;
      };

      const externalIdComment = await fetchColumnComment("EXTERNAL_ID");
      expect(externalIdComment).toBe("External identifier for documents");

      const metadataComment = await fetchColumnComment("METADATA");
      expect(metadataComment).toBe("JSON metadata payload");

      const meta = await getVectorColumnMetadata(metaConnection, annotatedTable);
      expect(meta.vectorDimensions).toBe(computedDimensions);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("FLOAT64");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("SPARSE");
    } finally {
      if (metaConnection) {
        await metaConnection.close();
      }
      await dropTablePurge(
        connection as oracledb.Connection,
        annotatedTable
      );
    }
  });

  test("initialize defaults to dense float32 vectors when unspecified", async () => {
    const defaultVectorTable = `${tableName}_dense_default`;
    await dropTablePurge(connection as oracledb.Connection, defaultVectorTable);

    const defaultStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: defaultVectorTable,
    });
    await defaultStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const meta = await getVectorColumnMetadata(metaConnection, defaultVectorTable);
      const expectedDim = defaultStore.embeddingDimension ?? 0;
      expect(meta.vectorDimensions).toBe(expectedDim);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("FLOAT32");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, defaultVectorTable);
    }
  });

  test("initialize supports dense binary vectors when embedding dimension is compatible", async () => {
    const binaryTable = `${tableName}_dense_binary`;
    await dropTablePurge(connection as oracledb.Connection, binaryTable);

    const binaryDimensions = (await embedder.embedQuery(dbConfig.query)).length;
    expect(binaryDimensions % 8).toBe(0);
    const binaryStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: binaryTable,
      vectorType: VectorType.DENSE,
      format: VectorElementFormat.BINARY,
    });
    await binaryStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const meta = await getVectorColumnMetadata(metaConnection, binaryTable);
      expect(meta.vectorDimensions).toBe(binaryDimensions);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("BINARY");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, binaryTable);
    }
  });

  test("initialize supports flex vector format with embedding dimensions", async () => {
    const flexTable = `${tableName}_dense_flex`;
    await dropTablePurge(connection as oracledb.Connection, flexTable);

    const flexDimensions = (await embedder.embedQuery(dbConfig.query)).length;
    const flexStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: flexTable,
      vectorType: VectorType.DENSE,
      format: VectorElementFormat.FLEX,
    });
    await flexStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const meta = await getVectorColumnMetadata(metaConnection, flexTable);
      expect(meta.vectorDimensions).toBe(flexDimensions);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("*");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, flexTable);
    }
  });

  test("addVectors stores dense binary vectors using configured format", async () => {
    const binaryTable = `${tableName}_dense_binary_store`;
    await dropTablePurge(connection as oracledb.Connection, binaryTable);

    const dimension = (await embedder.embedQuery(dbConfig.query)).length;
    expect(dimension % 8).toBe(0);
    const binaryStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: binaryTable,
      vectorType: VectorType.DENSE,
      format: VectorElementFormat.BINARY,
    });
    await binaryStore.initialize();

    const vector = new Array<number>(dimension).fill(0);
    for (let i = 0; i < dimension; i += 2) {
      vector[i] = 1;
    }

    await binaryStore.addVectors(
      [vector],
      [
        new Document({
          pageContent: "binary doc",
          metadata: {},
        }),
      ],
      { ids: ["binary-1"] },
    );

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const result = await metaConnection.execute(
        `SELECT embedding FROM ${binaryStore.tableName} WHERE external_id = :id`,
        ["binary-1"],
      );
      const row = result.rows?.[0] as unknown[] | undefined;
      const stored = row?.[0];
      expect(stored).toBeTruthy();
      const rawBytes = Buffer.isBuffer(stored)
        ? Uint8Array.from(stored)
        : ArrayBuffer.isView(stored)
        ? new Uint8Array(
            (stored as ArrayBufferView).buffer,
            (stored as ArrayBufferView).byteOffset,
            (stored as ArrayBufferView).byteLength,
          )
        : Uint8Array.from(stored as number[]);
      const unpacked = unpackBinaryVector(rawBytes, dimension);
      expect(unpacked).toEqual(vector);
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, binaryTable);
    }
  });

  test("addVectors converts dense embeddings to sparse storage", async () => {
    const sparseTable = `${tableName}_sparse_vectors`;
    await dropTablePurge(connection as oracledb.Connection, sparseTable);

    const dimension = (await embedder.embedQuery(dbConfig.query)).length;
    const sparseStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: sparseTable,
      vectorType: VectorType.SPARSE,
      format: VectorElementFormat.FLOAT32,
    });
    await sparseStore.initialize();

    const vector = new Array<number>(dimension).fill(0);
    vector[1] = 0.5;
    vector[dimension - 2] = -0.25;

    await sparseStore.addVectors(
      [vector],
      [
        new Document({
          pageContent: "sparse doc",
          metadata: {},
        }),
      ],
      { ids: ["sparse-1"] },
    );

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const result = await metaConnection.execute(
        `SELECT embedding FROM ${sparseStore.tableName} WHERE external_id = :id`,
        ["sparse-1"],
      );
      const row = result.rows?.[0] as unknown[] | undefined;
      const stored = row?.[0] as oracledb.SparseVector;
      expect(stored).toBeTruthy();
      const indices = Array.from(stored.indices ?? []);
      expect(indices).toEqual([1, dimension - 2]);
      const values = Array.from(stored.values ?? []);
      expect(values[0]).toBeCloseTo(0.5, 5);
      expect(values[1]).toBeCloseTo(-0.25, 5);
      expect(stored.numDimensions).toBe(dimension);
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, sparseTable);
    }
  });

  test("addVectors stores dense INT8 vectors with rounding", async () => {
    const int8Table = `${tableName}_dense_int8_store`;
    await dropTablePurge(connection as oracledb.Connection, int8Table);

    const dimension = (await embedder.embedQuery(dbConfig.query)).length;
    const int8Store = new OracleVS(embedder, {
      ...dbConfig,
      tableName: int8Table,
      vectorType: VectorType.DENSE,
      format: VectorElementFormat.INT8,
    });
    await int8Store.initialize();

    const vector = Array.from({ length: dimension }, (_, i) => {
      const cycle = i % 4;
      if (cycle === 0) return 62.6;
      if (cycle === 1) return -31.4;
      if (cycle === 2) return 0.49;
      return -80.9;
    });
    await int8Store.addVectors(
      [vector],
      [
        new Document({
          pageContent: "int8 doc",
          metadata: {},
        }),
      ],
      { ids: ["int8-1"] },
    );

    try {
      const searchResults =
        await int8Store.similaritySearchByVectorReturningEmbeddings(vector, 1);
      expect(searchResults).toHaveLength(1);
      const firstResult = searchResults[0];
      expect(firstResult[0].pageContent).toBe("int8 doc");
    } finally {
      await dropTablePurge(connection as oracledb.Connection, int8Table);
    }
  });

  test("addVectors falls back to metadata.id when ids option is omitted", async () => {
    const metadataIdTable = `${tableName}_metadata_id`;
    await dropTablePurge(connection as oracledb.Connection, metadataIdTable);

    const store = new OracleVS(embedder, {
      ...dbConfig,
      tableName: metadataIdTable,
    });
    await store.initialize();

    const docId = "metadata-derived-id";
    const doc = new Document({
      pageContent: "Uses metadata id",
      metadata: { id: docId, note: "stored from metadata" },
    });
    const vector = await embedder.embedQuery(doc.pageContent);

    await store.addVectors([vector], [doc]);

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const result = await metaConnection.execute(
        `SELECT external_id, metadata FROM ${store.tableName}`,
        [],
        { outFormat: oracledb.OUT_FORMAT_OBJECT },
      );
      const row =
        (result.rows?.[0] as
          | { EXTERNAL_ID?: string; external_id?: string; METADATA?: Metadata; metadata?: Metadata }
          | undefined) ?? {};
      const persistedId = row.EXTERNAL_ID ?? row.external_id;
      const persistedMetadata = row.METADATA ?? row.metadata;
      expect(persistedId).toBe(docId);
      expect(persistedMetadata).toMatchObject({ id: docId, note: "stored from metadata" });
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, metadataIdTable);
    }
  });

  test("initialize handles quoted table names via PL/SQL block", async () => {
    const quotedTable = `"Test Langchain Q"`;
    await dropTablePurge(connection as oracledb.Connection, quotedTable);

    const quotedStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: quotedTable,
      description: "Quoted table description",
      annotations: {
        text: "Holds document text",
      },
    });

    await quotedStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      const sampleVector = await embedder.embedQuery("quoted table test");
      await quotedStore.addVectors(
        [sampleVector],
        [new Document({ pageContent: "quoted table test", metadata: {} })],
      );

      metaConnection = await pool.getConnection();
      const countResult = await metaConnection.execute(
        `SELECT COUNT(*) AS CNT FROM ${quotedTable}`,
        [],
        { outFormat: oracledb.OUT_FORMAT_OBJECT },
      );
      const row = countResult.rows?.[0] as { CNT?: number; cnt?: number } | undefined;
      const rowCount = row?.CNT ?? row?.cnt ?? 0;
      expect(rowCount).toBe(1);
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, quotedTable);
    }
  });

  test("createTable rejects unsupported sparse binary configuration", async () => {
    const validationTable = `${tableName}_invalid_binary`;
    let tempConnection: oracledb.Connection | undefined;
    try {
      tempConnection = await pool.getConnection();
      await expect(
        createTable(tempConnection, validationTable, 128, {
          vectorType: VectorType.SPARSE,
          format: VectorElementFormat.BINARY,
        })
      ).rejects.toThrow(/BINARY format is not supported for SPARSE vectors./i);
    } finally {
      await tempConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, validationTable);
    }
  });

  test("createTable rejects format without dimensions", async () => {
    const formatOnlyTable = `${tableName}_format_no_dim`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await expect(
        createTable(localConnection, formatOnlyTable, undefined, {
          format: VectorElementFormat.FLOAT64,
        })
      ).rejects.toThrow(/Embedding dimension is required/i);
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, formatOnlyTable);
    }
  });

  test("createTable rejects vector type without explicit format", async () => {
    const vectorTypeOnlyTable = `${tableName}_type_no_format`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await expect(
        createTable(localConnection, vectorTypeOnlyTable, 256, {
          vectorType: VectorType.SPARSE,
        })
      ).rejects.toThrow(/Sparse vector type requires a vector format/i);
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, vectorTypeOnlyTable);
    }
  });

  test("createTable defaults invalid vector type to dense", async () => {
    const fallbackTypeTable = `${tableName}_invalid_type`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await createTable(localConnection, fallbackTypeTable, 128, {
        // Force an invalid value at the type level
        vectorType: "nonsense" as unknown as VectorType,
      });

      const meta = await getVectorColumnMetadata(localConnection, fallbackTypeTable);
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, fallbackTypeTable);
    }
  });

  test("createTable defaults invalid vector format to float32", async () => {
    const fallbackFormatTable = `${tableName}_invalid_format`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await createTable(localConnection, fallbackFormatTable, 128, {
        vectorType: VectorType.DENSE,
        format: "float128" as unknown as VectorElementFormat,
      });

      const meta = await getVectorColumnMetadata(localConnection, fallbackFormatTable);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("FLOAT32");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, fallbackFormatTable);
    }
  });

  test("createTable honors explicit dimensions when embedding dimension is omitted", async () => {
    const explicitTable = `${tableName}_explicit_override`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await createTable(localConnection, explicitTable, 96, {
        vectorType: VectorType.DENSE,
        format: VectorElementFormat.FLOAT64,
      });

      const meta = await getVectorColumnMetadata(localConnection, explicitTable);
      expect(meta.vectorDimensions).toBe(96);
      expect((meta.vectorFormat ?? "").toUpperCase()).toBe("FLOAT64");
      expect((meta.vectorType ?? "").toUpperCase()).toBe("DENSE");
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, explicitTable);
    }
  });

  test("createTable rejects non-positive embedding dimension", async () => {
    const mismatchTable = `${tableName}_invalid_dim`;
    let localConnection: oracledb.Connection | undefined;
    try {
      localConnection = await pool.getConnection();
      await expect(
        createTable(localConnection, mismatchTable, 0, {
          vectorType: VectorType.DENSE,
          format: VectorElementFormat.FLOAT32,
        })
      ).rejects.toThrow(/Embedding dimension must be a positive integer/i);
    } finally {
      await localConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, mismatchTable);
    }
  });

  test("Test vectorstore addDocuments and find using filter IN and NIN Clause", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const makeDoc = (
      content: string,
      author: string | string[],
      category = "research/AI"
    ) => ({
      pageContent: content,
      metadata: {
        category,
        author: Array.isArray(author) ? author : [author],
        tags: ["AI", "ML"],
        status: "release",
      },
    });

    const docs = [
      makeDoc(
        "Alice discusses the application of machine learning and AI research in predicting football match outcomes.",
        ["Alice", "Bob"],
        "sports"
      ),
      makeDoc(
        "Geoffrey Hinton explores the future of deep learning and its impact on AI research.",
        "Geoffrey Hinton"
      ),
      makeDoc(
        "Yoshua Bengio presents breakthroughs in neural network architectures for natural language understanding.",
        "Yoshua Bengio"
      ),
      makeDoc(
        "Andrew Ng shares insights on scaling AI education to democratize access to machine learning tools.",
        "Andrew Ng"
      ),
    ];

    await oraclevs.addDocuments(docs);

    // $in clause
    let filter: Metadata = { author: { $in: ["Andrew Ng", "Demis Hassabis"] } };
    let results = await oraclevs.similaritySearch(
      "latest advances in AI research for education",
      1,
      filter
    );

    expect(results).toHaveLength(1);
    expect(results).toEqual([
      expect.objectContaining({
        metadata: {
          category: "research/AI",
          author: ["Andrew Ng"],
          tags: ["AI", "ML"],
          status: "release",
        },
        pageContent:
          "Andrew Ng shares insights on scaling AI education to democratize access to machine learning tools.",
      }),
    ]);

    // without $in clause
    filter = { author: ["Andrew Ng", "Demis Hassabis"] };
    results = await oraclevs.similaritySearch(
      "latest advances in AI research for education",
      1,
      filter
    );

    expect(results).toHaveLength(1);
    expect(results).toEqual([
      expect.objectContaining({
        metadata: {
          category: "research/AI",
          author: ["Andrew Ng"],
          tags: ["AI", "ML"],
          status: "release",
        },
        pageContent:
          "Andrew Ng shares insights on scaling AI education to democratize access to machine learning tools.",
      }),
    ]);

    // with $nin clause
    filter = { author: { $nin: ["Andrew Ng", "Demis Hassabis"] } };
    results = await oraclevs.similaritySearch(
      "latest advances in AI research for education",
      5,
      filter
    );

    expect(results).toHaveLength(3);
    expect(results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          metadata: {
            category: "research/AI",
            author: ["Geoffrey Hinton"],
            tags: ["AI", "ML"],
            status: "release",
          },
          pageContent:
            "Geoffrey Hinton explores the future of deep learning and its impact on AI research.",
        }),
        expect.objectContaining({
          metadata: {
            category: "research/AI",
            author: ["Yoshua Bengio"],
            tags: ["AI", "ML"],
            status: "release",
          },
          pageContent:
            "Yoshua Bengio presents breakthroughs in neural network architectures for natural language understanding.",
        }),

        expect.objectContaining({
          metadata: {
            category: "sports",
            author: ["Alice", "Bob"],
            tags: ["AI", "ML"],
            status: "release",
          },
          pageContent:
            "Alice discusses the application of machine learning and AI research in predicting football match outcomes.",
        }),
      ])
    );
  });

  test("should handle simple conditions with and without _and clause", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample documents
    const docs = [
      new Document({
        pageContent: "A thrilling fantasy novel with dragons and magic.",
        metadata: { category: "books", price: 15 },
      }),
      new Document({
        pageContent: "A guide to healthy cooking with fresh vegetables.",
        metadata: { category: "books", price: 25 },
      }),
      new Document({
        pageContent: "A strategy board game with medieval warfare theme.",
        metadata: { category: "games", price: 40 },
      }),
      new Document({
        pageContent: "A suspense narrative in Paris.",
        metadata: { category: "books", price: 10 },
      }),
    ];

    await oraclevs.addDocuments(docs);

    // FilterCondition to have keywords , key, oper, value..
    let filter: Metadata = {
      $and: [{ category: "books" }, { price: { $lte: 20 } }],
    };
    let results = await oraclevs.similaritySearch("test", 5, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(2);
    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
      expect(doc.metadata.price).toBeLessThanOrEqual(20);
    });

    // FilterCondition to have a simple filter
    filter = {
      $and: [{ category: "books" }],
    };
    results = await oraclevs.similaritySearch("test", 4, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(3); // gives all rows with category books

    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
    });

    // filter with out _and keyword
    filter = { category: "books" };
    results = await oraclevs.similaritySearch("test", 4, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(3); // gives all rows with category books
    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
    });

    // filter with simple key/value and comparision with out _and keyword
    filter = { category: "books", price: 10 };
    results = await oraclevs.similaritySearch("test", 4, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(1); // gives all rows with category books and price 10
    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
      expect(doc.metadata.price).toBe(10);
    });

    // filter with simple key/value
    filter = { price: 10 };
    results = await oraclevs.similaritySearch("test", 4, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(1); // gives all rows with price 10
    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
      expect(doc.metadata.price).toBe(10);
    });

    // filter with $between
    filter = { price: { $between: [10, 25] } };
    results = await oraclevs.similaritySearch("test", 4, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(3); // gives all rows with price between [10, 25]
    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
      expect(doc.metadata.price).toBeGreaterThanOrEqual(10);
      expect(doc.metadata.price).toBeLessThanOrEqual(25);
    });

    // filter with $exists
    filter = { price: { $exists: true } };
    results = await oraclevs.similaritySearch("test", 10, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(4); // gives all rows with price key

    // filter with $exists to return rows which do not contain price
    filter = { price: { $exists: false } };
    await expect(oraclevs.similaritySearch("test", 10, filter)).rejects.toThrow(
      "No rows found"
    );

    // filter with $exists for non-existing key
    filter = { cost: { $exists: true } };
    await expect(oraclevs.similaritySearch("test", 10, filter)).rejects.toThrow(
      "No rows found"
    );
  });

  test("should handle a simple _or clause", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample documents
    const docs = [
      new Document({
        pageContent: "A thrilling fantasy novel with dragons and magic.",
        metadata: { category: "books", price: 15 },
      }),
      new Document({
        pageContent: "A guide to healthy cooking with fresh vegetables.",
        metadata: { category: "books", price: 25 },
      }),
      new Document({
        pageContent: "A strategy board game with medieval warfare theme.",
        metadata: { category: "games", price: 15 },
      }),
      new Document({
        pageContent: "A suspense narrative in Paris.",
        metadata: { category: "books", price: 10 },
      }),
      new Document({
        pageContent:
          "A strategy board game with medieval Civil Constructions theme.",
        metadata: { category: "games", price: 40 },
      }),
    ];
    await oraclevs.addDocuments(docs);
    const filter: Metadata = {
      $or: [{ category: "books" }, { price: { $lte: 20 } }],
    };
    const results = await oraclevs.similaritySearch("test", 6, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(4);

    results.forEach((doc) => {
      expect(
        doc.metadata.price <= 20 || doc.metadata.category === "books"
      ).toBe(true);
    });
  });

  test("should handle a nested _and and _or clause", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample docs
    const docs = [
      new Document({
        id: "1",
        pageContent: "A thrilling mystery novel",
        metadata: { category: "books", price: 15, rating: 4.5 },
      }),
      new Document({
        id: "2",
        pageContent: "An expensive historical book",
        metadata: { category: "books", price: 35, rating: 4.7 },
      }),

      new Document({
        id: "3",
        pageContent: "Affordable cooking guide",
        metadata: { category: "books", price: 18, rating: 4.2 },
      }),

      new Document({
        id: "4",
        pageContent: "Wireless Bluetooth headphones",
        metadata: { category: "electronics", price: 50, rating: 4.1 },
      }),

      new Document({
        id: "5",
        pageContent: "Budget wired earphones",
        metadata: { category: "electronics", price: 15, rating: 3.9 },
      }),
    ];

    // Insert into the vector store with ids provided.
    await oraclevs.addDocuments(docs, { ids: ["1", "2", "3", "4", "5"] });

    const filter: Metadata = {
      $or: [
        // First OR branch: simple AND group
        {
          $and: [{ category: "books" }, { price: { $lte: 20 } }],
        },
        // Second OR branch: nested OR inside AND
        {
          $and: [
            { category: "electronics" },
            {
              $or: [{ price: { $lte: 20 } }, { rating: { $gte: 4.5 } }],
            },
          ],
        },
      ],
    };

    // Complex filter example with _or and nested _and/_or
    const results = await oraclevs.similaritySearch("test", 10, filter);
    expect(results).toHaveLength(3);
    expect(results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          pageContent: "A thrilling mystery novel",
          metadata: { category: "books", price: 15, rating: 4.5 },
          id: "1",
        }),
        expect.objectContaining({
          pageContent: "Budget wired earphones",
          metadata: { category: "electronics", price: 15, rating: 3.9 },
          id: "5",
        }),
        expect.objectContaining({
          pageContent: "Affordable cooking guide",
          metadata: { category: "books", price: 18, rating: 4.2 },
          id: "3",
        }),
      ])
    );
  });

  test("Test MMR search", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const documents = [
      {
        pageContent: "Top 10 beaches in Spain with golden sands",
        metadata: { country: "Spain" },
      },
      {
        pageContent: "Hidden gems: remote Greek islands you must visit",
        metadata: { country: "Greece" },
      },
      {
        pageContent: "Spain's Costa Brava: a detailed travel guide",
        metadata: { country: "Spain" },
      },
      {
        pageContent: "Best beaches in Croatia with crystal-clear waters",
        metadata: { country: "Croatia" },
      },
      {
        pageContent: "Budget travel tips for backpacking across Europe",
        metadata: { country: "General" },
      },
    ];

    await oraclevs.addDocuments(documents);
    const results = await oraclevs.maxMarginalRelevanceSearch(
      "best beaches in Europe",
      {
        k: 3,
      }
    );

    // Extract only page contents for checking
    const pageContents = results.map((r) => r.pageContent);

    // Should have 3 results
    expect(pageContents).toHaveLength(3);

    // Results should be relevant but not all from the same country
    const countries = new Set(results.map((r) => r.metadata.country));
    expect(countries.size).toBeGreaterThan(1); // ensures diversity

    // The top result should be highly relevant to "best beaches"
    expect(pageContents[0].toLowerCase()).toMatch(/beach|island/);
  });

  test("maxMarginalRelevanceSearchByVector uses supplied embeddings", async () => {
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const documents = [
      {
        pageContent: "Dense rainforest ecology report",
        metadata: { topic: "nature" },
      },
      {
        pageContent: "Urban planning guidelines for smart cities",
        metadata: { topic: "cities" },
      },
      {
        pageContent: "Conservation strategies for endangered species",
        metadata: { topic: "nature" },
      },
    ];

    await oraclevs.addDocuments(documents);

    const customQuery = await embedder.embedQuery("urban development in smart cities");
    const mmrResults = await oraclevs.maxMarginalRelevanceSearchByVector(
      customQuery,
      {
        k: 2,
        fetchK: 3,
      },
    );

    expect(mmrResults).toHaveLength(2);
    const topics = new Set(mmrResults.map((d) => d.metadata.topic));
    expect(topics).toContain("cities");
  });

  test("Delete document by id", async () => {
    let connection: oracledb.Connection | undefined;

    const documents = [
      { pageContent: "Hello", metadata: { a: 1 } },
      { pageContent: "Bye", metadata: { a: 2 } },
      { pageContent: "FIFO", metadata: { a: 3 } },
    ];

    try {
      connection = await pool.getConnection();
      const oraclevs = new OracleVS(embedder, dbConfig);
      await oraclevs.initialize();
      await oraclevs.addDocuments(documents);

      const getIds = async (): Promise<Buffer[]> => {
        const res = await connection!.execute(`SELECT id FROM "${tableName}"`);
        return (res.rows ?? []).map((row: unknown) => {
          if (
            !Array.isArray(row) ||
            row.length === 0 ||
            !Buffer.isBuffer(row[0])
          ) {
            throw new Error(`Unexpected row format: ${JSON.stringify(row)}`);
          }
          return row[0];
        });
      };

      const [id1, id2, idKeep] = await getIds();
      await oraclevs.delete({ ids: [id1, id2] });

      const idsAfterDelete = await getIds();
      expect(idsAfterDelete).toHaveLength(1);
      expect(idsAfterDelete).toContainEqual(idKeep);
      expect(idsAfterDelete).not.toContainEqual(id1);
      expect(idsAfterDelete).not.toContainEqual(id2);
    } finally {
      await connection?.close();
    }
  });
  test("buildVectorColumnDefinition uses embedding dimension when no override provided", async () => {
    const fallbackTable = `${tableName}_fallback_dim`;
    await dropTablePurge(connection as oracledb.Connection, fallbackTable);

    const fallbackStore = new OracleVS(embedder, {
      ...dbConfig,
      tableName: fallbackTable,
      vectorType: VectorType.DENSE,
      format: VectorElementFormat.FLOAT32,
    });
    await fallbackStore.initialize();

    let metaConnection: oracledb.Connection | undefined;
    try {
      metaConnection = await pool.getConnection();
      const meta = await getVectorColumnMetadata(metaConnection, fallbackTable);
      const expected = fallbackStore.embeddingDimension ?? 0;
      expect(meta.vectorDimensions).toBe(expected);
    } finally {
      await metaConnection?.close();
      await dropTablePurge(connection as oracledb.Connection, fallbackTable);
    }
  });

});
