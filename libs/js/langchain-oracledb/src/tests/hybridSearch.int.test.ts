import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  test,
} from "vitest";
import { randomUUID } from "node:crypto";
import { env } from "node:process";
import oracledb from "oracledb";
import { Document } from "@langchain/core/documents";
import {
  OracleEmbeddings,
  ErrorCode,
  LangChainOracleError,
  OracleVS,
  DistanceStrategy,
  OracleHybridSearchRetriever,
  OracleVectorizerPreference,
  createHybridIndex,
  dropPreference,
  dropTablePurge,
  type HybridIndexParams,
  type OracleHybridSearchRetrieverInput,
} from "../index.js";
import { indexExist, quoteIdentifier, type Client } from "../utils.js";

const hasCredentials = Boolean(
  env.ORACLE_USERNAME && env.ORACLE_PASSWORD && env.ORACLE_DSN
);

const databaseDocs: Document[] = [
  new Document({
    pageContent:
      "If the answer to any preceding questions about database is yes, then say yes",
    metadata: {
      id: "cncpt_15.5.3.2.2_P4",
      link: "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-5387D7B2-C0CA-4C1E-811B-C7EB9B636442",
    },
  }),
  new Document({
    pageContent:
      "A tablespace can be online (accessible) or offline (not accessible) whenever the database is open.",
    metadata: {
      id: "cncpt_15.5.5_P1",
      link: "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-D02B2220-E6F5-40D9-AFB5-BC69BCEF6CD4",
    },
  }),
];

const threeDocs: Document[] = [
  new Document({
    pageContent:
      "The tablespace can be online or offline whenever the database is open.",
    metadata: { id: "doc_tablespace" },
  }),
  new Document({
    pageContent:
      "If the answer to any preceding questions about database is yes, say yes.",
    metadata: { id: "doc_questions" },
  }),
  new Document({
    pageContent:
      "Completely unrelated content about cooking recipes and kitchen tools.",
    metadata: { id: "doc_cooking" },
  }),
];

const refundDocs: Document[] = [
  new Document({
    pageContent:
      "Our refund policy for premium plan allows refunds within 30 days.",
    metadata: { id: "doc_refund", customer_id: "CUST_A" },
  }),
  new Document({
    pageContent: "Please review the latest SLA describing uptime commitments.",
    metadata: { id: "doc_sla", customer_id: "CUST_B" },
  }),
];

const recoveryDoc = new Document({
  pageContent: Array.from(
    { length: 8 },
    (_, i) =>
      `Section ${
        i + 1
      }. RMAN restores database backups after hardware failure. ` +
      "Recovery procedures protect business records and bring systems online. " +
      "Administrators verify archived logs and validate backup files before restoring data. " +
      "Database recovery requires careful planning and regular testing."
  ).join(" "),
  metadata: {
    id: "doc_recovery",
    category: "operations",
    chunk_id: "stored",
    score: -1,
  },
});

const longRecoverySentence =
  "Administrators restore database backups and validate archived logs during recovery. ";

const longRecoveryDoc = new Document({
  pageContent: longRecoverySentence.repeat(100),
  metadata: { id: "long_document" },
});

describe.skipIf(!hasCredentials)("hybrid search integration", () => {
  let connection: oracledb.Connection;
  let pool: oracledb.Pool;
  let embeddings: OracleEmbeddings;
  let tableName: string;
  let idxName: string;
  let prefName: string;
  let preferenceCreated: boolean;

  beforeAll(async () => {
    if (!env.DEMO_ONNX_MODEL)
      throw new Error(
        "DEMO_ONNX_MODEL is required for hybrid integration tests"
      );
    pool = await oracledb.createPool({
      user: env.ORACLE_USERNAME,
      password: env.ORACLE_PASSWORD,
      connectString: env.ORACLE_DSN,
    });
  });

  beforeEach(async () => {
    connection = await pool.getConnection();
    embeddings = new OracleEmbeddings(connection, {
      provider: "database",
      model: env.DEMO_ONNX_MODEL,
    });
    const suffix = randomUUID().replace(/-/g, "").slice(0, 12).toUpperCase();
    tableName = `TB_HY_${suffix}`;
    idxName = `IDX_HY_${suffix}`;
    prefName = `PREF_HY_${suffix}`;
    preferenceCreated = false;
  });

  afterEach(async () => {
    if (!connection) return;
    const errors: unknown[] = [];
    try {
      await dropTablePurge(connection, tableName);
    } catch (error) {
      errors.push(error);
    }

    if (preferenceCreated) {
      try {
        await dropPreference(connection, prefName);
      } catch (error) {
        errors.push(error);
      }
    }

    try {
      await connection.close();
    } catch (error) {
      errors.push(error);
    }

    if (errors.length)
      throw new AggregateError(errors, "Hybrid integration cleanup failed");
  });

  afterAll(async () => {
    await pool?.close();
  });

  function createStore(client: Client = connection) {
    return new OracleVS(embeddings, {
      client,
      tableName,
      query: "database",
      distanceStrategy: DistanceStrategy.COSINE,
    });
  }

  async function storeFromDocuments(
    docs = databaseDocs,
    client: Client = connection
  ) {
    return OracleVS.fromDocuments(docs, embeddings, {
      client,
      tableName,
      query: "database",
      distanceStrategy: DistanceStrategy.COSINE,
    });
  }

  async function preference(
    vs: OracleVS,
    params: Record<string, unknown> = {}
  ) {
    const pref = await OracleVectorizerPreference.createPreference(
      vs,
      prefName,
      params
    );
    preferenceCreated = true;
    return pref;
  }

  async function createIndexedStore(
    docs = databaseDocs,
    options: HybridIndexParams = {},
    client: Client = connection
  ) {
    const vs = await storeFromDocuments(docs, client);
    const pref = await preference(vs);
    await createHybridIndex(client, idxName, {
      vectorizerPreference: pref,
      ...options,
    });

    return { vs, pref };
  }

  async function createChunkedIndexedStore() {
    const vs = await storeFromDocuments([recoveryDoc]);
    const pref = await preference(vs, { by: "words", max: 50, overlap: 0 });
    await createHybridIndex(connection, idxName, {
      vectorizerPreference: pref,
    });
    return vs;
  }

  function createRetriever(
    vs: OracleVS,
    config: Partial<OracleHybridSearchRetrieverInput> = {}
  ) {
    return new OracleHybridSearchRetriever({
      vectorStore: vs,
      idxName,
      ...config,
    });
  }

  function expectScores(doc: Document) {
    for (const key of ["score", "text_score", "vector_score"]) {
      expect(typeof doc.metadata[key]).toBe("number");
      expect(Number.isFinite(doc.metadata[key])).toBe(true);
    }
  }

  test("creates a named preference and rejects model/embedder mismatches", async () => {
    const vs = createStore();
    const pref = await preference(vs);
    expect(pref.preferenceName).toBe(prefName);
    for (const params of [
      { model: "non_existing_model" },
      {
        embedder_spec: {
          provider: "oracleai",
          url: "http://invalid.example/score",
          model: "wrong_model",
        },
      },
    ]) {
      await expect(
        OracleVectorizerPreference.createPreference(
          vs,
          `${prefName}_BAD`,
          params
        )
      ).rejects.toThrow(/must.*match/);
    }
  });

  test("creates a usable hybrid index with an explicit preference", async () => {
    const { vs } = await createIndexedStore();
    expect(await createRetriever(vs, { k: 1 }).invoke("database")).toHaveLength(
      1
    );
  });

  test("hybrid retrieval ranks matching documents and preserves metadata without scores", async () => {
    const { vs } = await createIndexedStore();
    const r = createRetriever(vs, { k: 2 });
    expect(r.searchMode).toBe("hybrid");
    const docs = await r.invoke("database questions");
    expect(docs).toHaveLength(2);
    expect(docs[0].metadata.id).toBe(databaseDocs[0].metadata.id);
    for (const doc of docs) {
      const source = databaseDocs.find(
        (d) => d.metadata.id === doc.metadata.id
      )!;
      expect(doc.pageContent).toBe(source.pageContent);
      expect(doc.metadata).toEqual(source.metadata);
      expect(doc.id).toEqual(expect.any(String));
    }
  });

  test.each([
    {
      mode: "semantic" as const,
      query: "tablespace",
      expected: 1,
      active: "vector_score",
      inactive: "text_score",
    },
    {
      mode: "keyword" as const,
      query: "preceding questions",
      expected: 0,
      active: "text_score",
      inactive: "vector_score",
    },
  ])(
    "$mode retrieval returns the expected document and component scores",
    async ({ mode, query, expected, active, inactive }) => {
      const { vs } = await createIndexedStore();
      const docs = await createRetriever(vs, {
        k: 1,
        searchMode: mode,
        returnScores: true,
      }).invoke(query);
      expect(docs).toHaveLength(1);
      expect(docs[0].metadata.id).toBe(databaseDocs[expected].metadata.id);
      expectScores(docs[0]);
      expect(docs[0].metadata[active]).toBeGreaterThan(0);
      expect(docs[0].metadata[inactive]).toBe(0);
    }
  );

  test.each([
    { query: "refund policy for premium plan", expectedId: "doc_refund" },
    { query: "latest SLA", expectedId: "doc_sla" },
  ])(
    "ranks $expectedId first with a PARALLEL 4 index",
    async ({ query, expectedId }) => {
      const { vs } = await createIndexedStore(refundDocs, { parallel: 4 });
      const docs = await createRetriever(vs, {
        k: 2,
        returnScores: true,
      }).invoke(query);
      expect(docs.length).toBeGreaterThanOrEqual(1);
      expect(docs[0].metadata.id).toBe(expectedId);
      expectScores(docs[0]);
    }
  );

  test("rejects invalid parallel and reserved DDL parameters, then creates the index", async () => {
    const vs = await storeFromDocuments();
    const pref = await preference(vs);
    await expect(
      createHybridIndex(connection, idxName, {
        vectorizerPreference: pref,
        parallel: "4" as unknown as number,
      })
    ).rejects.toThrow(/positive integer/);
    for (const key of [
      "model",
      "embedder_spec",
      "vectorizer",
      "vector_idxtype",
    ]) {
      await expect(
        createHybridIndex(connection, idxName, {
          vectorizerPreference: pref,
          parameters: { [key]: "bad" },
        })
      ).rejects.toThrow(/Vectorization parameters/);
    }
    await createHybridIndex(connection, idxName, {
      vectorizerPreference: pref,
    });
    expect(await createRetriever(vs, { k: 1 }).invoke("database")).toHaveLength(
      1
    );
  });

  test("call-time vector score weight changes the score without changing defaults", async () => {
    const { vs } = await createIndexedStore();
    const r = createRetriever(vs, { k: 1, returnScores: true });
    const weighted = await r.invoke("preceding questions", {
      params: { vector: { score_weight: 1 } },
    });
    const defaults = await r.invoke("preceding questions");
    expect(weighted).toHaveLength(1);
    expect(defaults).toHaveLength(1);
    expect(weighted[0].metadata.score).not.toBe(defaults[0].metadata.score);
    expect(r.params).toEqual({});
  });

  test("creates a usable index from a vector store", async () => {
    const vs = await storeFromDocuments();
    await createHybridIndex(connection, idxName, { vectorStore: vs });
    expect(
      (await createRetriever(vs, { k: 1 }).invoke("database")).length
    ).toBe(1);
  });

  test("creating the same index twice is idempotent and preserves retrieval", async () => {
    const { vs, pref } = await createIndexedStore(threeDocs);
    await createHybridIndex(connection, idxName, {
      vectorizerPreference: pref,
    });

    expect(
      (await createRetriever(vs, { k: 1 }).invoke("tablespace database")).length
    ).toBe(1);
  });

  test("index lookup distinguishes table schemas and resolves unqualified names", async () => {
    await createIndexedStore();
    const result = await connection.execute<[string]>(
      "SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM dual",
      {},
      { outFormat: oracledb.OUT_FORMAT_ARRAY }
    );
    const schema = result.rows![0][0];
    const index = quoteIdentifier(idxName);
    const table = quoteIdentifier(tableName);
    const qualifiedIndex = `${quoteIdentifier(schema)}.${index}`;
    const qualifiedTable = `${quoteIdentifier(schema)}.${table}`;

    expect(await indexExist(connection, index, table)).toBe(true);
    expect(await indexExist(connection, index, qualifiedTable)).toBe(true);
    expect(await indexExist(connection, qualifiedIndex, table)).toBe(true);
    expect(await indexExist(connection, qualifiedIndex, qualifiedTable)).toBe(
      true
    );

    // The same table name in another schema must not match this index.
    const otherSchema = quoteIdentifier(`OTHER_${randomUUID().slice(0, 8)}`);
    expect(
      await indexExist(connection, qualifiedIndex, `${otherSchema}.${table}`)
    ).toBe(false);
    expect(
      await indexExist(connection, `${otherSchema}.${index}`, qualifiedTable)
    ).toBe(false);
  });

  test("drops and recreates the preference and index", async () => {
    const { vs, pref } = await createIndexedStore(threeDocs);
    await pref.dropPreference();
    preferenceCreated = false;
    await connection.execute(`DROP INDEX ${quoteIdentifier(idxName)}`);
    const replacement = await preference(vs);
    await createHybridIndex(connection, idxName, {
      vectorizerPreference: replacement,
    });

    expect(
      (await createRetriever(vs, { k: 1 }).invoke("tablespace")).length
    ).toBe(1);
  });

  test("accepts PARALLEL 2 and retrieves documents", async () => {
    const { vs } = await createIndexedStore(threeDocs, { parallel: 2 });
    expect(
      (await createRetriever(vs, { k: 1 }).invoke("database")).length
    ).toBe(1);
  });

  test("call-time k overrides constructor k without changing later calls", async () => {
    const { vs } = await createIndexedStore(threeDocs);
    const r = createRetriever(vs, { k: 1 });
    expect(await r.invoke("database", { k: 3 })).toHaveLength(3);
    expect(await r.invoke("database")).toHaveLength(1);
  });

  test("handles k larger than the document count", async () => {
    const { vs } = await createIndexedStore(threeDocs);
    const docs = await createRetriever(vs, { k: 100 }).invoke(
      "database tablespace cooking"
    );
    expect(docs.length).toBeGreaterThanOrEqual(1);
    expect(docs.length).toBeLessThanOrEqual(threeDocs.length);
    expect(new Set(docs.map((d) => d.id)).size).toBe(docs.length);
  });

  test("empty fallback query raises a validation error and leaves retrieval usable", async () => {
    const { vs } = await createIndexedStore(threeDocs);
    const r = createRetriever(vs);
    const result = r.invoke("");
    await expect(result).rejects.toBeInstanceOf(LangChainOracleError);
    await expect(result).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_INPUT,
    });
    expect(await r.invoke("database", { k: 1 })).toHaveLength(1);
  });

  test("creates an index through a pool with a connection-backed vector store", async () => {
    const vs = await storeFromDocuments(threeDocs);
    const pref = await preference(vs);
    const baselineConnectionsInUse = pool.connectionsInUse;
    await createHybridIndex(pool, idxName, { vectorizerPreference: pref });
    expect(pool.connectionsInUse).toBe(baselineConnectionsInUse);
    expect(
      (await createRetriever(vs, { k: 1 }).invoke("tablespace")).length
    ).toBe(1);
  });

  test("uses separate semantic and keyword inputs instead of the fallback query", async () => {
    const { vs } = await createIndexedStore(threeDocs);
    const params = {
      vector: { search_text: "database tablespace" },
      text: { contains: "cooking" },
      search_fusion: "INTERSECT",
    };
    const r = createRetriever(vs, { k: 3, returnScores: true, params });
    const docs = await r.invoke("");
    expect(docs).toHaveLength(1);
    expect(docs[0].metadata.id).toBe("doc_cooking");
    expect(docs[0].metadata.text_score).toBeGreaterThan(0);
    expect(docs[0].metadata.vector_score).toBeGreaterThan(0);
    expect(
      await createRetriever(vs, { k: 3, returnScores: true }).invoke(
        "unrelated fallback",
        { params }
      )
    ).toEqual(docs);
    expect(r.params).toEqual(params);
  });

  test.each([
    { contains: "database AND questions", ids: ["doc_questions"] },
    { contains: "tablespace,cooking", ids: ["doc_cooking", "doc_tablespace"] },
    { contains: "cooking recipes", ids: ["doc_cooking"] },
  ])(
    "passes the contains query $contains to Oracle Text",
    async ({ contains, ids }) => {
      const { vs } = await createIndexedStore(threeDocs);
      const r = createRetriever(vs, { searchMode: "keyword", k: 3 });
      const docs = await r.invoke("", { params: { text: { contains } } });
      expect(docs.map((doc) => doc.metadata.id).sort()).toEqual(ids);
      for (const doc of docs) {
        expect(doc.pageContent).toBe(
          threeDocs.find((source) => source.metadata.id === doc.metadata.id)!
            .pageContent
        );
      }
      expect(r.params).toEqual({});
    }
  );

  test.each(["semantic", "keyword"] as const)(
    "%s search_text overrides query for just one invocation",
    async (searchMode) => {
      const { vs } = await createIndexedStore(threeDocs);
      const r = createRetriever(vs, { searchMode, k: 1, returnScores: true });
      const section = searchMode === "semantic" ? "vector" : "text";
      const expected = await r.invoke("cooking recipes");
      expect(expected[0].metadata.id).toBe("doc_cooking");
      const overridden = await r.invoke("tablespace", {
        params: { [section]: { search_text: "cooking recipes" } },
      });
      expect(overridden).toEqual(expected);
      expect((await r.invoke("tablespace"))[0].metadata.id).toBe(
        "doc_tablespace"
      );
      expect(r.params).toEqual({});
    }
  );

  test.each(["array", "string"] as const)(
    "a precomputed vector supplied as %s matches Oracle query embedding",
    async (format) => {
      const { vs } = await createIndexedStore(threeDocs);
      const query = "database tablespace";
      const vector = await embeddings.embedQuery(query);
      const searchVector = format === "array" ? vector : JSON.stringify(vector);
      const params = { vector: { search_vector: searchVector } };
      const original = structuredClone(params);
      const r = createRetriever(vs, {
        searchMode: "semantic",
        k: 3,
        returnScores: true,
      });
      const expected = await r.invoke(query);
      expect(expected).toHaveLength(3);
      const overridden = await r.invoke("", { params });
      const configured = await createRetriever(vs, {
        searchMode: "semantic",
        k: 3,
        returnScores: true,
        params,
      }).invoke("");
      for (const docs of [overridden, configured]) {
        expect(docs.map((doc) => [doc.id, doc.pageContent])).toEqual(
          expected.map((doc) => [doc.id, doc.pageContent])
        );
        docs.forEach((doc, i) => {
          expect(doc.metadata.vector_score).toBeCloseTo(
            expected[i].metadata.vector_score,
            5
          );
          expect(doc.metadata.text_score).toBe(0);
        });
      }
      expect(params).toEqual(original);
      expect(r.params).toEqual({});
    }
  );

  test.each([
    { name: "wrong vector dimension", search_vector: [0.1, 0.2] },
    { name: "invalid serialized vector", search_vector: "not a vector" },
  ])(
    "preserves the Oracle error for $name and recovers on the next call",
    async ({ search_vector }) => {
      const { vs } = await createIndexedStore(threeDocs);
      const r = createRetriever(vs, { searchMode: "semantic", k: 1 });
      await expect(
        r.invoke("", { params: { vector: { search_vector } } })
      ).rejects.toMatchObject({
        code: ErrorCode.SYSTEM_ERROR,
        cause: expect.objectContaining({
          message: expect.stringContaining("ORA-"),
        }),
      });
      expect((await r.invoke("tablespace"))[0].metadata.id).toBe(
        "doc_tablespace"
      );
      expect(r.params).toEqual({});
    }
  );

  test.each([true, false])(
    "CHUNK results preserve passages and source metadata with returnScores=%s",
    async (returnScores) => {
      const vs = await createChunkedIndexedStore();
      const r = createRetriever(vs, {
        searchMode: "semantic",
        k: 4,
        returnScores,
        params: { vector: { search_mode: "CHUNK" } },
      });
      const docs = await r.invoke("database recovery");
      expect(docs).toHaveLength(4);
      expect(new Set(docs.map((doc) => doc.id)).size).toBe(1);
      expect(new Set(docs.map((doc) => doc.metadata.chunk_id)).size).toBe(4);
      for (const doc of docs) {
        expect(doc.id).toBe("doc_recovery");
        expect(doc.metadata.id).toBe("doc_recovery");
        expect(doc.metadata.category).toBe("operations");
        expect(doc.metadata.chunk_id).toMatch(/^\d+$/);
        expect(doc.pageContent.length).toBeGreaterThan(0);
        expect(doc.pageContent.length).toBeLessThan(
          recoveryDoc.pageContent.length
        );
        expect(recoveryDoc.pageContent).toContain(doc.pageContent);
        if (returnScores) {
          expectScores(doc);
          expect(doc.metadata.score).toBeGreaterThan(0);
        } else {
          expect(doc.metadata.score).toBe(-1);
          expect(doc.metadata).not.toHaveProperty("vector_score");
          expect(doc.metadata).not.toHaveProperty("text_score");
        }
      }
      expect(recoveryDoc.metadata.chunk_id).toBe("stored");
      const documentResults = await r.invoke("database recovery", {
        params: { vector: { search_mode: "DOCUMENT" } },
      });
      expect(documentResults).toHaveLength(1);
      expect(documentResults[0].pageContent).toBe(recoveryDoc.pageContent);
      expect(documentResults[0].metadata.chunk_id).toBe("stored");
      expect(await r.invoke("database recovery")).toEqual(docs);
    }
  );

  test("hybrid CHUNK queries preserve both component scores", async () => {
    const vs = await createChunkedIndexedStore();
    const r = createRetriever(vs, {
      k: 4,
      returnScores: true,
      params: {
        vector: { search_mode: "CHUNK", search_text: "database recovery" },
        text: { contains: "RMAN" },
      },
    });
    const docs = await r.invoke("");
    expect(docs).toHaveLength(4);
    for (const doc of docs) {
      expect(doc.metadata.chunk_id).toMatch(/^\d+$/);
      expect(recoveryDoc.pageContent).toContain(doc.pageContent);
      expect(doc.pageContent).not.toBe(recoveryDoc.pageContent);
      expect(doc.metadata.vector_score).toBeGreaterThan(0);
      expect(doc.metadata.text_score).toBeGreaterThan(0);
    }
  });

  test("chunks a long source without truncating its stored document text", async () => {
    const vs = createStore();
    await vs.initialize();
    await vs.addVectors(
      [await embeddings.embedQuery(longRecoverySentence)],
      [longRecoveryDoc]
    );

    const pref = await preference(vs, { by: "words", max: 1000, overlap: 0 });
    await createHybridIndex(connection, idxName, {
      vectorizerPreference: pref,
    });

    const r = createRetriever(vs, {
      k: 20,
      searchMode: "semantic",
      params: { vector: { search_mode: "CHUNK" } },
    });

    const docs = await r.invoke("database recovery");

    expect(docs.length).toBeGreaterThan(1);
    expect(docs.some((doc) => doc.pageContent.length === 4000)).toBe(true);
    for (const doc of docs) {
      expect(longRecoveryDoc.pageContent).toContain(doc.pageContent);
      expect(doc.metadata.chunk_id).toMatch(/^\d+$/);
    }
    const ordered = [...docs].sort(
      (a, b) => Number(a.metadata.chunk_id) - Number(b.metadata.chunk_id)
    );
    expect(
      ordered
        .map((doc) => doc.pageContent)
        .join("")
        .replace(/\s+/g, "")
    ).toBe(longRecoveryDoc.pageContent.replace(/\s+/g, ""));
    const documentResults = await r.invoke("database recovery", {
      params: { vector: { search_mode: "DOCUMENT" } },
    });
    expect(documentResults).toHaveLength(1);
    expect(documentResults[0].pageContent).toBe(longRecoveryDoc.pageContent);
    expect(documentResults[0].metadata).toEqual(longRecoveryDoc.metadata);
  });

  test.each([
    { mode: "hybrid" as const, score: "score" },
    { mode: "semantic" as const, score: "vector_score" },
    { mode: "keyword" as const, score: "text_score" },
  ])("$mode returns scores in descending order", async ({ mode, score }) => {
    const { vs } = await createIndexedStore(threeDocs);
    const docs = await createRetriever(vs, {
      k: 3,
      searchMode: mode,
      returnScores: true,
    }).invoke("tablespace database");
    expect(docs.length).toBeGreaterThan(1);
    const scores = docs.map((d) => d.metadata[score] as number);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });
});
