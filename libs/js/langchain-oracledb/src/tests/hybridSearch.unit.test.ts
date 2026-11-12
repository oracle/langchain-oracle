import { describe, expect, test, vi } from "vitest";
import oracledb from "oracledb";
import { OracleEmbeddings } from "../embeddings.js";
import { OracleVS } from "../vectorstores.js";
import { ErrorCode, LangChainOracleError } from "../errors.js";
import {
  createHybridIndex,
  dropPreference,
  OracleHybridSearchRetriever,
  OracleVectorizerPreference,
  type HybridIndexParams,
  type HybridSearchParams,
  type OracleHybridSearchRetrieverInput,
} from "../hybridSearch.js";

function setup(
  pref: Record<string, unknown> = {
    provider: "database",
    model: "allminilm",
  }
) {
  const execute = vi.fn().mockResolvedValue({ rows: [] });
  const close = vi.fn().mockResolvedValue(undefined);
  const client = { execute, close } as unknown as oracledb.Connection;
  const embeddings = new OracleEmbeddings(client, pref);
  const vectorStore = new OracleVS(embeddings, {
    client,
    tableName: "MY_TABLE",
    query: "test",
  });

  const retriever = (config: Partial<OracleHybridSearchRetrieverInput> = {}) =>
    new OracleHybridSearchRetriever({
      vectorStore,
      idxName: "MY_IDX",
      ...config,
    });
  return { execute, close, client, embeddings, vectorStore, retriever };
}

async function indexSql(options: HybridIndexParams = {}) {
  const mocks = setup();
  const vectorizerPreference =
    await OracleVectorizerPreference.createPreference(
      mocks.vectorStore,
      "MY_PREF"
    );

  mocks.execute.mockClear();
  await createHybridIndex(mocks.client, "IDX", {
    vectorizerPreference,
    ...options,
  });
  const call = mocks.execute.mock.calls.find(([sql]) =>
    String(sql).startsWith("CREATE HYBRID VECTOR INDEX")
  );
  expect(call).toBeDefined();
  return String(call![0]);
}

const invalidParams: HybridSearchParams[] = [
  { search_text: "bad" },
  { search_vector: [0.1, 0.2] },
  { return: { topN: 5 } },
  { hybrid_index_name: "OTHER_IDX" },
  { vector: { contains: "bad" } },
  { text: { search_vector: [0.1, 0.2] } },
];

describe("OracleVectorizerPreference parameter validation", () => {
  test.each([{}, { some_other_key: "value" }, { model: "allminilm" }])(
    "resolves database model and preserves explicit parameters: %j",
    async (params) => {
      const { vectorStore, execute } = setup();
      const pref = await OracleVectorizerPreference.createPreference(
        vectorStore,
        "MY_PREF",
        params
      );
      expect(pref.params).toEqual({ ...params, model: "allminilm" });
      expect(pref.params).not.toHaveProperty("embedder_spec");
      expect(execute.mock.calls[0][1]).toEqual({
        preference_name: "MY_PREF",
        params: { val: pref.params, type: oracledb.DB_TYPE_JSON },
      });
    }
  );

  test.each([
    [{ provider: "database", model: "allminilm" }, { model: "other_model" }],
    [{ provider: "openai", model: "m" }, { model: "m" }],
    [
      { provider: "oracleai", model: "a", url: "http://host/score" },
      {
        embedder_spec: {
          provider: "oracleai",
          model: "b",
          url: "http://other/score",
        },
      },
    ],
  ])(
    "rejects mismatched embedding parameters before DB access: %j",
    async (embeddingParams, params) => {
      const { vectorStore, execute } = setup(embeddingParams);
      await expect(
        OracleVectorizerPreference.createPreference(
          vectorStore,
          "MY_PREF",
          params
        )
      ).rejects.toThrow(/must.*match/);
      expect(execute).not.toHaveBeenCalled();
    }
  );

  test.each([
    undefined,
    {
      embedder_spec: {
        model: "m",
        url: "http://host/score",
        provider: "oracleai",
      },
    },
  ])(
    "resolves external embedder, ignoring object key order: %j",
    async (params) => {
      const spec = {
        provider: "oracleai",
        url: "http://host/score",
        model: "m",
      };
      const { vectorStore } = setup(spec);
      const pref = await OracleVectorizerPreference.createPreference(
        vectorStore,
        "MY_PREF",
        params
      );
      expect(pref.params).toEqual({ embedder_spec: spec });
    }
  );

  test("rejects non-Oracle embeddings", async () => {
    const { client, execute } = setup();
    const store = new OracleVS(
      { embedQuery: vi.fn(), embedDocuments: vi.fn() },
      {
        client,
        tableName: "DOCS",
        query: "test",
      }
    );
    await expect(
      OracleVectorizerPreference.createPreference(store)
    ).rejects.toThrow(/Only OracleEmbeddings/);
    expect(execute).not.toHaveBeenCalled();
  });

  test("generates a preference name and does not mutate caller parameters", async () => {
    const { vectorStore } = setup();
    const params = { chunking: { by: "words" } };
    const pref = await OracleVectorizerPreference.createPreference(
      vectorStore,
      undefined,
      params
    );
    expect(pref.preferenceName).toMatch(/^PREF_[A-Za-z0-9]+$/);
    expect(params).toEqual({ chunking: { by: "words" } });
    expect(pref.params.chunking).not.toBe(params.chunking);
  });
});

describe("createHybridIndex DDL", () => {
  test("includes required clauses and omits optional clauses", async () => {
    const sql = await indexSql();
    expect(sql).toContain(
      'CREATE HYBRID VECTOR INDEX "IDX" ON "MY_TABLE"(text)'
    );
    expect(sql).toContain("PARAMETERS ('vectorizer MY_PREF')");
    expect(sql).not.toMatch(/FILTER BY|ORDER BY|PARALLEL/);
  });

  test.each([
    "model",
    "embedder_spec",
    "vectorizer",
    "vector_idxtype",
    "MODEL",
    "EMBEDDER_SPEC",
    "VECTORIZER",
    "VECTOR_IDXTYPE",
  ])("rejects reserved parameter %s", async (key) => {
    await expect(indexSql({ parameters: { [key]: "val" } })).rejects.toThrow(
      /Vectorization parameters must be given/
    );
  });

  test.each<[HybridIndexParams, string]>([
    [{ filterBy: ["col1", "col2"] }, 'FILTER BY "COL1","COL2"'],
    [{ orderBy: ["ts"], orderByAsc: true }, 'ORDER BY "TS" ASC'],
    [{ orderBy: ["ts"], orderByAsc: false }, 'ORDER BY "TS" DESC'],
    [
      { filterBy: ['"col1"'], orderBy: ['"ts"'] },
      'FILTER BY "col1"\nORDER BY "ts" ASC',
    ],
    [
      {
        filterBy: ["schema.col1", '"Mixed"."CaseCol"'],
        orderBy: ["schema.ts"],
      },
      'FILTER BY "SCHEMA"."COL1","Mixed"."CaseCol"\nORDER BY "SCHEMA"."TS" ASC',
    ],
    [{ parallel: 4 }, "PARALLEL 4"],
    [{ parameters: { word_min_len: 2 } }, "word_min_len 2"],
    [{ parameters: { lexer: "LEX'X" } }, "lexer LEX''X"],
  ])("formats optional clauses: %j", async (options, fragment) => {
    expect(await indexSql(options)).toContain(fragment);
  });

  test.each([
    "col name",
    "col,name",
    "col)",
    "col;drop",
    "col--comment",
    "schema.col name",
    '"bad name"',
  ])("rejects invalid filter identifier %s", async (identifier) => {
    await expect(indexSql({ filterBy: [identifier] })).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_IDENTIFIER,
    });
  });

  test.each([
    { filterBy: "col1" },
    { orderBy: [1] },
    { orderByAsc: "true" },
    ...["4", 4.5, true, false, 0, -1].map((parallel) => ({ parallel })),
  ])("rejects malformed options: %j", async (options) => {
    await expect(indexSql(options as HybridIndexParams)).rejects.toThrow();
  });

  test("rejects quotes in preference names before executing SQL", async () => {
    const { vectorStore, execute } = setup();
    await expect(
      OracleVectorizerPreference.createPreference(vectorStore, "PREF'X")
    ).rejects.toMatchObject({ code: ErrorCode.VALIDATION_INVALID_IDENTIFIER });
    expect(execute).not.toHaveBeenCalled();
  });
});

describe("OracleHybridSearchRetriever validation", () => {
  test("uses the default retriever settings", () => {
    expect(setup().retriever()).toMatchObject({
      searchMode: "hybrid",
      k: 4,
      returnScores: false,
      params: {},
      idxName: '"MY_IDX"',
    });
  });

  test.each(invalidParams)("rejects managed query parameters: %j", (params) => {
    expect(() => setup().retriever({ params })).toThrow(/cannot be supplied/);
  });

  test("preserves score weights", () => {
    const params = {
      vector: { score_weight: 0.7 },
      text: { score_weight: 0.3 },
    };
    expect(setup().retriever({ params }).params).toEqual(params);
  });

  test.each([{ params: "bad" }, { params: [] }, { params: 42 }])(
    "rejects malformed constructor params: %j",
    ({ params }) => {
      expect(() =>
        setup().retriever({ params: params as unknown as HybridSearchParams })
      ).toThrow(/params must be an object/);
    }
  );
});

describe("OracleHybridSearchRetriever search parameters", () => {
  test.each(["hybrid", "semantic", "keyword"] as const)(
    "builds %s search parameters",
    async (searchMode) => {
      const { retriever, execute } = setup();
      await retriever({ searchMode, k: 7 }).invoke("my query");
      const params = execute.mock.calls[0][1].search_params;
      expect(params.type).toBe(oracledb.DB_TYPE_JSON);
      expect(params.val).toEqual({
        hybrid_index_name: '"MY_IDX"',
        ...(searchMode !== "keyword"
          ? { vector: { search_text: "my query" } }
          : {}),
        ...(searchMode !== "semantic"
          ? { text: { search_text: "my query" } }
          : {}),
        return: {
          topN: 7,
          format: "JSON",
          values: ["rowid", "score", "vector_score", "text_score"],
        },
      });
    }
  );

  test("merges call-time parameters and k without modifying defaults", async () => {
    const { retriever, execute } = setup();
    const params = {
      vector: { score_weight: 0.7 },
      text: { score_weight: 0.3 },
    };
    const r = retriever({ params });
    await r.invoke("q", { k: 10, params: { vector: { score_weight: 0.8 } } });
    expect(execute.mock.calls[0][1].search_params.val).toMatchObject({
      vector: { score_weight: 0.8, search_text: "q" },
      text: { score_weight: 0.3, search_text: "q" },
      return: { topN: 10 },
    });
    await r.invoke("next");
    expect(execute.mock.calls[1][1].search_params.val).toMatchObject({
      vector: { score_weight: 0.7, search_text: "next" },
      return: { topN: 4 },
    });
    expect(r.params).toEqual(params);
    expect(params.vector).not.toHaveProperty("search_text");
  });

  test.each(invalidParams)(
    "rejects managed call-time parameters: %j",
    async (params) => {
      const { retriever, execute } = setup();
      await expect(retriever().invoke("q", { params })).rejects.toThrow(
        /cannot be supplied/
      );
      expect(execute).not.toHaveBeenCalled();
    }
  );

  test("falls back to k=4 for a null runtime value", async () => {
    const { retriever, execute } = setup();
    await retriever({ k: null as unknown as number }).invoke("q");
    expect(execute.mock.calls[0][1].search_params.val.return.topN).toBe(4);
  });

  test.each([{ params: "bad" }, { params: [] }, { params: 42 }])(
    "rejects malformed call-time params: %j",
    async ({ params }) => {
      const { retriever, execute } = setup();
      await expect(
        retriever().invoke("q", {
          params: params as unknown as HybridSearchParams,
        })
      ).rejects.toThrow();
      expect(execute).not.toHaveBeenCalled();
    }
  );
});

describe("hybrid search runtime validation", () => {
  test.each([
    { searchMode: "semantic" as const, params: { text: { contains: "RMAN" } } },
    {
      searchMode: "semantic" as const,
      params: { text: { search_text: "RMAN" } },
    },
    {
      searchMode: "keyword" as const,
      params: { vector: { search_text: "RMAN" } },
    },
    {
      searchMode: "keyword" as const,
      params: { vector: { search_vector: [0.1] } },
    },
  ])("rejects explicit inputs for an inactive section: %j", async (config) => {
    const { retriever, execute } = setup();
    expect(() => retriever(config)).toThrowError(
      expect.objectContaining({ code: ErrorCode.VALIDATION_INVALID_INPUT })
    );
    await expect(
      retriever({ searchMode: config.searchMode }).invoke("query", {
        params: config.params,
      })
    ).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_INPUT,
      message: expect.stringContaining(`searchMode "${config.searchMode}"`),
    });
    expect(execute).not.toHaveBeenCalled();
  });

  test.each([
    { vector: { search_text: "semantic query" } },
    { text: { contains: "RMAN" } },
  ])(
    "still requires query when another active section needs it: %j",
    async (params) => {
      const { retriever, execute } = setup();
      await expect(retriever({ params }).invoke("")).rejects.toMatchObject({
        code: ErrorCode.VALIDATION_INVALID_INPUT,
      });
      expect(execute).not.toHaveBeenCalled();
    }
  );

  test.each(["", "SEMANTIC", "other", 42, true])(
    "rejects an invalid searchMode %j before DB access",
    (searchMode) => {
      const { retriever, execute } = setup();
      expect(() =>
        retriever({
          searchMode:
            searchMode as OracleHybridSearchRetrieverInput["searchMode"],
        })
      ).toThrowError(
        expect.objectContaining({ code: ErrorCode.VALIDATION_INVALID_INPUT })
      );
      expect(execute).not.toHaveBeenCalled();
    }
  );
});

describe("hybrid search query inputs", () => {
  test.each([
    {
      name: "independent semantic and keyword queries",
      params: {
        vector: { search_text: "restore backups" },
        text: { contains: "RMAN AND recovery" },
      },
      vector: { search_text: "restore backups" },
      text: { contains: "RMAN AND recovery" },
    },
    {
      name: "keyword override with default semantic query",
      params: { text: { contains: "C, Python" } },
      vector: { search_text: "fallback" },
      text: { contains: "C, Python" },
    },
    {
      name: "semantic override with default keyword query",
      params: { vector: { search_text: "restore backups" } },
      vector: { search_text: "restore backups" },
      text: { search_text: "fallback" },
    },
    {
      name: "plain keyword text without expression conversion in the client",
      params: { text: { search_text: "  restore database backups  " } },
      vector: { search_text: "fallback" },
      text: { search_text: "  restore database backups  " },
    },
  ])(
    "sends $name at construction and invocation",
    async ({ params, vector, text }) => {
      const { retriever, execute } = setup();
      await retriever({ params }).invoke("fallback");
      await retriever().invoke("fallback", { params });
      expect(execute).toHaveBeenCalledTimes(2);
      for (const [sql, binds] of execute.mock.calls) {
        expect(binds.search_params.val.vector).toEqual(vector);
        expect(binds.search_params.val.text).toEqual(text);
        expect(sql).not.toContain("fallback");
      }
    }
  );

  test.each([{ searchVector: [0, -0.5, 1] }, { searchVector: "[0,-0.5,1]" }])(
    "passes a precomputed vector without embedding it: $searchVector",
    async ({ searchVector }) => {
      const { retriever, execute, embeddings } = setup();
      const embedQuery = vi.spyOn(embeddings, "embedQuery");
      const params = { vector: { search_vector: searchVector } };
      const original = structuredClone(params);
      const r = retriever({ searchMode: "semantic", params });
      await r.invoke("");
      await retriever({ searchMode: "semantic" }).invoke("", { params });
      for (const [, binds] of execute.mock.calls) {
        expect(binds.search_params.val.vector).toEqual({
          search_vector: "[0,-0.5,1]",
        });
        expect(binds.search_params.val).not.toHaveProperty("text");
      }
      expect(params).toEqual(original);
      expect(r.params).toEqual(original);
      expect(embedQuery).not.toHaveBeenCalled();
    }
  );

  test.each([
    {
      searchMode: "semantic" as const,
      params: { vector: { search_text: "restore backups" } },
    },
    { searchMode: "keyword" as const, params: { text: { contains: "RMAN" } } },
    {
      searchMode: "keyword" as const,
      params: { text: { search_text: "restore backups" } },
    },
    {
      searchMode: "hybrid" as const,
      params: {
        vector: { search_vector: [0.1, 0.2] },
        text: { contains: "RMAN" },
      },
    },
  ])("accepts an empty unused query: %j", async (config) => {
    const { retriever, execute } = setup();
    await expect(retriever(config).invoke("")).resolves.toEqual([]);
    expect(execute).toHaveBeenCalledTimes(1);
  });

  test.each(["semantic", "keyword"] as const)(
    "omits inactive tuning settings in %s mode",
    async (searchMode) => {
      const { retriever, execute } = setup();
      await retriever({
        searchMode,
        params: {
          vector: { score_weight: 2, search_mode: "CHUNK" },
          text: { score_weight: 3 },
        },
      }).invoke("query");
      const params = execute.mock.calls[0][1].search_params.val;
      expect(params).not.toHaveProperty(
        searchMode === "semantic" ? "text" : "vector"
      );
      expect(params.return.values.includes("chunk_text")).toBe(
        searchMode === "semantic"
      );
    }
  );

  test("undefined inputs fall back to query without treating them as conflicts", async () => {
    const { retriever, execute } = setup();
    await retriever({
      params: {
        vector: { search_text: undefined, search_vector: undefined },
        text: { search_text: undefined, contains: undefined },
      },
    }).invoke("query");
    const params = execute.mock.calls[0][1].search_params.val;
    expect(params.vector.search_text).toBe("query");
    expect(params.text.search_text).toBe("query");
  });
});

describe("hybrid search call isolation", () => {
  test("replaces a section rather than combining incompatible query inputs", async () => {
    const { retriever, execute } = setup();
    const r = retriever({
      params: {
        vector: { search_text: "default query", score_weight: 2 },
        text: { contains: "RMAN" },
        search_scorer: "RRF",
      },
    });
    await r.invoke("", { params: { vector: { search_vector: [0.1, 0.2] } } });
    expect(execute.mock.calls[0][1].search_params.val).toMatchObject({
      text: { contains: "RMAN" },
      search_scorer: "RRF",
    });
    expect(execute.mock.calls[0][1].search_params.val.vector).toEqual({
      search_vector: "[0.1,0.2]",
    });
    expect(r.params.vector).toEqual({
      search_text: "default query",
      score_weight: 2,
    });
  });

  test("preserves callbacks, tags and metadata when applying overrides", async () => {
    const { retriever } = setup();
    const start = vi.fn();
    const end = vi.fn();
    const r = retriever({
      callbacks: [{ handleRetrieverStart: start, handleRetrieverEnd: end }],
      tags: ["default-tag"],
      metadata: { source: "default" },
    });
    const callStart = vi.fn();
    await r.invoke("query", {
      k: 2,
      params: { text: { contains: "RMAN" } },
      callbacks: [{ handleRetrieverStart: callStart }],
      tags: ["call-tag"],
      metadata: { request: "one" },
      runName: "hybrid-query",
    });
    expect(start).toHaveBeenCalledTimes(1);
    expect(callStart).toHaveBeenCalledTimes(1);
    expect(end).toHaveBeenCalledTimes(1);
    expect(start.mock.calls[0][1]).toBe("query");
    expect(start.mock.calls[0][4]).toEqual(
      expect.arrayContaining(["default-tag", "call-tag"])
    );
    expect(start.mock.calls[0][5]).toMatchObject({
      source: "default",
      request: "one",
    });
    expect(start.mock.calls[0][6]).toBe("hybrid-query");
  });
});

describe("hybrid search chunk results", () => {
  test.each([true, false])(
    "maps distinct chunks from the same source with returnScores=%s",
    async (returnScores) => {
      const { retriever, execute } = setup();
      const stored = Object.freeze({
        category: "recovery",
        chunk_id: "stored",
        score: 99,
      });
      const longChunk = "matching passage ".repeat(400);
      execute.mockResolvedValue({
        rows: [
          ["doc-1", "full source", stored, 0.9, 0.8, 0.7, longChunk, "7"],
          [
            "doc-1",
            "full source",
            JSON.stringify(stored),
            0.6,
            null,
            0.5,
            "another passage",
            "2",
          ],
        ],
      });
      const docs = await retriever({
        returnScores,
        params: { vector: { search_mode: "CHUNK" } },
      }).invoke("query");
      expect(
        docs.map((d) => [d.id, d.pageContent, d.metadata.chunk_id])
      ).toEqual([
        ["doc-1", longChunk, "7"],
        ["doc-1", "another passage", "2"],
      ]);
      expect(docs.map((d) => d.metadata)).toEqual([
        {
          ...stored,
          chunk_id: "7",
          ...(returnScores
            ? { score: 0.9, text_score: 0.8, vector_score: 0.7 }
            : {}),
        },
        {
          ...stored,
          chunk_id: "2",
          ...(returnScores
            ? { score: 0.6, text_score: null, vector_score: 0.5 }
            : {}),
        },
      ]);
      expect(stored).toEqual({
        category: "recovery",
        chunk_id: "stored",
        score: 99,
      });
      const [sql, binds, options] = execute.mock.calls[0];
      expect(binds.search_params.val.return.values).toEqual([
        "rowid",
        "score",
        "vector_score",
        "text_score",
        "chunk_text",
        "chunk_id",
      ]);
      expect(sql).toContain("chunk_text CLOB PATH '$.chunk_text'");
      expect(sql).toContain("ORDER BY hit.result_order");
      expect(options.fetchInfo.CHUNK_TEXT.type).toBe(oracledb.STRING);
      expect(execute).toHaveBeenCalledTimes(1);
    }
  );

  test("falls back to source text for missing chunks without inventing chunk IDs", async () => {
    const { retriever, execute } = setup();
    execute.mockResolvedValue({
      rows: [
        ["a", "full source", { category: "text" }, 1, 1, null, null, null],
        ["b", null, null, 1, null, 1, null, "0"],
        ["c", "full source", null, 1, null, 1, "", "3"],
      ],
    });
    const docs = await retriever({
      params: { vector: { search_mode: "CHUNK" } },
    }).invoke("query");
    expect(docs.map((d) => [d.pageContent, d.metadata])).toEqual([
      ["full source", { category: "text" }],
      ["", { chunk_id: "0" }],
      ["", { chunk_id: "3" }],
    ]);
  });

  test.each(["CHUNK", "chunk", "Chunk"])(
    "enables chunk output for search_mode=%s",
    async (search_mode) => {
      const { retriever, execute } = setup();
      execute.mockResolvedValue({
        rows: [["a", "full source", null, 1, 1, 1, "passage", "1"]],
      });
      const r = retriever();
      expect(
        (await r.invoke("query", { params: { vector: { search_mode } } }))[0]
          .pageContent
      ).toBe("passage");
      expect(
        execute.mock.calls[0][1].search_params.val.return.values
      ).toContain("chunk_text");
      expect((await r.invoke("query"))[0].pageContent).toBe("full source");
      expect(
        execute.mock.calls[1][1].search_params.val.return.values
      ).not.toContain("chunk_text");
    }
  );

  test("a DOCUMENT override returns full text and preserves stored chunk metadata", async () => {
    const { retriever, execute } = setup();
    const metadata = { chunk_id: "stored" };
    execute.mockResolvedValue({
      rows: [["a", "full source", metadata, 1, 1, 1, "passage", "1"]],
    });
    const r = retriever({ params: { vector: { search_mode: "CHUNK" } } });
    const [doc] = await r.invoke("query", {
      params: { vector: { search_mode: "DOCUMENT" } },
    });
    expect(doc.pageContent).toBe("full source");
    expect(doc.metadata).toEqual(metadata);
    expect(
      execute.mock.calls[0][1].search_params.val.return.values
    ).not.toContain("chunk_id");
    expect(r.params.vector!.search_mode).toBe("CHUNK");
  });
});

describe("hybrid search results and lifecycle", () => {
  test.each([true, false])(
    "maps results with returnScores=%s, preserving rank and duplicate hits",
    async (returnScores) => {
      const { retriever, execute, close } = setup();
      const metadata = { category: "research" };
      execute.mockResolvedValue({
        rows: [
          ["b", "second document", metadata, 0.9, 0.8, 0.7],
          ["a", "first document", JSON.stringify(metadata), 0.6, null, 0.5],
          ["b", "second document", null, 0.4, 0.3, null],
        ],
      });
      const docs = await retriever({ returnScores }).invoke("q");
      expect(docs.map((d) => [d.id, d.pageContent])).toEqual([
        ["b", "second document"],
        ["a", "first document"],
        ["b", "second document"],
      ]);
      expect(docs.map((d) => d.metadata)).toEqual(
        returnScores
          ? [
              { ...metadata, score: 0.9, text_score: 0.8, vector_score: 0.7 },
              { ...metadata, score: 0.6, text_score: null, vector_score: 0.5 },
              { score: 0.4, text_score: 0.3, vector_score: null },
            ]
          : [metadata, metadata, {}]
      );
      expect(metadata).toEqual({ category: "research" });
      expect(execute).toHaveBeenCalledTimes(1);
      expect(execute.mock.calls[0][0]).toContain("ORDER BY hit.result_order");
      expect(close).not.toHaveBeenCalled();
    }
  );

  test.each([false, true])(
    "cleans up temporary preferences on DDL failure=%s",
    async (fails) => {
      const { client, execute, vectorStore, close } = setup();
      const error = new Error("DDL failed");
      execute.mockImplementation(async (sql: string) => {
        if (fails && sql.startsWith("CREATE HYBRID VECTOR INDEX")) throw error;
        return { rows: [] };
      });
      const result = createHybridIndex(client, "IDX", { vectorStore });
      if (fails)
        await expect(result).rejects.toMatchObject({
          code: ErrorCode.SYSTEM_ERROR,
          cause: error,
        });
      else await result;
      expect(execute).toHaveBeenCalledTimes(4);
      expect(execute.mock.calls[3][0]).toContain("DROP_PREFERENCE");
      expect(execute.mock.calls[3][1].preference_name).toBe(
        execute.mock.calls[1][1].preference_name
      );
      expect(close).not.toHaveBeenCalled();
    }
  );

  test("does not recreate an existing index", async () => {
    const { client, execute, vectorStore } = setup();
    execute.mockResolvedValue({ rows: [["IDX"]] });
    await createHybridIndex(client, "IDX", { vectorStore });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0][0]).toContain("all_indexes");
  });
});

describe("hybrid search error contract", () => {
  test("invalid parallel has a validation code and never executes DDL", async () => {
    const { client, vectorStore, execute } = setup();
    await expect(
      createHybridIndex(client, "IDX", { vectorStore, parallel: 0 })
    ).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_INPUT,
      message: "parallel must be a positive integer.",
    });
    expect(execute).not.toHaveBeenCalled();
  });

  test("invalid index, preference and column names use the identifier code", async () => {
    const { client, vectorStore, retriever, execute } = setup();
    expect(() => retriever({ idxName: 'bad"name' })).toThrowError(
      expect.objectContaining({ code: ErrorCode.VALIDATION_INVALID_IDENTIFIER })
    );
    await expect(
      createHybridIndex(client, 'bad"name', { vectorStore })
    ).rejects.toMatchObject({ code: ErrorCode.VALIDATION_INVALID_IDENTIFIER });
    await expect(
      OracleVectorizerPreference.createPreference(vectorStore, "bad name")
    ).rejects.toMatchObject({ code: ErrorCode.VALIDATION_INVALID_IDENTIFIER });
    await expect(dropPreference(client, "bad name")).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_IDENTIFIER,
    });
    await expect(
      createHybridIndex(client, "IDX", { vectorStore, orderBy: ["bad name"] })
    ).rejects.toMatchObject({ code: ErrorCode.VALIDATION_INVALID_IDENTIFIER });
    expect(execute).not.toHaveBeenCalled();
  });

  test("DDL and preference cleanup errors are preserved in one AggregateError cause", async () => {
    const { client, vectorStore, execute } = setup();
    const ddlError = new Error("DDL failed");
    const cleanupError = new Error("preference cleanup failed");
    execute.mockImplementation(async (sql: string) => {
      if (sql.startsWith("CREATE HYBRID VECTOR INDEX")) throw ddlError;
      if (sql.includes("DROP_PREFERENCE")) throw cleanupError;
      return { rows: [] };
    });
    const error = await createHybridIndex(client, "IDX", { vectorStore }).catch(
      (e: unknown) => e
    );
    expect(error).toMatchObject({ code: ErrorCode.SYSTEM_ERROR });
    const cause = (error as LangChainOracleError).cause as AggregateError;
    expect(cause).toBeInstanceOf(AggregateError);
    expect(cause.errors[0]).toBe(ddlError);
    expect(cause.errors[1]).toBe(cleanupError);
  });
});
