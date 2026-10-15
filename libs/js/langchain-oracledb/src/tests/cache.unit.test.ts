import { describe, expect, test, vi } from "vitest";
import oracledb from "oracledb";
import { AIMessage } from "@langchain/core/messages";
import type { ChatGeneration, Generation } from "@langchain/core/outputs";
import {
  defaultHashKeyEncoder,
  serializeGeneration,
} from "@langchain/core/caches";
import {
  OracleCache,
  OracleSemanticCache,
  type OracleCacheClearOptions,
} from "../cache.js";
import { ErrorCode } from "../errors.js";

function mockConnection() {
  const execute = vi.fn().mockResolvedValue({ rows: [] });
  const executeMany = vi.fn().mockResolvedValue({});
  const commit = vi.fn().mockResolvedValue(undefined);
  const close = vi.fn().mockResolvedValue(undefined);
  const client = {
    execute,
    executeMany,
    commit,
    close
  } as unknown as oracledb.Connection;
  return { client, execute, executeMany, commit, close };
}

const embeddings = {
  embedQuery: async () => [1, 0, 0],
  embedDocuments: async (texts: string[]) => texts.map(() => [1, 0, 0])
};
const hash = (value: string) => defaultHashKeyEncoder(value);

describe.each(["exact", "semantic"] as const)("%s cache", (kind) => {
  function setup() {
    const mocks = mockConnection();
    const cache =
      kind === "exact"
        ? new OracleCache(mocks.client, "CACHE_TEST")
        : new OracleSemanticCache(embeddings, {
            client: mocks.client,
            tableName: "CACHE_TEST",
            query: "test"
          });
    function payload(value: unknown, distance = 0) {
      mocks.execute.mockResolvedValue({
        rows:
          kind === "exact"
            ? [{ generations: value }]
            : [
                [
                  "entry-id",
                  "p",
                  { return_val: value },
                  distance,
                  new Float32Array([1, 0, 0]),
                ]
              ]
      });
    }
    return { ...mocks, cache, payload };
  }

  test("rejects a missing client and invalid table identifier", () => {
    const make = (client: oracledb.Connection, tableName: string) =>
      kind === "exact"
        ? new OracleCache(client, tableName)
        : new OracleSemanticCache(embeddings, {
            client,
            tableName,
            query: "test"
          });
    expect(() =>
      make(null as unknown as oracledb.Connection, "CACHE_TEST")
    ).toThrow(/client.*required/);
    expect(() => make(mockConnection().client, 'bad"name')).toThrow(
      /not valid/
    );
  });

  test("returns null for a cache miss", async () => {
    const { cache } = setup();

    expect(await cache.lookup("p", "l")).toBeNull();
  });

  test("returns null for malformed JSON", async () => {
    const { cache, payload } = setup();
    payload("not-json");

    await expect(cache.lookup("p", "l")).resolves.toBeNull();
  });

  test("returns null for a TypeError during message decoding", async () => {
    const { cache, payload } = setup();
    const stored = [{ text: "v", message: null }];
    payload(kind === "exact" ? JSON.stringify(stored) : stored);
    await expect(cache.lookup("p", "l")).resolves.toBeNull();
  });

  test("classifies other message decoding errors and preserves the cause", async () => {
    const { cache, payload } = setup();
    const stored = [{ text: "v", message: { type: "unsupported", data: {} } }];
    payload(kind === "exact" ? JSON.stringify(stored) : stored);
    await expect(cache.lookup("p", "l")).rejects.toMatchObject({
      code: ErrorCode.CACHE_DESERIALIZATION_FAILED,
      cause: expect.objectContaining({
        message: "Got unexpected type: unsupported",
      }),
    });
  });

  if (kind === "exact") {
    test("looks up an exact-cache entry by its hashed ID", async () => {
      const { cache, execute } = setup();

      await cache.lookup("p", "l");

      expect(execute).toHaveBeenCalledWith(
        expect.stringContaining("WHERE id = :id"),
        { id: hash(JSON.stringify(["p", "l"])) },
        expect.any(Object)
      );
    });
  } else {
    test("filters a semantic-cache lookup by the hashed LLM key", async () => {
      const { cache, execute } = setup();

      await cache.lookup("p", "l");

      expect(execute).toHaveBeenCalledWith(
        expect.stringContaining("JSON_EXISTS"),
        expect.arrayContaining([hash("l"), 1]),
        expect.any(Object)
      );
    });
  }

  test("round-trips multiple generations through update and lookup", async () => {
    const { cache, execute, executeMany, payload, commit } = setup();
    const values: Generation[] = [{ text: "hello" }, { text: "world" }];
    await cache.update("p", "l", values);
    if (kind === "exact") {
      const [sql, binds] = execute.mock.calls[0];
      expect(sql).toContain('MERGE INTO "CACHE_TEST"');
      expect(binds).toMatchObject({
        id: hash(JSON.stringify(["p", "l"])),
        prompt_hash: hash("p"),
        llm_key_hash: hash("l")
      });
      payload(binds.generations);
      expect(commit).toHaveBeenCalledOnce();
    } else {
      const [sql, rows] = executeMany.mock.calls[0];
      expect(sql).toContain('MERGE INTO "CACHE_TEST"');
      expect(rows[0].ext_id).toBe(
        hash(JSON.stringify(["p", "l"])).slice(0, 36)
      );
      expect(rows[0].metadata).toMatchObject({
        prompt_hash: hash("p"),
        llm_key_hash: hash("l"),
      });
      payload(rows[0].metadata.return_val);
      expect(commit).toHaveBeenCalledOnce();
    }
    expect(await cache.lookup("p", "l")).toEqual(values);
  });

  test("restores chat messages and resets cached IDs", async () => {
    const { cache, payload, execute } = setup();
    const value: ChatGeneration = {
      text: "hi",
      message: new AIMessage({ content: "hi", id: "lc_run--original" })
    };
    const stored = [serializeGeneration(value)];
    payload(kind === "exact" ? JSON.stringify(stored) : stored);
    const hit = await cache.lookup("p", "l");
    expect(hit).toHaveLength(1);
    expect((hit![0] as ChatGeneration).message).toBeInstanceOf(AIMessage);
    expect((hit![0] as ChatGeneration).message.content).toBe("hi");
    expect((hit![0] as ChatGeneration).message.id).toBeUndefined();
    if (kind === "exact") {
      expect(execute).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        {
          outFormat: oracledb.OUT_FORMAT_OBJECT,
          fetchInfo: { generations: { type: oracledb.STRING } },
        }
      );
    }
  });

  test.each([false, true])(
    "skips tool-call generations (invalid: %s)",
    async (invalid) => {
      const { cache, execute, executeMany } = setup();
      const message = new AIMessage({
        content: "",
        ...(invalid
          ? {
              invalid_tool_calls: [
                {
                  name: "t",
                  args: "{",
                  id: "x",
                  type: "invalid_tool_call" as const
                }
              ]
            }
          : {
              tool_calls: [
                { name: "t", args: {}, id: "x", type: "tool_call" as const },
              ]
            })
      });
      await cache.update("p", "l", [{ text: "", message } as ChatGeneration]);
      expect(execute).not.toHaveBeenCalled();
      expect(executeMany).not.toHaveBeenCalled();
    }
  );

  test.each([
    {},
    { prompt: "p" },
    { llmKey: "l" },
    { prompt: "p", llmKey: "l" },
  ])("clears only the requested filters: %j", async (options) => {
    const { cache, execute } = setup();
    await cache.clear(options);
    const [sql, binds] = execute.mock.calls[0];
    expect(sql).toContain('DELETE FROM "CACHE_TEST"');
    expect(Object.values(binds)).toEqual(Object.values(options).map(hash));
    expect(sql.includes("WHERE")).toBe(Object.keys(options).length > 0);
    if ("prompt" in options) expect(sql).toContain("prompt_hash");
    if ("llmKey" in options) expect(sql).toContain("llm_key_hash");
    if (Object.keys(options).length === 2) expect(sql).toContain(" AND ");
  });

  test("rejects unknown clear filters before deleting", async () => {
    const { cache, execute } = setup();
    await expect(
      cache.clear({ unknown: "x" } as OracleCacheClearOptions)
    ).rejects.toMatchObject({ code: ErrorCode.VALIDATION_INVALID_INPUT });
    expect(execute).not.toHaveBeenCalled();
  });

  test("reads legacy text dictionaries", async () => {
    const { cache, payload } = setup();
    const stored = [{ text: "legacy", generation_info: { kind: "old" } }];
    payload(kind === "exact" ? JSON.stringify(stored) : stored);
    expect(await cache.lookup("p", "l")).toEqual([{ text: "legacy" }]);
  });
});

describe("behavior specific to each cache", () => {
  test("creates the exact table only when initialized", async () => {
    const { client, execute, commit } = mockConnection();
    const cache = new OracleCache(client);
    expect(execute).not.toHaveBeenCalled();
    await cache.initialize();
    expect(execute).toHaveBeenCalledWith(
      expect.stringContaining(
        'CREATE TABLE IF NOT EXISTS "langchain_exact_cache"'
      )
    );
    expect(commit).toHaveBeenCalledOnce();
  });

  test("rejects negative semantic distance thresholds", () => {
    expect(
      () =>
        new OracleSemanticCache(
          embeddings,
          { client: mockConnection().client, query: "test" },
          { scoreThreshold: -0.1 }
        )
    ).toThrow(/non-negative/);
  });

  test("validates the index identifier during explicit initialization", async () => {
    const { client, execute } = mockConnection();
    const cache = new OracleSemanticCache(
      embeddings,
      { client, query: "test" },
      { createIndexIfMissing: true, indexName: 'bad"name' }
    );
    await expect(cache.initialize()).rejects.toMatchObject({
      code: ErrorCode.VALIDATION_INVALID_IDENTIFIER
    });
    expect(execute).toHaveBeenCalledWith(
      expect.stringContaining("CREATE TABLE")
    );
    expect(execute).toHaveBeenCalledTimes(1);
  });

  test("rejects a non-array semantic payload", async () => {
    const { client, execute } = mockConnection();
    execute.mockResolvedValue({
      rows: [
        ["entry-id", "p", { return_val: "{}" }, 0, new Float32Array([1, 0, 0])]
      ]
    });
    const cache = new OracleSemanticCache(embeddings, {
      client,
      query: "test",
    });
    expect(await cache.lookup("p", "l")).toBeNull();
  });

  test.each([
    [0, 0, true],
    [0, 5e-13, true],
    [0, 0.1, false],
    [0.2, 0.3, false],
    [undefined, 2, true],
  ])(
    "applies maximum distance %s to score %s (hit: %s)",
    async (scoreThreshold, score, hit) => {
      const { client, execute } = mockConnection();
      execute.mockResolvedValue({
        rows: [
          [
            "entry-id",
            "p",
            { return_val: [{ text: "v" }] },
            score,
            new Float32Array([1, 0, 0]),
          ]
        ]
      });
      const cache = new OracleSemanticCache(
        embeddings,
        {
          client,
          query: "test"
        },
        { scoreThreshold }
      );
      expect(await cache.lookup("p", "l")).toEqual(
        hit ? [{ text: "v" }] : null
      );
    }
  );
});
