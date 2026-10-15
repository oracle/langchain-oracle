import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  test,
} from "vitest";
import { createHash, randomUUID } from "node:crypto";
import { env } from "node:process";
import oracledb from "oracledb";
import { AIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import type { ChatGeneration } from "@langchain/core/outputs";
import { defaultHashKeyEncoder } from "@langchain/core/caches";
import {
  OracleCache,
  OracleSemanticCache,
  type OracleCacheClearOptions,
} from "../cache.js";
import { quoteIdentifier } from "../utils.js";
import { OracleVS } from "../vectorstores.js";

const hasCredentials = Boolean(
  env.ORACLE_USERNAME && env.ORACLE_PASSWORD && env.ORACLE_DSN
);

function vector(text: string): number[] {
  return Array.from(
    createHash("sha256").update(text).digest().subarray(0, 6),
    (byte) => (byte - 127.5) / 127.5
  );
}
const embeddings = {
  embedQuery: async (text: string) => vector(text),
  embedDocuments: async (texts: string[]) => texts.map(vector),
};

describe.skipIf(!hasCredentials).each(["exact", "semantic"] as const)(
  "%s cache integration",
  (kind) => {
    let pool: oracledb.Pool | undefined;
    let connection: oracledb.Connection | undefined;
    let tableName: string;
    let cache: OracleCache | OracleSemanticCache;

    beforeAll(async () => {
      pool = await oracledb.createPool({
        user: env.ORACLE_USERNAME,
        password: env.ORACLE_PASSWORD,
        connectString: env.ORACLE_DSN,
      });
    });

    beforeEach(async () => {
      tableName = `CACHE_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
      connection = await pool!.getConnection();
      cache =
        kind === "exact"
          ? new OracleCache(connection, tableName)
          : new OracleSemanticCache(embeddings, {
              client: connection,
              tableName,
              query: "test",
            });
      await cache.initialize();
    });

    afterEach(async () => {
      const current = connection;
      connection = undefined;
      if (current) {
        try {
          await OracleCache.dropTable(current, tableName);
        } finally {
          await current.close();
        }
      }
    });
    afterAll(async () => {
      await pool?.close();
    });

    async function countPrompt(prompt: string): Promise<number> {
      const column =
        kind === "exact"
          ? "prompt_hash"
          : "JSON_VALUE(metadata, '$.prompt_hash')";
      const result = await connection!.execute<{ count: number }>(
        `SELECT COUNT(*) AS "count" FROM ${quoteIdentifier(
          tableName
        )} WHERE ${column} = :hash`,
        { hash: defaultHashKeyEncoder(prompt) },
        { outFormat: oracledb.OUT_FORMAT_OBJECT }
      );
      return result.rows![0].count;
    }

    test("starts empty and misses an unknown prompt", async () => {
      expect(await cache.lookup("Sample prompt", "model-a")).toBeNull();
      expect(await cache.lookup("Nonexistent prompt", "model-a")).toBeNull();
    });

    test("stores and retrieves a generation", async () => {
      const prompt = "Sample prompt for testing";
      const llmKey = "test-model-a";
      const generation = {
        text: "Sample generated text.",
        generationInfo: { reason: "test" },
      };
      await cache.update(prompt, llmKey, [generation]);
      expect(await cache.lookup(prompt, llmKey)).toEqual([
        { text: "Sample generated text." },
      ]);
    });

    test("uses a fresh empty table after other tests", async () => {
      const prompt = "Sample prompt for testing";
      const llmKey = "test-model-a";
      expect(await cache.lookup(prompt, llmKey)).toBeNull();
    });

    test("round-trips multiple generations and clears the entire cache", async () => {
      const prompt = "Sample prompt for testing";
      const llmKey = "test-model-a";
      const values = [{ text: "hello" }, { text: "world" }];
      await cache.update(prompt, llmKey, values);
      expect(await cache.lookup(prompt, llmKey)).toEqual(values);
      await cache.update("other", "other-model", [{ text: "other" }]);
      await cache.clear();
      expect(await cache.lookup(prompt, llmKey)).toBeNull();
      expect(await cache.lookup("other", "other-model")).toBeNull();
    });

    test("isolates model keys", async () => {
      await cache.update("What is the capital of France?", "model-a", [
        { text: "Paris from A" },
      ]);
      await cache.update("What is the capital of France?", "model-b", [
        { text: "Paris from B" },
      ]);
      expect(
        await cache.lookup("What is the capital of France?", "model-a")
      ).toEqual([{ text: "Paris from A" }]);
      expect(
        await cache.lookup("What is the capital of France?", "model-b")
      ).toEqual([{ text: "Paris from B" }]);
      expect(
        await cache.lookup("What is the capital of France?", "unknown-model")
      ).toBeNull();
    });

    test("updates an existing entry without duplicating rows", async () => {
      await cache.update("p", "l", [{ text: "v1" }]);
      await cache.update("p", "l", [{ text: "v2" }]);
      expect(await cache.lookup("p", "l")).toEqual([{ text: "v2" }]);
      expect(await countPrompt("p")).toBe(1);
    });

    test("clears by model key then by exact prompt", async () => {
      await cache.update("p1", "l1", [{ text: "a" }]);
      await cache.update("p1", "l2", [{ text: "b" }]);
      await cache.update("p2", "l1", [{ text: "c" }]);
      await cache.clear({ llmKey: "l1" });
      expect(await cache.lookup("p1", "l1")).toBeNull();
      expect(await cache.lookup("p2", "l1")).toBeNull();
      expect(await cache.lookup("p1", "l2")).toEqual([{ text: "b" }]);
      await cache.clear({ prompt: "p1" });
      expect(await cache.lookup("p1", "l2")).toBeNull();
    });

    test("prompt clearing preserves other prompts for the same model", async () => {
      await cache.update("prompt one", "model-a", [{ text: "A" }]);
      await cache.update("prompt two", "model-a", [{ text: "B" }]);
      await cache.clear({ prompt: "prompt one" });
      expect(await countPrompt("prompt one")).toBe(0);
      expect(await countPrompt("prompt two")).toBe(1);
    });

    test("combines prompt and model filters", async () => {
      await cache.update("p1", "l1", [{ text: "a" }]);
      await cache.update("p1", "l2", [{ text: "b" }]);
      await cache.update("p2", "l1", [{ text: "c" }]);
      await cache.clear({ prompt: "p1", llmKey: "l1" });
      expect(await countPrompt("p1")).toBe(1);
      expect(await countPrompt("p2")).toBe(1);
      expect(await cache.lookup("p1", "l2")).toEqual([{ text: "b" }]);
    });

    test("rejects unsupported filters without deleting stored data", async () => {
      await cache.update("p", "l", [{ text: "keep" }]);
      await expect(
        cache.clear({ unknownFilter: "value" } as OracleCacheClearOptions)
      ).rejects.toThrow(/Unsupported clear options/);
      expect(await cache.lookup("p", "l")).toEqual([{ text: "keep" }]);
    });

    test("skips tool-call generations", async () => {
      const value: ChatGeneration = {
        text: "",
        message: new AIMessage({
          content: "",
          tool_calls: [{ name: "t", args: {}, id: "x", type: "tool_call" }],
        }),
      };
      await cache.update("p", "l", [value]);
      expect(await cache.lookup("p", "l")).toBeNull();
    });

    test("resets cached message IDs", async () => {
      const value: ChatGeneration = {
        text: "hi",
        message: new AIMessage({ content: "hi", id: "lc_run--original" }),
      };
      await cache.update("p", "l", [value]);
      const hit = await cache.lookup("p", "l");
      expect(hit).toHaveLength(1);
      expect((hit![0] as ChatGeneration).message.content).toBe("hi");
      expect((hit![0] as ChatGeneration).message.id).toBeUndefined();
      expect(value.message.id).toBe("lc_run--original");
    });

    if (kind === "exact") {
      test("does not match near-identical prompts", async () => {
        await cache.update("what is 2+2?", "l", [{ text: "4" }]);
        expect(await cache.lookup("what is 2+2?", "l")).toEqual([
          { text: "4" },
        ]);
        expect(await cache.lookup("what is 2 + 2?", "l")).toBeNull();
      });
    } else {
      test("uses the score threshold as a maximum distance", async () => {
        const bounded = new OracleSemanticCache(
          embeddings,
          { client: connection!, tableName, query: "test" },
          { scoreThreshold: 0 }
        );
        await bounded.initialize();
        await bounded.update("oracle database semantic cache", "l", [
          { text: "cached" },
        ]);
        expect(
          await bounded.lookup("oracle database semantic cache", "l")
        ).toEqual([{ text: "cached" }]);
        expect(
          await bounded.lookup("completely different question", "l")
        ).toBeNull();
      });

      test("returns null for metadata with a non-array return value", async () => {
        const prompt = "prompt one";
        const llmKey = "model-a";
        const vectorStore = new OracleVS(embeddings, {
          client: connection!,
          tableName,
          query: "test",
        });
        await vectorStore.addDocuments(
          [
            new Document({
              pageContent: prompt,
              metadata: {
                prompt_hash: defaultHashKeyEncoder(prompt),
                llm_key_hash: defaultHashKeyEncoder(llmKey),
                return_val: "not-an-array",
              },
            }),
          ],
          {
            ids: [
              defaultHashKeyEncoder(JSON.stringify([prompt, llmKey])).slice(
                0,
                36
              ),
            ],
          }
        );
        expect(await cache.lookup(prompt, llmKey)).toBeNull();
      });

      test("creates the requested vector index during initialization", async () => {
        const indexName = `IDX_${tableName}`;
        const indexed = new OracleSemanticCache(
          embeddings,
          { client: connection!, tableName, query: "test" },
          {
            createIndexIfMissing: true,
            indexName,
            indexParams: { parallel: 1 },
          }
        );
        await indexed.initialize();
        const result = await connection!.execute<{ INDEX_NAME: string }>(
          "SELECT index_name FROM user_indexes WHERE table_name = :table_name AND index_name = :index_name",
          { table_name: tableName, index_name: indexName },
          { outFormat: oracledb.OUT_FORMAT_OBJECT }
        );
        expect(result.rows).toEqual([{ INDEX_NAME: indexName }]);
      });
    }
  }
);
