import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { describe, expect, test, vi } from "vitest";
import type oracledb from "oracledb";

import {
  generateWhereClause,
  OracleVS,
  type OracleDBVSArgs,
} from "../vectorstores.js";

const embeddings = {
  embedDocuments: async (texts: string[]) => texts.map(() => [0.1, 0.2]),
  embedQuery: async () => [0.1, 0.2],
} as EmbeddingsInterface;

const createConfig = (
  client: OracleDBVSArgs["client"],
): OracleDBVSArgs => ({
  client,
  tableName: "test_vectors",
  query: "test query",
});

describe("OracleVS client provider", () => {
  test("returns pool-borrowed connections after internal operations", async () => {
    const connection = {
      execute: vi.fn(async () => ({})),
      close: vi.fn(async () => {}),
    } as unknown as oracledb.Connection;
    const pool = {
      close: vi.fn(async () => {}),
      getConnection: vi.fn(async () => connection),
    } as unknown as oracledb.Pool;
    const getClient = vi.fn(async () => pool);
    const vectorStore = new OracleVS(embeddings, createConfig(getClient));

    await vectorStore.initialize();

    expect(getClient).toHaveBeenCalledTimes(1);
    expect(pool.getConnection).toHaveBeenCalledTimes(1);
    expect(connection.execute).toHaveBeenCalledTimes(1);
    expect(connection.close).toHaveBeenCalledTimes(1);
  });

  test("does not close provider-owned pool from end", async () => {
    const pool = {
      close: vi.fn(async () => {}),
      getConnection: vi.fn(),
    } as unknown as oracledb.Pool;
    const getClient = vi.fn(async () => pool);
    const vectorStore = new OracleVS(embeddings, createConfig(getClient));

    await vectorStore.end();

    expect(getClient).not.toHaveBeenCalled();
    expect(pool.close).not.toHaveBeenCalled();
  });

  test("returns provider pool connections from public getConnection and retConnection", async () => {
    const connection = {
      close: vi.fn(async () => {}),
    } as unknown as oracledb.Connection;
    const pool = {
      close: vi.fn(async () => {}),
      getConnection: vi.fn(async () => connection),
    } as unknown as oracledb.Pool;
    const getClient = vi.fn(async () => pool);
    const vectorStore = new OracleVS(embeddings, createConfig(getClient));

    const resolvedConnection = await vectorStore.getConnection();
    await vectorStore.retConnection(resolvedConnection);

    expect(getClient).toHaveBeenCalledTimes(1);
    expect(pool.getConnection).toHaveBeenCalledTimes(1);
    expect(connection.close).toHaveBeenCalledTimes(1);
  });

  test("does not close direct connections returned by provider after internal operations", async () => {
    const connection = {
      execute: vi.fn(async () => ({})),
      close: vi.fn(async () => {}),
    } as unknown as oracledb.Connection;
    const getClient = vi.fn(async () => connection);
    const vectorStore = new OracleVS(embeddings, createConfig(getClient));

    await vectorStore.initialize();

    expect(getClient).toHaveBeenCalledTimes(1);
    expect(connection.execute).toHaveBeenCalledTimes(1);
    expect(connection.close).not.toHaveBeenCalled();
  });

  test("does not close concrete direct connections after internal operations", async () => {
    const connection = {
      execute: vi.fn(async () => ({})),
      close: vi.fn(async () => {}),
    } as unknown as oracledb.Connection;
    const vectorStore = new OracleVS(embeddings, createConfig(connection));

    await vectorStore.initialize();

    expect(connection.execute).toHaveBeenCalledTimes(1);
    expect(connection.close).not.toHaveBeenCalled();
  });
});

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

  test("rejects metadata keys containing SQL injection payloads", () => {
    expect(() =>
      generateWhereClause({ ["author') OR 1=1 --"]: "alice" }, [])
    ).toThrow(/Invalid metadata key/);
  });
});
