import { describe, expect, test, vi } from "vitest";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import oracledb from "oracledb";
import { OracleChatMessageHistory } from "../chatMessageHistories.js";
import { ErrorCode } from "../errors.js";

function mockConnection() {
  const execute = vi.fn().mockResolvedValue({ rows: [] });
  const executeMany = vi.fn().mockResolvedValue({});
  const commit = vi.fn().mockResolvedValue(undefined);
  const rollback = vi.fn().mockResolvedValue(undefined);
  const close = vi.fn().mockResolvedValue(undefined);
  const client = {
    execute,
    executeMany,
    commit,
    rollback,
    close,
  } as unknown as oracledb.Connection;
  return { client, execute, executeMany, commit, rollback, close };
}

async function generatedIndexName(tableName: string, sessionIdKey: string) {
  const { client, execute } = mockConnection();
  await OracleChatMessageHistory.createTables({
    client,
    tableName,
    sessionIdKey,
  });
  const ddl = execute.mock.calls[1][0] as string;
  const match = /CREATE INDEX IF NOT EXISTS "([^"]+)"/.exec(ddl);
  expect(match).not.toBeNull();
  return match![1];
}

describe("OracleChatMessageHistory", () => {
  test("validates constructor arguments", () => {
    const { client } = mockConnection();
    expect(
      () =>
        new OracleChatMessageHistory({
          client,
          sessionId: 123 as unknown as string,
        })
    ).toThrow(/sessionId must be a non-empty string/);
    expect(
      () =>
        new OracleChatMessageHistory({
          client: null as unknown as oracledb.Connection,
          sessionId: "session-1",
        })
    ).toThrow(/client.*required/);
    expect(
      () =>
        new OracleChatMessageHistory({
          client,
          sessionId: "session-1",
          historySize: 0,
        })
    ).toThrow(/historySize must be greater than 0/);
  });

  test("chat history default index name is truncated when needed", async () => {
    const tableName = `table_${"x".repeat(80)}`;
    const sessionIdKey = `session_${"y".repeat(80)}`;

    const name = await generatedIndexName(tableName, sessionIdKey);

    expect(name).toMatch(/^idx_/);
    expect(Buffer.byteLength(name, "utf8")).toBeLessThanOrEqual(128);
    expect(await generatedIndexName(tableName, sessionIdKey)).toBe(name);
  });

  test("distinguishes table and column combinations with identical prefixes", async () => {
    const first = await generatedIndexName("support_chat", "session_id");
    const second = await generatedIndexName("support", "chat_session_id");
    expect(first).not.toBe(second);
  });

  test.each([
    { not: "a string" },
    "invalid JSON",
    JSON.stringify({ type: "unsupported", data: {} }),
  ])("rejects an invalid stored payload: %j", async (payload) => {
    const { client, execute } = mockConnection();
    execute.mockResolvedValue({ rows: [{ payload }] });
    const history = new OracleChatMessageHistory({ client, sessionId: "s1" });
    await expect(history.getMessages()).rejects.toMatchObject({
      code: ErrorCode.HISTORY_DESERIALIZATION_FAILED,
      cause: expect.any(Error),
    });
  });

  test("restores message types and requests CLOB payloads as strings", async () => {
    const { client, execute } = mockConnection();
    const messages = [
      new SystemMessage("instructions"),
      new HumanMessage("hello"),
      new AIMessage("world"),
    ];
    execute.mockResolvedValue({
      rows: messages.map((message) => ({
        payload: JSON.stringify(message.toDict()),
      })),
    });
    const history = new OracleChatMessageHistory({ client, sessionId: "s1" });
    expect(await history.getMessages()).toEqual(messages);
    expect(execute).toHaveBeenCalledWith(
      expect.any(String),
      { session_id: "s1" },
      expect.objectContaining({
        outFormat: oracledb.OUT_FORMAT_OBJECT,
        fetchInfo: { payload: { type: oracledb.STRING } },
      })
    );
  });

  test("rolls back replacement when insertion fails", async () => {
    const { client, execute, executeMany, rollback } = mockConnection();
    const failure = new Error("insert failed");
    executeMany.mockRejectedValue(failure);
    const history = new OracleChatMessageHistory({ client, sessionId: "s1" });

    await expect(
      history.replaceMessages([new HumanMessage("replacement")])
    ).rejects.toMatchObject({
      code: ErrorCode.HISTORY_REPLACEMENT_FAILED,
      cause: failure,
    });

    expect(execute).toHaveBeenCalledWith(
      expect.stringContaining("DELETE FROM"),
      { session_id: "s1" },
      { autoCommit: false }
    );
    expect(executeMany).toHaveBeenCalledOnce();
    expect(rollback).toHaveBeenCalledOnce();
    expect(execute.mock.invocationCallOrder[0]).toBeLessThan(
      executeMany.mock.invocationCallOrder[0]
    );
    expect(executeMany.mock.invocationCallOrder[0]).toBeLessThan(
      rollback.mock.invocationCallOrder[0]
    );
  });

  test("keeps database read failures separate from decoding failures", async () => {
    const { client, execute } = mockConnection();
    const failure = new Error("database unavailable");
    execute.mockRejectedValue(failure);
    const history = new OracleChatMessageHistory({ client, sessionId: "s1" });
    await expect(history.getMessages()).rejects.toMatchObject({
      code: ErrorCode.SYSTEM_ERROR,
      cause: failure,
    });
  });

  test("preserves both replacement and rollback failures", async () => {
    const { client, executeMany, rollback } = mockConnection();
    const failure = new Error("insert failed");
    const rollbackFailure = new Error("rollback failed");
    executeMany.mockRejectedValue(failure);
    rollback.mockRejectedValue(rollbackFailure);
    const history = new OracleChatMessageHistory({ client, sessionId: "s1" });
    const operation = history.replaceMessages([
      new HumanMessage("replacement"),
    ]);
    await expect(operation).rejects.toMatchObject({
      code: ErrorCode.HISTORY_REPLACEMENT_FAILED,
      cause: expect.any(AggregateError),
    });
    // Check both original failures survive in the aggregate cause.
    await expect(operation).rejects.toMatchObject({
      cause: { errors: [failure, rollbackFailure] },
    });
  });

  test("empty batches do not resolve the client provider", async () => {
    const { client } = mockConnection();
    const provider = vi.fn().mockResolvedValue(client);
    const history = new OracleChatMessageHistory({
      client: provider,
      sessionId: "s1",
    });
    await history.addMessages([]);
    expect(provider).not.toHaveBeenCalled();
  });

  test.each([false, true])(
    "returns a borrowed connection after replacement (failure: %s)",
    async (fails) => {
      const { client, executeMany, close, commit, rollback } = mockConnection();
      const getConnection = vi.fn().mockResolvedValue(client);
      const pool = { getConnection } as unknown as oracledb.Pool;
      const history = new OracleChatMessageHistory({
        client: pool,
        sessionId: "s1",
      });
      if (fails) executeMany.mockRejectedValue(new Error("insert failed"));

      const operation = history.replaceMessages([new HumanMessage("new")]);
      if (fails) {
        await expect(operation).rejects.toMatchObject({
          code: ErrorCode.HISTORY_REPLACEMENT_FAILED,
          cause: expect.objectContaining({ message: "insert failed" }),
        });
        expect(commit).not.toHaveBeenCalled();
        expect(rollback).toHaveBeenCalledOnce();
      } else {
        await operation;
        expect(commit).toHaveBeenCalledOnce();
        expect(rollback).not.toHaveBeenCalled();
      }
      expect(getConnection).toHaveBeenCalledOnce();
      expect(close).toHaveBeenCalledOnce();
      const endTransaction = fails ? rollback : commit;
      // Commit or roll back before returning the connection to the pool.
      expect(endTransaction.mock.invocationCallOrder[0]).toBeLessThan(
        close.mock.invocationCallOrder[0]
      );
    }
  );
});
