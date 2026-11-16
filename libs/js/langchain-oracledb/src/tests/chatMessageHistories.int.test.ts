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
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import oracledb from "oracledb";
import { OracleChatMessageHistory } from "../chatMessageHistories.js";
import { quoteIdentifier } from "../utils.js";

const hasCredentials = Boolean(
  env.ORACLE_USERNAME && env.ORACLE_PASSWORD && env.ORACLE_DSN
);

describe.skipIf(!hasCredentials)("OracleChatMessageHistory integration", () => {
  let pool: oracledb.Pool | undefined;
  let connection: oracledb.Connection | undefined;
  let tableName: string;
  const sessionId = "primary-session";
  const otherSessionId = "other-session";

  beforeAll(async () => {
    pool = await oracledb.createPool({
      user: env.ORACLE_USERNAME,
      password: env.ORACLE_PASSWORD,
      connectString: env.ORACLE_DSN,
    });
  });

  beforeEach(async () => {
    tableName = `CHAT_HISTORY_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
    connection = await pool!.getConnection();
  });

  afterEach(async () => {
    const current = connection;
    connection = undefined;
    if (current) {
      try {
        await OracleChatMessageHistory.dropTable(current, tableName);
      } finally {
        await current.close();
      }
    }
  });

  afterAll(async () => {
    await pool?.close();
  });

  async function createHistory(historySize?: number) {
    await OracleChatMessageHistory.createTables({
      client: connection!,
      tableName,
    });
    return new OracleChatMessageHistory({
      client: connection!,
      tableName,
      sessionId,
      historySize,
    });
  }

  async function sessionIndexes(column = "session_id") {
    const result = await connection!.execute<{ INDEX_NAME: string }>(
      `SELECT index_name FROM user_ind_columns
       WHERE table_name = :table_name AND column_name = :column_name`,
      { table_name: tableName, column_name: column },
      { outFormat: oracledb.OUT_FORMAT_OBJECT }
    );
    return result.rows ?? [];
  }

  test("creates, adds, retrieves, and clears history", async () => {
    const history = await createHistory();
    expect(await history.getMessages()).toEqual([]);
    const messages = [
      new HumanMessage("hello"),
      new AIMessage("world"),
      new HumanMessage("goodbye"),
    ];
    await history.addMessages(messages.slice(0, 2));
    await history.addMessage(messages[2]);
    expect(await history.getMessages()).toEqual(messages);

    await history.clear();
    expect(await history.getMessages()).toEqual([]);
  });

  test("isolates sessions and returns the latest messages in chronological order", async () => {
    const primary = await createHistory(2);
    const secondary = new OracleChatMessageHistory({
      client: pool!,
      tableName,
      sessionId: otherSessionId,
    });
    const messages = [
      new HumanMessage("one"),
      new AIMessage("two"),
      new HumanMessage("three"),
    ];
    const otherMessages = [new HumanMessage("isolated")];
    await primary.addMessages(messages);
    await secondary.addMessages(otherMessages);
    expect(await primary.getMessages()).toEqual(messages.slice(-2));

    const unbounded = new OracleChatMessageHistory({
      client: connection!,
      tableName,
      sessionId,
    });
    expect(await unbounded.getMessages()).toEqual(messages);
    expect(await secondary.getMessages()).toEqual(otherMessages);
    await primary.clear();
    expect(await unbounded.getMessages()).toEqual([]);
    expect(await secondary.getMessages()).toEqual(otherMessages);
  });

  test("replaces only the requested session, including an empty replacement", async () => {
    const history = await createHistory();
    const secondary = new OracleChatMessageHistory({
      client: connection!,
      tableName,
      sessionId: otherSessionId,
    });
    const otherMessages = [new HumanMessage("keep me")];
    await secondary.addMessages(otherMessages);
    await history.addMessages([
      new HumanMessage("before"),
      new AIMessage("before-response"),
    ]);
    const replacement = [new SystemMessage("replacement")];
    await history.replaceMessages(replacement);
    expect(await history.getMessages()).toEqual(replacement);
    expect(await secondary.getMessages()).toEqual(otherMessages);

    await history.replaceMessages([]);
    expect(await history.getMessages()).toEqual([]);
    expect(await secondary.getMessages()).toEqual(otherMessages);
  });

  test("empty batches leave both empty and populated history unchanged", async () => {
    const history = await createHistory();
    await history.addMessages([]);
    expect(await history.getMessages()).toEqual([]);
    const messages = [new HumanMessage("keep me")];
    await history.addMessages(messages);
    await history.addMessages([]);
    expect(await history.getMessages()).toEqual(messages);
  });

  test("supports optional index creation and repeated setup without losing data", async () => {
    await OracleChatMessageHistory.createTables({
      client: connection!,
      tableName,
      createIndex: false,
    });
    expect(await sessionIndexes()).toEqual([]);
    const history = new OracleChatMessageHistory({
      client: connection!,
      tableName,
      sessionId,
    });
    const messages = [new HumanMessage("keep during setup")];
    await history.addMessages(messages);

    await OracleChatMessageHistory.createTables({
      client: connection!,
      tableName,
    });
    const indexes = await sessionIndexes();
    expect(indexes).toHaveLength(1);
    await OracleChatMessageHistory.createTables({
      client: connection!,
      tableName,
    });
    expect(await sessionIndexes()).toEqual(indexes);
    expect(await history.getMessages()).toEqual(messages);
  });

  test("reads custom columns and creates an index on the custom session column", async () => {
    const sessionIdKey = "conversation_id";
    const historyKey = "payload";
    await OracleChatMessageHistory.createTables({
      client: connection!,
      tableName,
      sessionIdKey,
      historyKey,
    });
    const message = new HumanMessage("custom-columns");
    await connection!.execute(
      `INSERT INTO ${quoteIdentifier(tableName)}
       ("conversation_id", "payload") VALUES (:1, :2)`,
      [sessionId, JSON.stringify(message.toDict())],
      { autoCommit: true }
    );
    const history = new OracleChatMessageHistory({
      client: connection!,
      tableName,
      sessionId,
      sessionIdKey,
      historyKey,
    });
    expect(await history.getMessages()).toEqual([message]);
    const indexes = await sessionIndexes(sessionIdKey);
    expect(indexes).toHaveLength(1);
    expect(indexes[0].INDEX_NAME).toMatch(/^idx_/);
    expect(
      Buffer.byteLength(indexes[0].INDEX_NAME, "utf8")
    ).toBeLessThanOrEqual(128);
    await history.addMessage(new AIMessage("custom-response"));
    expect(await history.getMessages()).toEqual([
      message,
      new AIMessage("custom-response"),
    ]);
  });
});
