import { BaseListChatMessageHistory } from "@langchain/core/chat_history";
import { dropTablePurge } from "./vectorstores.js";
import { createHash } from "node:crypto";
import oracledb from "oracledb";
import {
  ErrorCode,
  throwError,
  handleError,
  createErrorFromCodeWithCause,
} from "./errors.js";
import {
  BaseMessage,
  StoredMessage,
  mapStoredMessageToChatMessage,
} from "@langchain/core/messages";
import {
  withConnection,
  quoteIdentifier,
  OracleDBClient,
  OracleDBClientProvider,
} from "./utils.js";

const DEFAULT_TABLE_NAME = "langchain_message_store";
const DEFAULT_SESSION_ID_KEY = "session_id";
const DEFAULT_HISTORY_KEY = "message";
const DEFAULT_ID_KEY = quoteIdentifier("id");

interface OracleChatMessageHistoryBaseInput {
  client: OracleDBClient | OracleDBClientProvider;
  tableName?: string;
  sessionIdKey?: string;
  historyKey?: string;
}

interface OracleChatMessageHistoryInput
  extends OracleChatMessageHistoryBaseInput {
  sessionId: string;
  historySize?: number;
}

interface OracleChatMessageHistorySetupInput
  extends OracleChatMessageHistoryBaseInput {
  createIndex?: boolean;
}

function unqualifiedIdentifier(name: string): string {
  return name.replaceAll('"', "").split(".").pop()!;
}

function defaultIndexName(tableName: string, sessionIdKey: string): string {
  const baseName =
    `idx_${unqualifiedIdentifier(tableName)}_` +
    unqualifiedIdentifier(sessionIdKey);

  const digest = createHash("sha1")
    .update(JSON.stringify([tableName, sessionIdKey]))
    .digest("hex")
    .slice(0, 8);

  const prefix = new TextDecoder("utf-8").decode(
    Buffer.from(baseName, "utf-8").subarray(0, 119),
    { stream: true }
  );

  return `${prefix}_${digest}`;
}

/**
 * Stores conversation messages in Oracle Database, with one row per message
 * and a session ID identifying each conversation. Extends BaseListChatMessageHistory.
 *
 * Call `createTables()` before using the history unless the table already exists.
 * A single table can hold many sessions separated by the configured session ID column.
 * @example
 * ```typescript
 * // `pool` is an Oracle connection pool.
 * await OracleChatMessageHistory.createTables({
 *    client: pool,
 *    tableName: "langchain_chat_histories"
 * });
 *
 * const chatHistory = new OracleChatMessageHistory({
 *   client: pool,
 *   tableName: "langchain_chat_histories",
 *   sessionId: "lc-example",
 *   historySize: 20,
 * });
 *
 * ```
 */
export class OracleChatMessageHistory extends BaseListChatMessageHistory {
  lc_namespace = ["langchain", "stores", "message", "oracle"];
  readonly client: OracleDBClient | OracleDBClientProvider;
  readonly sessionId: string;
  private readonly quotedTableName: string;
  private readonly quotedSessionIdKey: string;
  private readonly quotedHistoryKey: string;
  readonly historySize?: number | undefined;

  /**
   * Creates an OracleChatMessageHistory instance without performing database setup.
   * @param {OracleChatMessageHistoryInput} options The history configuration.
   * @param {OracleDBClient | OracleDBClientProvider} options.client The connection, pool, or async provider to use.
   * @param {string} options.sessionId The session ID used to store and retrieve messages.
   * @param {string} options.tableName Table used to store chat history rows. Defaults to `langchain_message_store`.
   * @param {string} options.sessionIdKey Column containing the session identifier. Defaults to `session_id`.
   * @param {string} options.historyKey Column containing the serialized message payload. Defaults to `message`.
   * @param {number} options.historySize Optional maximum number of most recent messages to return.
   * @throws If the client is missing, the session ID is empty or not a string,
   * an identifier is invalid, or the history size is not a positive safe integer.
   */
  constructor({
    client,
    sessionId,
    tableName = DEFAULT_TABLE_NAME,
    sessionIdKey = DEFAULT_SESSION_ID_KEY,
    historyKey = DEFAULT_HISTORY_KEY,
    historySize,
  }: OracleChatMessageHistoryInput) {
    super();

    if (typeof sessionId !== "string" || !sessionId) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "sessionId must be a non-empty string"
      );
    }

    if (client == null) {
      throwError(ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER, "client");
    }

    if (
      historySize !== undefined &&
      (!Number.isSafeInteger(historySize) || historySize < 1)
    ) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "historySize must be greater than 0"
      );
    }

    this.client = client;
    this.sessionId = sessionId;
    this.historySize = historySize;

    this.quotedTableName = quoteIdentifier(tableName);
    this.quotedSessionIdKey = quoteIdentifier(sessionIdKey);
    this.quotedHistoryKey = quoteIdentifier(historyKey);
  }

  /**
   * Creates the history table and optional session index if they do not exist.
   * @param {OracleChatMessageHistorySetupInput} options The database setup configuration.
   * @param {boolean} options.createIndex Whether to create the session index. Defaults to `true`.
   * @returns Promise that resolves when database setup completes.
   */
  public static async createTables({
    client,
    tableName = DEFAULT_TABLE_NAME,
    createIndex = true,
    sessionIdKey = DEFAULT_SESSION_ID_KEY,
    historyKey = DEFAULT_HISTORY_KEY,
  }: OracleChatMessageHistorySetupInput): Promise<void> {
    try {
      const quotedTableName = quoteIdentifier(tableName);
      const quotedSessionIdKey = quoteIdentifier(sessionIdKey);
      const quotedHistoryKey = quoteIdentifier(historyKey);

      const createTableSql = `
      CREATE TABLE IF NOT EXISTS ${quotedTableName} (
        ${DEFAULT_ID_KEY} NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        ${quotedSessionIdKey} VARCHAR2(255) NOT NULL,
        ${quotedHistoryKey} CLOB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
      )`;

      const indexName = defaultIndexName(tableName, sessionIdKey);

      const createIndexSql = `
        CREATE INDEX IF NOT EXISTS ${quoteIdentifier(indexName)}
        ON ${quotedTableName} (${quotedSessionIdKey})
      `;

      await withConnection(client, async (connection) => {
        await connection.execute(createTableSql);

        if (createIndex) {
          await connection.execute(createIndexSql);
        }
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Drops and purges the history table if it exists, removing every session's messages.
   * @param {OracleDBClient | OracleDBClientProvider} client The connection, pool, or async provider to use.
   * @param {string} tableName The table to drop. Defaults to `langchain_message_store`.
   * @returns Promise that resolves when the table has been dropped or does not exist.
   */
  static async dropTable(
    client: OracleDBClient | OracleDBClientProvider,
    tableName = DEFAULT_TABLE_NAME
  ): Promise<void> {
    try {
      await withConnection(client, async (connection) => {
        await dropTablePurge(connection, tableName);
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Adds one message to the session.
   * @param {BaseMessage} message The message to store.
   * @returns Promise that resolves when the message has been stored.
   */
  public async addMessage(message: BaseMessage): Promise<void> {
    try {
      const binds = [this.sessionId, JSON.stringify(message.toDict())];

      await withConnection(this.client, async (connection) => {
        await connection.execute(this.insertQuery, binds);
        await connection.commit();
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Adds a batch of messages in insertion order.
   * @param {BaseMessage[]} messages The messages to store.
   * @returns Promise that resolves when the messages have been stored.
   */
  public async addMessages(messages: BaseMessage[]): Promise<void> {
    try {
      if (messages.length === 0) return;

      const rows = messages.map((message) => [
        this.sessionId,
        JSON.stringify(message.toDict()),
      ]);

      await withConnection(this.client, async (connection) => {
        await connection.executeMany(this.insertQuery, rows, {
          autoCommit: false,
        });
        await connection.commit();
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Retrieves messages for the configured session in ascending ID order.
   * When `historySize` is set, returns only the most recent messages in that order.
   * @returns Promise resolving to the stored messages, or an empty array for an empty session.
   */
  public async getMessages(): Promise<BaseMessage[]> {
    try {
      let query: string;
      let params: Record<string, string | number>;
      if (this.historySize === undefined) {
        query = `
        SELECT ${this.quotedHistoryKey} AS "payload", ${DEFAULT_ID_KEY}
        FROM ${this.quotedTableName}
        WHERE ${this.quotedSessionIdKey} = :session_id
        ORDER BY ${DEFAULT_ID_KEY}
      `;

        params = { session_id: this.sessionId };
      } else {
        query = `
        SELECT payload AS "payload", ${DEFAULT_ID_KEY}
        FROM (
          SELECT ${this.quotedHistoryKey} AS payload, ${DEFAULT_ID_KEY}
          FROM ${this.quotedTableName}
          WHERE ${this.quotedSessionIdKey} = :session_id
          ORDER BY ${DEFAULT_ID_KEY} DESC
        )
        WHERE ROWNUM <= :history_size
        ORDER BY ${DEFAULT_ID_KEY}
      `;

        params = {
          session_id: this.sessionId,
          history_size: this.historySize,
        };
      }

      const result = await withConnection(this.client, (connection) =>
        connection.execute<{ payload: string }>(query, params, {
          outFormat: oracledb.OUT_FORMAT_OBJECT,
          fetchInfo: {
            payload: { type: oracledb.STRING },
          },
        })
      );

      return (result.rows ?? []).map((row) => {
        try {
          return mapStoredMessageToChatMessage(
            JSON.parse(row.payload) as StoredMessage
          );
        } catch (error) {
          throw createErrorFromCodeWithCause(
            ErrorCode.HISTORY_DESERIALIZATION_FAILED,
            error
          );
        }
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Deletes all messages belonging to the configured session.
   * @returns Promise that resolves when this session's messages have been cleared.
   */
  public async clear(): Promise<void> {
    try {
      await withConnection(this.client, async (connection) => {
        await connection.execute(
          this.deleteQuery,
          { session_id: this.sessionId },
          { autoCommit: true }
        );
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Replaces the configured session's stored messages. An empty array clears the session.
   * @param {BaseMessage[]} messages The replacement messages in insertion order.
   * @returns Promise that resolves when the replacement has been committed.
   */
  public async replaceMessages(messages: BaseMessage[]): Promise<void> {
    try {
      const rows = messages.map((message) => [
        this.sessionId,
        JSON.stringify(message.toDict()),
      ]);

      await withConnection(this.client, async (connection) => {
        try {
          await connection.execute(
            this.deleteQuery,
            { session_id: this.sessionId },
            { autoCommit: false }
          );

          if (rows.length > 0) {
            await connection.executeMany(this.insertQuery, rows, {
              autoCommit: false,
              batchErrors: false,
            });
          }

          await connection.commit();
        } catch (error) {
          let cause = error;
          try {
            await connection.rollback();
          } catch (rollbackError) {
            cause = new AggregateError(
              [error, rollbackError],
              "Replacing messages failed, and rollback also failed."
            );
          }
          throw createErrorFromCodeWithCause(
            ErrorCode.HISTORY_REPLACEMENT_FAILED,
            cause
          );
        }
      });
    } catch (error) {
      handleError(error);
    }
  }

  private get insertQuery(): string {
    return `
      INSERT INTO ${this.quotedTableName}
        (
          ${this.quotedSessionIdKey},
          ${this.quotedHistoryKey}
        )
      VALUES (:1, :2)
    `;
  }

  private get deleteQuery(): string {
    return `
      DELETE FROM ${this.quotedTableName}
      WHERE ${this.quotedSessionIdKey} = :session_id
    `;
  }
}
