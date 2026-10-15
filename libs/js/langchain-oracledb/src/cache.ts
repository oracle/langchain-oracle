import oracledb from "oracledb";
import type { Generation } from "@langchain/core/outputs";
import { BaseMessage, AIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
  ErrorCode,
  throwError,
  handleError,
  createErrorFromCodeWithCause,
} from "./errors.js";
import {
  BaseCache,
  serializeGeneration,
  deserializeStoredGeneration,
} from "@langchain/core/caches";
import {
  OracleVS,
  createIndex,
  OracleDBVSArgs,
  IndexParams,
  dropTablePurge,
} from "./vectorstores.js";
import {
  withConnection,
  quoteIdentifier,
  OracleDBClient,
  OracleDBClientProvider,
} from "./utils.js";

const DEFAULT_TABLE_NAME = "langchain_semantic_cache";
const DISTANCE_EPSILON = 1e-12;

export type OracleSemanticCacheDBConfig = Omit<OracleDBVSArgs, "tableName"> & {
  tableName?: string;
};

export interface OracleSemanticCacheOptions {
  createIndexIfMissing?: boolean;
  indexName?: string;
  indexParams?: IndexParams;
  scoreThreshold?: number;
}

export interface OracleCacheClearOptions {
  prompt?: string;
  llmKey?: string;
}

/** Deserializes cached generations and clears message IDs before reuse. */
function deserializeGenerations(payload: unknown): Generation[] | null {
  try {
    const stored = typeof payload === "string" ? JSON.parse(payload) : payload;
    if (!Array.isArray(stored)) {
      console.warn("Ignoring malformed cached generations: expected an array.");
      return null;
    }

    const generations: Generation[] = [];

    for (const entry of stored) {
      if (
        entry === null ||
        typeof entry !== "object" ||
        typeof entry.text !== "string"
      ) {
        console.warn(
          "Ignoring malformed cached generations: expected objects with string text."
        );
        return null;
      }

      const generation = deserializeStoredGeneration(entry);

      // Clears cached message IDs before reuse so LangGraph's message reducer does not
      // replace an earlier message with the same ID instead of appending a new one.
      // Messages whose IDs cannot be assigned are left unchanged on TypeError.
      if (
        "message" in generation &&
        BaseMessage.isInstance(generation.message)
      ) {
        try {
          generation.message.id = undefined;
        } catch (error) {
          // eslint-disable-next-line no-instanceof/no-instanceof
          if (!(error instanceof TypeError)) {
            throw error;
          }
        }
      }

      generations.push(generation);
    }

    return generations;
  } catch (error) {
    // eslint-disable-next-line no-instanceof/no-instanceof
    if (error instanceof SyntaxError || error instanceof TypeError) {
      console.warn(
        `Ignoring malformed cached generations: ${error.name} during decoding.`
      );
      return null;
    }

    throw createErrorFromCodeWithCause(
      ErrorCode.CACHE_DESERIALIZATION_FAILED,
      error
    );
  }
}

/** Detects tool-call responses, which represent agent steps rather than final answers. */
function hasToolCalls(generations: Generation[]): boolean {
  return generations.some(
    (generation) =>
      "message" in generation &&
      AIMessage.isInstance(generation.message) &&
      ((generation.message.tool_calls?.length ?? 0) > 0 ||
        (generation.message.invalid_tool_calls?.length ?? 0) > 0)
  );
}

function validateClearOptions(options: OracleCacheClearOptions) {
  const unknownKeys = Object.keys(options).filter(
    (key) => key !== "prompt" && key !== "llmKey"
  );

  if (unknownKeys.length > 0) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      `Unsupported clear options: ${unknownKeys.join(", ")}`
    );
  }
}

/**
 * Caches model generations in Oracle Database using vector similarity between prompts.
 * @example
 * ```typescript
 * // `embeddings` is an EmbeddingsInterface and `pool` is an Oracle connection pool.
 * const cache = new OracleSemanticCache(embeddings, { client: pool }, {
 *   scoreThreshold: 0.2,
 *   createIndexIfMissing: true,
 * });
 * ```
 */
export class OracleSemanticCache extends BaseCache<Generation[]> {
  private readonly vectorStore: OracleVS;

  readonly createIndexIfMissing?: boolean;
  readonly indexName?: string;
  readonly indexParams?: IndexParams;
  readonly scoreThreshold?: number;

  private readonly LLM_HASH = "llm_key_hash";
  private readonly PROMPT_HASH = "prompt_hash";
  private readonly RETURN_VAL = "return_val";

  /**
   * Creates a semantic cache instance without performing database setup.
   *
   * @param {EmbeddingsInterface} embeddings Embedding model used to vectorize prompts.
   * @param {OracleSemanticCacheDBConfig} dbConfig Oracle vector-store configuration.
   * @param {OracleDBClient | OracleDBClientProvider} dbConfig.client Connection, pool, or async provider to use.
   * @param {string} dbConfig.tableName Cache table name. Defaults to `langchain_semantic_cache`.
   *
   * @param {OracleSemanticCacheOptions} options Optional index and distance threshold settings.
   * @param {boolean} options.createIndexIfMissing Whether initialize() creates a vector index
   * if it does not exist. Defaults to `false`.
   * @param {number} options.scoreThreshold Maximum accepted vector distance.
   *
   * @throws If the client is missing, the table identifier is invalid,
   * or the distance threshold is negative.
   */

  constructor(
    embeddings: EmbeddingsInterface,
    dbConfig: OracleSemanticCacheDBConfig,
    options: OracleSemanticCacheOptions = {}
  ) {
    super();

    if (options.scoreThreshold !== undefined && options.scoreThreshold < 0) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "scoreThreshold must be non-negative"
      );
    }

    if (dbConfig.client == null) {
      throwError(ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER, "client");
    }

    this.vectorStore = new OracleVS(embeddings, {
      ...dbConfig,
      tableName: dbConfig.tableName ?? DEFAULT_TABLE_NAME,
    });

    this.createIndexIfMissing = options.createIndexIfMissing ?? false;
    this.indexName = options.indexName;
    this.indexParams = options.indexParams;
    this.scoreThreshold = options.scoreThreshold;
  }

  /**
   * Initializes the underlying vector store and creates the optional vector index.
   * @returns Promise that resolves when database setup completes.
   */
  async initialize(): Promise<void> {
    try {
      await this.vectorStore.initialize();

      if (this.createIndexIfMissing) {
        const params: IndexParams = {
          ...(this.indexParams ?? {}),
        };

        if (this.indexName !== undefined) {
          params.idxName = this.indexName;
        }

        const connection = await this.vectorStore.getConnection();
        try {
          await createIndex(connection, this.vectorStore, params);
        } finally {
          await this.vectorStore.retConnection(connection);
        }
      }
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Retrieves the nearest cached response for the given model configuration key.
   * @param {string} prompt Prompt to embed and search for.
   * @param {string} llmKey Key identifying the model and its configuration.
   * @returns Cached generations, or `null` when no match passes the threshold or the payload is malformed.
   * @throws If search fails or deserialization encounters an unexpected error.
   */
  async lookup(prompt: string, llmKey: string): Promise<Generation[] | null> {
    try {
      const filter = {
        [this.LLM_HASH]: {
          $eq: this.keyEncoder(llmKey),
        },
      };

      const searchResponse = await this.vectorStore.similaritySearchWithScore(
        prompt,
        1,
        filter
      );

      if (searchResponse.length === 0) {
        return null;
      }

      const [document, score] = searchResponse[0];

      if (
        this.scoreThreshold != undefined &&
        score > this.scoreThreshold + DISTANCE_EPSILON
      ) {
        return null;
      }

      return deserializeGenerations(document.metadata[this.RETURN_VAL]);
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Inserts or updates the entry for an exact prompt and model configuration key.
   * Skips the entire update if any generation contains valid or invalid tool calls,
   * avoiding reuse of procedural responses within an agent loop.
   * @param {string} prompt Prompt to embed and store.
   * @param {string} llmKey Key identifying the model and its configuration.
   * @param {Generation[]} value Model generations to cache.
   * @returns Promise that resolves when the write completes or the update is skipped.
   */
  async update(
    prompt: string,
    llmKey: string,
    value: Generation[]
  ): Promise<void> {
    try {
      if (hasToolCalls(value)) {
        return;
      }

      const metadata = {
        [this.PROMPT_HASH]: this.keyEncoder(prompt),
        [this.LLM_HASH]: this.keyEncoder(llmKey),
        [this.RETURN_VAL]: value.map(serializeGeneration),
      };

      await this.vectorStore.addDocuments(
        [
          new Document({
            pageContent: prompt,
            metadata,
          }),
        ],
        {
          ids: [this.keyEncoder(JSON.stringify([prompt, llmKey])).slice(0, 36)],
          mutateOnDuplicate: true,
        }
      );
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Deletes entries matching all supplied exact filters, without similarity search.
   * With no filters, deletes every entry in the table.
   * @param {OracleCacheClearOptions} options Optional prompt and model configuration filters.
   * @param {string} options.prompt Optional prompt to match exactly.
   * @param {string} options.llmKey Optional model configuration key to match exactly.
   * @returns Promise that resolves when deletion has been committed.
   * @throws If an unsupported filter is supplied or deletion fails.
   */
  async clear(options: OracleCacheClearOptions = {}): Promise<void> {
    try {
      validateClearOptions(options);
      const { prompt, llmKey } = options;
      let query = `DELETE FROM ${this.vectorStore.tableName}`;
      const binds: Record<string, string> = {};
      const conditions: string[] = [];

      if (prompt !== undefined) {
        conditions.push(
          `JSON_VALUE(metadata, '$.${this.PROMPT_HASH}') = :promptHash`
        );
        binds.promptHash = this.keyEncoder(prompt);
      }

      if (llmKey !== undefined) {
        conditions.push(
          `JSON_VALUE(metadata, '$.${this.LLM_HASH}') = :llmHash`
        );
        binds.llmHash = this.keyEncoder(llmKey);
      }

      if (conditions.length > 0) {
        query += ` WHERE ${conditions.join(" AND ")}`;
      }

      await withConnection(this.vectorStore.client, async (connection) => {
        await connection.execute(query, binds);
        await connection.commit();
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Drops and purges the semantic cache table if it exists, removing all entries.
   * @param {OracleDBClient | OracleDBClientProvider} client Connection, pool, or async provider to use.
   * @param {string} tableName Table to drop. Defaults to `langchain_semantic_cache`.
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
}

const DEFAULT_EXACT_TABLE_NAME = "langchain_exact_cache";

/**
 * Caches model generations by exact prompt and model configuration key in Oracle Database.
 * Lookup uses a hashed primary key and does not require an embedding model.
 *
 * @example
 * ```typescript
 * // `pool` is an Oracle connection pool.
 * const cache = new OracleCache(pool);
 * await cache.initialize();
 * await cache.update("Hello", "model-config", [{ text: "Hi!" }]);
 * const generations = await cache.lookup("Hello", "model-config");
 * ```
 */
export class OracleCache extends BaseCache<Generation[]> {
  readonly client: OracleDBClient | OracleDBClientProvider;

  readonly tableName;
  private readonly quotedTableName: string;

  /**
   * Creates an exact cache instance without performing database setup.
   * @param {OracleDBClient | OracleDBClientProvider} client Connection, pool, or async provider to use.
   * @param {string} tableName Cache table name. Defaults to `langchain_exact_cache`.
   * @throws If the client is missing or the table identifier is invalid.
   */
  constructor(
    client: OracleDBClient | OracleDBClientProvider,
    tableName: string = DEFAULT_EXACT_TABLE_NAME
  ) {
    super();

    if (client == null) {
      throwError(ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER, "client");
    }

    this.client = client;
    this.tableName = tableName;
    this.quotedTableName = quoteIdentifier(tableName);
  }

  /**
   * Creates the cache table if it does not already exist.
   * @returns Promise that resolves when database setup completes.
   */
  async initialize(): Promise<void> {
    try {
      await this.ensureTable();
    } catch (error) {
      handleError(error);
    }
  }

  /** Stores serialized generations in a CLOB, with separate hashes for filtered deletion. */
  private async ensureTable(): Promise<void> {
    const ddl = `
      CREATE TABLE IF NOT EXISTS ${this.quotedTableName} (
        id VARCHAR2(64) PRIMARY KEY,
        prompt_hash VARCHAR2(64) NOT NULL,
        llm_key_hash VARCHAR2(64) NOT NULL,
        generations CLOB NOT NULL,
        created_at TIMESTAMP DEFAULT SYSTIMESTAMP
      )
    `;

    await withConnection(this.client, async (connection) => {
      await connection.execute(ddl);
      await connection.commit();
    });
  }

  /**
   * Retrieves the cached response for an exact prompt and model configuration key.
   * Restored chat message IDs are cleared before returning the generations.
   * @param {string} prompt Exact prompt used when writing the entry.
   * @param {string} llmKey Key identifying the model and its configuration.
   * @returns Cached generations, or `null` when the entry is absent or its payload is malformed.
   * @throws If the query fails or deserialization encounters an unexpected error.
   */
  async lookup(prompt: string, llmKey: string): Promise<Generation[] | null> {
    try {
      return await withConnection(this.client, async (connection) => {
        const result = await connection.execute<{ generations: string }>(
          `SELECT generations AS "generations"
          FROM ${this.quotedTableName}
          WHERE id = :id`,
          { id: this.keyEncoder(JSON.stringify([prompt, llmKey])) },
          {
            outFormat: oracledb.OUT_FORMAT_OBJECT,
            fetchInfo: {
              generations: { type: oracledb.STRING },
            },
          }
        );

        const row = result.rows?.[0];
        if (typeof row?.generations !== "string") {
          return null;
        }

        return deserializeGenerations(row.generations);
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Inserts or replaces generations for an exact prompt and model configuration key.
   * Skips the entire update if any generation contains valid or invalid tool calls.
   * Successful writes commit the connection's transaction.
   * @param {string} prompt Prompt used to identify the entry.
   * @param {string} llmKey Key identifying the model and its configuration.
   * @param {Generation[]} value Model generations to cache.
   * @returns Promise that resolves when the write is committed or the update is skipped.
   */
  async update(
    prompt: string,
    llmKey: string,
    value: Generation[]
  ): Promise<void> {
    try {
      if (hasToolCalls(value)) {
        return;
      }

      const merge: string = `
        MERGE INTO ${this.quotedTableName} t
        USING (SELECT :id AS id FROM dual) s
        ON (t.id = s.id)
        WHEN MATCHED THEN UPDATE SET
          t.generations = :generations,
          t.created_at = SYSTIMESTAMP
        WHEN NOT MATCHED THEN INSERT
          (id, prompt_hash, llm_key_hash, generations)
          VALUES (:id, :prompt_hash, :llm_key_hash, :generations)
      `;

      const binds: Record<string, string> = {
        id: this.keyEncoder(JSON.stringify([prompt, llmKey])),
        prompt_hash: this.keyEncoder(prompt),
        llm_key_hash: this.keyEncoder(llmKey),
        generations: JSON.stringify(value.map(serializeGeneration)),
      };

      await withConnection(this.client, async (connection) => {
        await connection.execute(merge, binds);
        await connection.commit();
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Deletes entries matching all supplied exact-match filters.
   * With no filters, deletes every entry in the table. Commits the connection's transaction.
   * @param {OracleCacheClearOptions} options Optional prompt and model configuration filters.
   * @param {string} options.prompt Optional prompt to match exactly.
   * @param {string} options.llmKey Optional model configuration key to match exactly.
   * @returns Promise that resolves when deletion has been committed.
   * @throws If an unsupported filter is supplied or deletion fails.
   */
  async clear(options: OracleCacheClearOptions = {}): Promise<void> {
    try {
      validateClearOptions(options);
      const { prompt, llmKey } = options;
      let query = `DELETE FROM ${this.quotedTableName}`;
      const binds: Record<string, string> = {};
      const conditions: string[] = [];

      if (prompt !== undefined) {
        conditions.push(`prompt_hash = :prompt_hash`);
        binds.prompt_hash = this.keyEncoder(prompt);
      }

      if (llmKey !== undefined) {
        conditions.push(`llm_key_hash = :llm_hash`);
        binds.llm_hash = this.keyEncoder(llmKey);
      }

      if (conditions.length > 0) {
        query += ` WHERE ${conditions.join(" AND ")}`;
      }

      await withConnection(this.client, async (connection) => {
        await connection.execute(query, binds);
        await connection.commit();
      });
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Drops and purges the exact cache table if it exists, removing all entries.
   * @param {OracleDBClient | OracleDBClientProvider} client Connection, pool, or async provider to use.
   * @param {string} tableName Table to drop. Defaults to `langchain_exact_cache`.
   * @returns Promise that resolves when the table has been dropped or does not exist.
   */
  static async dropTable(
    client: OracleDBClient | OracleDBClientProvider,
    tableName = DEFAULT_EXACT_TABLE_NAME
  ): Promise<void> {
    try {
      await withConnection(client, async (connection) => {
        await dropTablePurge(connection, tableName);
      });
    } catch (error) {
      handleError(error);
    }
  }
}
