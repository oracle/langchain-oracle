import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import oracledb from "oracledb";
import { Document } from "@langchain/core/documents";
import {
  BaseRetriever,
  type BaseRetrieverInput,
} from "@langchain/core/retrievers";
import type { RunnableConfig } from "@langchain/core/runnables";
import { OracleEmbeddings } from "./embeddings.js";
import { ErrorCode, handleError, throwError } from "./errors.js";
import type { OracleVS } from "./vectorstores.js";
import {
  type Client,
  indexExist,
  quoteIdentifier,
  withConnection,
} from "./utils.js";

type Params = Record<string, unknown>;

function objectParams(value: unknown, name: string): Params {
  if (value == null) return {};
  if (typeof value !== "object" || Array.isArray(value)) {
    throwError(ErrorCode.VALIDATION_INVALID_INPUT, `${name} must be an object`);
  }

  return value as Params;
}

// Builds vectorizer preference parameters from the store's OracleEmbeddings.
function getPreferenceParams(store: OracleVS, params: Params): Params {
  const embeddings = store.embeddings;
  // eslint-disable-next-line no-instanceof/no-instanceof
  if (!(embeddings instanceof OracleEmbeddings)) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "Only OracleEmbeddings can be used to create a vectorizer preference"
    );
  }

  const embeddingParams = embeddings.pref;
  const result = structuredClone(params);
  if ("model" in result && "embedder_spec" in result) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "model and embedder_spec are mutually exclusive"
    );
  }

  if ("model" in result) {
    if (
      embeddingParams.provider !== "database" ||
      result.model !== embeddingParams.model
    ) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "model must match the database model in OracleEmbeddings.pref"
      );
    }
  } else if ("embedder_spec" in result) {
    if (!isDeepStrictEqual(result.embedder_spec, embeddingParams)) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "embedder_spec must exactly match OracleEmbeddings.params"
      );
    }
  } else if (embeddingParams.provider === "database") {
    if (embeddingParams.model == null) {
      throwError(
        ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER,
        "OracleEmbeddings.pref.model"
      );
    }
    if (
      typeof embeddingParams.model !== "string" ||
      !embeddingParams.model.trim()
    ) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "OracleEmbeddings.pref.model is required for a database provider"
      );
    }
    result.model = embeddingParams.model;
  } else {
    result.embedder_spec = embeddingParams;
  }

  return result;
}

function validatePreferenceName(name: string): void {
  if (name == null) {
    throwError(
      ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER,
      "preferenceName"
    );
  }

  if (typeof name !== "string" || !/^[A-Za-z][A-Za-z0-9_$#]*$/.test(name)) {
    throwError(ErrorCode.VALIDATION_INVALID_IDENTIFIER, name);
  }
}

/**
 * Manages DBMS_VECTOR_CHAIN vectorizer preferences used by hybrid vector indexes.
 * Derives embedding configuration from the OracleEmbeddings attached to an OracleVS.
 * Database providers use a model name; external providers use an embedder specification.
 *
 * Call `createPreference()` to create the preference in Oracle Database.
 * An explicitly created preference remains available until `dropPreference()` is called.
 * @example
 * ```typescript
 * // `vectorStore` is an initialized OracleVS using OracleEmbeddings.
 * const preference = await OracleVectorizerPreference.createPreference(
 *   vectorStore,
 *   "PREF_DOCS",
 *   { by: "words", max: 100, overlap: 10 }
 * );
 * await createHybridIndex(vectorStore.client, "IDX_DOCS_HYB", {
 *   vectorizerPreference: preference,
 *   parallel: 2,
 * });
 *
 * // Drop the preference when it is no longer needed.
 * await preference.dropPreference();
 * ```
 */
export class OracleVectorizerPreference {
  private constructor(
    readonly vectorStore: OracleVS,
    readonly preferenceName: string,
    readonly params: Params
  ) {}

  /**
   * Creates a vectorizer preference using the store's embedding configuration.
   * Explicit `model` or `embedder_spec` settings must match OracleEmbeddings.pref
   * and cannot be supplied together. Additional chunking and vector index settings
   * are passed to Oracle with the resolved embedding configuration.
   * @param {OracleVS} vectorStore Store whose OracleEmbeddings configure the vectorizer.
   * @param {string} preferenceName Simple Oracle identifier. Defaults to a generated `PREF_` name.
   * @param {Record<string, unknown>} params Optional vectorizer preference settings.
   * @returns Promise resolving to the created preference and its resolved parameters.
   * @throws If the name or embedding configuration is invalid, the store does not
   * use OracleEmbeddings, or database preference creation fails.
   */
  static async createPreference(
    vectorStore: OracleVS,
    preferenceName = `PREF_${randomUUID().replace(/-/g, "").slice(0, 15)}`,
    params: Params = {}
  ): Promise<OracleVectorizerPreference> {
    try {
      validatePreferenceName(preferenceName);
      const prefParams = getPreferenceParams(vectorStore, params);
      await withConnection(vectorStore.client, async (connection) => {
        await executeCreatePreference(connection, preferenceName, prefParams);
      });

      return new OracleVectorizerPreference(
        vectorStore,
        preferenceName,
        prefParams
      );
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Drops this vectorizer preference using the vector store's client.
   */
  async dropPreference(): Promise<void> {
    await dropPreference(this.vectorStore.client, this.preferenceName);
  }
}

async function executeCreatePreference(
  connection: oracledb.Connection,
  name: string,
  params: Params
): Promise<void> {
  await connection.execute(
    `BEGIN DBMS_VECTOR_CHAIN.CREATE_PREFERENCE(
      :preference_name,
      DBMS_VECTOR_CHAIN.VECTORIZER,
      :params);
    END;`,
    {
      preference_name: name,
      params: { val: params, type: oracledb.DB_TYPE_JSON },
    }
  );
}

/**
 * Drops a named DBMS_VECTOR_CHAIN preference.
 * @param {Client} client Connection, pool, or async provider to use.
 * @param {string} preferenceName Name of the preference to drop.
 * @returns Promise that resolves when the preference has been dropped.
 */
export async function dropPreference(
  client: Client,
  preferenceName: string
): Promise<void> {
  try {
    validatePreferenceName(preferenceName);
    await withConnection(client, async (connection) => {
      await executeDropPreference(connection, preferenceName);
    });
  } catch (error) {
    handleError(error);
  }
}

async function executeDropPreference(
  connection: oracledb.Connection,
  preferenceName: string
): Promise<void> {
  await connection.execute(
    "BEGIN DBMS_VECTOR_CHAIN.DROP_PREFERENCE(:preference_name); END;",
    { preference_name: preferenceName }
  );
}

export interface HybridIndexParams {
  parameters?: Record<string, string | number>;
  filterBy?: string[];
  orderBy?: string[];
  orderByAsc?: boolean;
  parallel?: number;
}

export type CreateHybridIndexOptions = HybridIndexParams &
  (
    | { vectorStore: OracleVS; vectorizerPreference?: never }
    | { vectorizerPreference: OracleVectorizerPreference; vectorStore?: never }
  );

function quoteIdentifierList(values: string[], name: string): string {
  if (!Array.isArray(values) || values.some((v) => typeof v !== "string")) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      `${name} must contain only column names`
    );
  }

  const baseIdentifier = "[A-Za-z][A-Za-z0-9_$#]*";
  const columnName = `(?:"${baseIdentifier}"|${baseIdentifier})`;
  const reg = new RegExp(`^${columnName}(?:\\.${columnName})*$`);

  return values
    .map((value) => {
      const identifier = value.trim();
      if (!reg.test(identifier)) {
        throwError(
          ErrorCode.VALIDATION_INVALID_IDENTIFIER,
          `${name}: ${value}`
        );
      }

      return identifier
        .split(".")
        .map((part) =>
          part.startsWith('"') ? part : `"${part.toUpperCase()}"`
        )
        .join(".");
    })
    .join(",");
}

/** Build the CREATE HYBRID VECTOR INDEX DDL stateme */
function hybridIndexDDL(
  store: OracleVS,
  name: string,
  preference: string,
  params: HybridIndexParams
): string {
  validatePreferenceName(preference);
  const parameters = [`vectorizer ${preference}`];
  for (const [key, value] of Object.entries(params.parameters ?? {})) {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(key))
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "Invalid index parameter name"
      );

    if (
      ["model", "embedder_spec", "vector_idxtype", "vectorizer"].includes(
        key.toLowerCase()
      )
    ) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "Vectorization parameters must be given with OracleVectorizerPreference" +
          "do not include any of {model, embedder_spec, vector_idxtype, vectorizer}" +
          "under params.parameters."
      );
    }

    if (typeof value !== "string" && typeof value !== "number") {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "Invalid index parameter value"
      );
    }

    parameters.push(`${key} ${value}`);
  }

  if (
    params.orderByAsc !== undefined &&
    typeof params.orderByAsc !== "boolean"
  ) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "orderByAsc must be a boolean"
    );
  }

  const clauses = [
    `CREATE HYBRID VECTOR INDEX ${name} ON ${quoteIdentifier(
      store.tableName
    )}(text)`,
  ];

  if (params.filterBy !== undefined) {
    const columns = quoteIdentifierList(params.filterBy, "filterBy");
    if (columns) clauses.push(`FILTER BY ${columns}`);
  }

  if (params.orderBy !== undefined) {
    const columns = quoteIdentifierList(params.orderBy, "orderBy");
    if (columns)
      clauses.push(
        `ORDER BY ${columns} ${params.orderByAsc === false ? "DESC" : "ASC"}`
      );
  }

  clauses.push(`PARAMETERS ('${parameters.join(" ").replace(/'/g, "''")}')`);
  if (params.parallel !== undefined) {
    if (!Number.isInteger(params.parallel) || params.parallel <= 0) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "parallel must be a positive integer."
      );
    }
    clauses.push(`PARALLEL ${params.parallel}`);
  }

  return clauses.join("\n");
}

/**
 * Creates a hybrid vector index if it does not already exist.
 * The document table must already exist. Supply exactly one of `vectorStore` or
 * `vectorizerPreference`. A store creates a temporary preference that is dropped
 * after index creation or on DDL failure. A supplied preference is left in place.
 *
 * @param {Client} client Connection, pool, or async provider used for index creation.
 * @param {string} idxName Index name, quoted before use in SQL.
 * @param {CreateHybridIndexOptions} options Vectorizer source and optional index settings.
 * @param {OracleVS} options.vectorStore Store using OracleEmbeddings for a temporary preference.
 * @param {OracleVectorizerPreference} options.vectorizerPreference Existing vectorizer preference.
 * @returns Promise that resolves when the index has been created or already exists.
 * @throws LangChainOracleError if options are invalid or database setup fails.
 * @example
 * ```typescript
 * // `vectorStore` is an initialized OracleVS using OracleEmbeddings.
 * await createHybridIndex(vectorStore.client, "IDX_DOCS_HYB", {
 *   vectorStore,
 *   parallel: 2,
 * });
 * ```
 */
export async function createHybridIndex(
  client: Client,
  idxName: string,
  options: CreateHybridIndexOptions
): Promise<void> {
  try {
    if (
      Boolean(options.vectorStore) === Boolean(options.vectorizerPreference)
    ) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "Exactly one of vectorStore or vectorizerPreference must be provided"
      );
    }
    const store =
      options.vectorStore ?? options.vectorizerPreference!.vectorStore;
    const preference =
      options.vectorizerPreference?.preferenceName ??
      `PREF_${randomUUID().replace(/-/g, "").slice(0, 15)}`;
    const indexName = quoteIdentifier(idxName);
    const tableName = quoteIdentifier(store.tableName);
    const ddl = hybridIndexDDL(store, indexName, preference, options);
    const ownsPreference = Boolean(options.vectorStore);
    await withConnection(client, async (connection) => {
      if (await indexExist(connection, indexName, tableName)) {
        console.log(`Index ${idxName} already exists`);
        return;
      }

      if (ownsPreference) {
        const params = getPreferenceParams(store, {});

        await executeCreatePreference(connection, preference, params);
      }

      try {
        await connection.execute(ddl);
      } catch (ddlError) {
        if (ownsPreference) {
          try {
            await executeDropPreference(connection, preference);
          } catch (cleanupError) {
            throw new AggregateError(
              [ddlError, cleanupError],
              "Index creation and preference cleanup failed"
            );
          }
        }

        throw ddlError;
      }

      if (ownsPreference) {
        await executeDropPreference(connection, preference);
      }
    });
  } catch (error) {
    handleError(error);
  }
}

export type HybridSearchMode = "hybrid" | "keyword" | "semantic";

export interface HybridSearchParams extends Params {
  vector?: Params & {
    search_text?: string;
    search_vector?: number[] | string;
  };

  text?: Params & {
    search_text?: string;
    contains?: string;
  };
}

export interface HybridSearchCallOptions {
  k?: number;
  params?: HybridSearchParams;
}

export interface OracleHybridSearchRetrieverInput
  extends BaseRetrieverInput,
    HybridSearchCallOptions {
  vectorStore: OracleVS;
  idxName: string;
  searchMode?: HybridSearchMode;
  returnScores?: boolean;
}

function validateSearchParams(
  params: HybridSearchParams,
  searchMode: HybridSearchMode
): void {
  params = objectParams(params, "params");
  for (const key of [
    "search_text",
    "search_vector",
    "return",
    "hybrid_index_name",
  ]) {
    if (key in params)
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        `${key} is managed by the retriever and cannot be supplied in params`
      );
  }

  for (const section of ["vector", "text"] as const) {
    const nested = objectParams(params[section], section);
    const invalidKeys = section === "vector" ? ["contains"] : ["search_vector"];
    for (const key of invalidKeys) {
      if (key in nested)
        throwError(
          ErrorCode.VALIDATION_INVALID_INPUT,
          `${key} cannot be supplied in params.${section}`
        );
    }
    const queryKeys =
      section === "vector"
        ? ["search_text", "search_vector"]
        : ["search_text", "contains"];

    for (const key of queryKeys) {
      const value = nested[key];
      if (value === undefined) continue;

      // Makes sure search_text and contains are non-empty strings when provided.
      if (
        key !== "search_vector" &&
        (typeof value !== "string" || !value.trim())
      ) {
        throwError(
          ErrorCode.VALIDATION_INVALID_INPUT,
          `${section}.${key} must be a nonempty string`
        );
      }
      if (
        (section === "text" && searchMode === "semantic") ||
        (section === "vector" && searchMode === "keyword")
      ) {
        throwError(
          ErrorCode.VALIDATION_INVALID_INPUT,
          `${section}.${key} cannot be used with searchMode "${searchMode}". Use "hybrid" or "${
            section === "text" ? "keyword" : "semantic"
          }".`
        );
      }
    }
  }

  const { vector } = params;

  if (vector?.search_text !== undefined && vector.search_vector !== undefined) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "Provide either params.vector.search_text for Oracle to embed, or " +
        "params.vector.search_vector containing an existing query embedding."
    );
  }

  const searchVector = vector?.search_vector;

  if (
    searchVector !== undefined &&
    !Array.isArray(searchVector) &&
    (typeof searchVector !== "string" || !searchVector.trim())
  ) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "vector.search_vector must be a numeric array or a nonempty serialized vector string"
    );
  }
}

/**
 * Retrieves OracleVS documents through DBMS_HYBRID_VECTOR.SEARCH.
 * Supports combined vector/text ranking, keyword search, and semantic search.
 * Semantic query text uses the embedding model configured in the hybrid index's
 * vectorizer. An explicit vector.search_vector bypasses query text embedding.
 *
 * Create the hybrid index before invoking the retriever. Results preserve the
 * database ranking and contain the stored text, metadata, and external document ID.
 * With vector.search_mode set to CHUNK, results use the matching chunk text when
 * available and include its chunk_id in metadata, replacing any stored chunk_id.
 * Hits without chunk text retain the full stored text.
 * When `returnScores` is true, database scores overwrite any stored metadata config
 * named score, text_score, or vector_score in the returned documents.
 * @example
 * ```typescript
 * // `vectorStore` has an existing hybrid index named IDX_DOCS_HYB.
 * const retriever = new OracleHybridSearchRetriever({
 *   vectorStore,
 *   idxName: "IDX_DOCS_HYB",
 *   searchMode: "hybrid",
 *   k: 4,
 *   returnScores: true,
 * });
 * const documents = await retriever.invoke("refund policy for premium plan", {
 *   k: 2,
 *   params: { vector: { score_weight: 1 } },
 * });
 * ```
 */
export class OracleHybridSearchRetriever extends BaseRetriever {
  lc_namespace = ["langchain_oracledb"];

  readonly vectorStore: OracleVS;
  readonly idxName: string;
  readonly searchMode: HybridSearchMode;
  readonly k: number;
  readonly params: HybridSearchParams;
  readonly returnScores: boolean;

  /**
   * @param {OracleHybridSearchRetrieverInput} config Retriever configuration.
   * @param {OracleVS} config.vectorStore Store containing the documents to retrieve.
   * @param {string} config.idxName Hybrid index name, quoted before use.
   * @param {HybridSearchMode} config.searchMode Search strategy. Defaults to `hybrid`.
   * @param {number} config.k Positive integer maximum number of results. Defaults to 4.
   * @param {HybridSearchParams} config.params Additional search settings.
   * @param {boolean} config.returnScores Whether to add database scores to metadata. Defaults to false.
   */
  constructor(config: OracleHybridSearchRetrieverInput) {
    super(config ?? {});
    if (config == null) {
      throwError(ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER, "config");
    }

    this.vectorStore = config.vectorStore;
    this.idxName = quoteIdentifier(config.idxName);
    this.searchMode = config.searchMode ?? "hybrid";

    if (!["hybrid", "keyword", "semantic"].includes(this.searchMode)) {
      throwError(ErrorCode.VALIDATION_INVALID_INPUT, "Invalid searchMode");
    }

    this.k = config.k ?? 4;

    if (!Number.isInteger(this.k) || this.k <= 0) {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "k must be a positive integer."
      );
    }

    validateSearchParams(config.params ?? {}, this.searchMode);
    this.params = structuredClone(config.params ?? {});
    this.returnScores = config.returnScores ?? false;
  }

  /**
   * Searches the hybrid index with optional overrides for this call only.
   * Parameter merging is shallow: supplying `params.vector` or `params.text`
   * replaces that entire section of the constructor settings. Later calls retain
   * the original defaults. Standard LangChain runnable configuration is forwarded.
   * Explicit inputs in params.vector and params.text take precedence over query.
   * Text inputs must be nonempty strings. The query is checked only when used
   * as a default for an active search. Oracle validates query syntax and vector compatibility.
   * @param {string} query Default text for each active search without an explicit query override.
   * @param {RunnableConfig & HybridSearchCallOptions} options Runnable configuration and search overrides.
   * @param {number} options.k Optional positive integer maximum number of results.
   * @param {HybridSearchParams} options.params Optional search settings merged over the defaults.
   * @returns Promise resolving to ranked documents, or an empty array when no rows match.
   * @example
   * ```typescript
   * await retriever.invoke("How do I restore a database after a failure?", {
   *   params: {
   *     text: { contains: "RMAN AND recovery" },
   *     search_fusion: "INTERSECT",
   *   },
   * });
   * ```
   */
  async invoke(
    query: string,
    options?: RunnableConfig & HybridSearchCallOptions
  ): Promise<Document[]> {
    try {
      if (options?.k === undefined && options?.params === undefined)
        return await super.invoke(query, options);

      if (options.params !== undefined) {
        validateSearchParams(options.params, this.searchMode);
      }

      const invocationRetriever = new OracleHybridSearchRetriever({
        vectorStore: this.vectorStore,
        idxName: this.idxName,
        searchMode: this.searchMode,
        returnScores: this.returnScores,
        k: options.k ?? this.k,
        params: { ...this.params, ...options.params },
        callbacks: this.callbacks,
        tags: this.tags,
        metadata: this.metadata,
        verbose: this.verbose,
      });

      const config = { ...options };
      delete config.k;
      delete config.params;

      return await invocationRetriever.invoke(query, config);
    } catch (error) {
      handleError(error);
    }
  }

  /**
   * Execute a hybrid search via DBMS_HYBRID_VECTOR.SEARCH.
   * Used by BaseRetriever; use `invoke()` for callbacks and overrides for individual calls.
   * @param {string} query Text to search for using the configured mode.
   * @returns Promise resolving to documents in database result order.
   */
  async _getRelevantDocuments(query: string): Promise<Document[]> {
    try {
      validateSearchParams(this.params, this.searchMode);
      const params: HybridSearchParams = {
        ...this.params,
        hybrid_index_name: this.idxName,
      };

      // Keyword search should send only text settings; semantic search only vector
      // settings. Clear both here, then add back what the selected search mode needs
      delete params.vector;
      delete params.text;
      let usesQuery = false;

      if (this.searchMode !== "keyword") {
        const vector = { ...this.params.vector };
        if (
          vector.search_text === undefined &&
          vector.search_vector === undefined
        ) {
          vector.search_text = query;
          usesQuery = true;
        }
        if (Array.isArray(vector.search_vector))
          vector.search_vector = JSON.stringify(vector.search_vector);

        params.vector = vector;
      }

      if (this.searchMode !== "semantic") {
        const text = { ...this.params.text };
        if (text.search_text === undefined && text.contains === undefined) {
          text.search_text = query;
          usesQuery = true;
        }

        params.text = text;
      }

      if (usesQuery && (typeof query !== "string" || !query.trim())) {
        throwError(
          ErrorCode.VALIDATION_INVALID_INPUT,
          "query must be a nonempty string when used as search input"
        );
      }

      const returnChunks =
        typeof params.vector?.search_mode === "string" &&
        params.vector.search_mode.toUpperCase() === "CHUNK";

      params.return = {
        topN: this.k,
        values: [
          "rowid",
          "score",
          "vector_score",
          "text_score",
          ...(returnChunks ? ["chunk_text", "chunk_id"] : []),
        ],
        format: "JSON",
      };

      return await withConnection(
        this.vectorStore.client,
        async (connection) => {
          const result = await connection.execute<
            [
              string | null,
              string | null,
              Params | string | null,
              number | null,
              number | null,
              number | null,
              string | null,
              string | null
            ]
          >(
            `SELECT 
            document.external_id, 
            document.text AS text, 
            document.metadata, 
            hit.score, 
            hit.text_score, 
            hit.vector_score,
            hit.chunk_text,
            hit.chunk_id
           FROM JSON_TABLE(
             DBMS_HYBRID_VECTOR.SEARCH(JSON(:search_params)), 
             '$[*]'
             COLUMNS (
               result_order FOR ORDINALITY,
               rid VARCHAR2(4000) PATH '$.rowid',
               score NUMBER PATH '$.score',
               text_score NUMBER PATH '$.text_score',
               vector_score NUMBER PATH '$.vector_score',
               chunk_text CLOB PATH '$.chunk_text',
               chunk_id VARCHAR2(4000) PATH '$.chunk_id'
             )
           ) hit
           JOIN ${quoteIdentifier(this.vectorStore.tableName)} document 
            ON document.ROWID = CHARTOROWID(hit.rid) 
           ORDER BY hit.result_order`,
            { search_params: { val: params, type: oracledb.DB_TYPE_JSON } },
            {
              outFormat: oracledb.OUT_FORMAT_ARRAY,
              fetchInfo: {
                TEXT: { type: oracledb.STRING },
                CHUNK_TEXT: { type: oracledb.STRING },
              },
            }
          );

          return (result.rows ?? []).map(
            ([
              id,
              text,
              storedMetadata,
              score,
              textScore,
              vectorScore,
              chunkText,
              chunkId,
            ]) => {
              const metadata = {
                ...objectParams(
                  typeof storedMetadata === "string"
                    ? JSON.parse(storedMetadata)
                    : storedMetadata,
                  "metadata"
                ),
              };

              if (this.returnScores)
                Object.assign(metadata, {
                  score,
                  text_score: textScore,
                  vector_score: vectorScore,
                });

              if (returnChunks && chunkId != null) metadata.chunk_id = chunkId;

              return new Document({
                id: id ?? undefined,
                pageContent: (returnChunks ? chunkText ?? text : text) ?? "",
                metadata,
              });
            }
          );
        }
      );
    } catch (error) {
      handleError(error);
    }
  }
}
