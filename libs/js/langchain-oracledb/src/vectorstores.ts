import crypto from "node:crypto";
import oracledb from "oracledb";
import {
  type MaxMarginalRelevanceSearchOptions,
  VectorStore,
} from "@langchain/core/vectorstores";
import { Document, type DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";

export type Metadata = Record<string, unknown>;

export function generateWhereClause(
  dbFilter: Metadata,
  bindValues: unknown[]
): string {
  // Handle $and
  if ("$and" in dbFilter && Array.isArray(dbFilter.$and)) {
    const andConditions = dbFilter.$and.map((cond) =>
      generateWhereClause(cond as Metadata, bindValues)
    );
    return `(${andConditions.join(" AND ")})`;
  }

  // Handle $or
  if ("$or" in dbFilter && Array.isArray(dbFilter.$or)) {
    const orConditions = dbFilter.$or.map((cond) =>
      generateWhereClause(cond as Metadata, bindValues)
    );
    return `(${orConditions.join(" OR ")})`;
  }

  // Otherwise, normal object with key: condition(s)
  const conditions: string[] = [];
  for (const [col, val] of Object.entries(dbFilter)) {
    if (typeof val === "object" && val !== null && !Array.isArray(val)) {
      // Operator-based filters ($eq, $lte, $in,...)
      for (const [op, operand] of Object.entries(val)) {
        conditions.push(
          generateOperatorCondition(col, op, operand, bindValues)
        );
      }
    } else if (Array.isArray(val)) {
      // shorthand { tags: ["a","b"] } → IN
      conditions.push(generateOperatorCondition(col, "$in", val, bindValues));
    } else {
      // shorthand { name: "John" } → equality
      conditions.push(generateOperatorCondition(col, "$eq", val, bindValues));
    }
  }

  return conditions.length > 1
    ? `(${conditions.join(" AND ")})`
    : conditions[0];
}

function generateOperatorCondition(
  column: string,
  operator: string,
  value: unknown,
  bindValues: unknown[]
): string {
  switch (operator) {
    case "$eq":
      return jsonCompare(column, "=", value, bindValues);

    case "$ne":
      return jsonCompare(column, "!=", value, bindValues);

    case "$lt":
      return jsonCompare(column, "<", value, bindValues);

    case "$lte":
      return jsonCompare(column, "<=", value, bindValues);

    case "$gt":
      return jsonCompare(column, ">", value, bindValues);

    case "$gte":
      return jsonCompare(column, ">=", value, bindValues);

    case "$in": {
      if (!Array.isArray(value)) throw new Error("$in requires array");
      const inClauses = value.map((v) =>
        jsonCompare(column, "=", v, bindValues)
      );
      return `(${inClauses.join(" OR ")})`;
    }

    case "$nin": {
      if (!Array.isArray(value)) throw new Error("$nin requires array");
      const ninClauses = value.map((v) =>
        jsonCompare(column, "!=", v, bindValues)
      );
      return `(${ninClauses.join(" AND ")})`;
    }

    case "$between": {
      if (!Array.isArray(value) || value.length !== 2)
        throw new Error("$between requires [low, high]");
      const [low, high] = value;
      bindValues.push(low, high);
      const posLow = bindValues.length - 1;
      const posHigh = bindValues.length;
      return `JSON_EXISTS(metadata, '$.${column}?(@ >= $low && @ <= $high)' PASSING :${posLow} AS "low", :${posHigh} AS "high")`;
    }

    case "$exists":
      if (value) {
        return `JSON_EXISTS(metadata, '$.${column}')`;
      } else {
        return `NOT JSON_EXISTS(metadata, '$.${column}')`;
      }

    default:
      throw new Error(`Unsupported operator: ${operator}`);
  }
}

// Helper: build JSON_EXISTS clause for comparison
function jsonCompare(
  column: string,
  op: string,
  value: unknown,
  bindValues: unknown[]
): string {
  bindValues.push(value);
  const pos = bindValues.length;
  const alias = `val${pos}`; // unique bind alias
  const operator = op === "=" ? "==" : op; // Oracle requires '==' for equality
  return `JSON_EXISTS(metadata, '$.${column}?(@ ${operator} $${alias})' PASSING :${pos} AS "${alias}")`;
}

export const VectorType = {
  DENSE: "DENSE",
  SPARSE: "SPARSE",
} as const;

export type VectorType = (typeof VectorType)[keyof typeof VectorType];

export const VectorElementFormat = {
  INT8: "INT8",
  FLOAT32: "FLOAT32",
  FLOAT64: "FLOAT64",
  BINARY: "BINARY",
  FLEX: "*",
} as const;

export type VectorElementFormat =
  (typeof VectorElementFormat)[keyof typeof VectorElementFormat];

export interface OracleDBVSArgs {
  tableName: string;
  schemaName?: string | null;
  client: oracledb.Pool | oracledb.Connection;
  query: string;
  distanceStrategy?: DistanceStrategy;
  filter?: Metadata;
  description?: string;
  annotations?: Record<string, string>;
  vectorType?: VectorType;
  format?: VectorElementFormat;
}

export const DistanceStrategy = {
  COSINE: "COSINE",
  DOT_PRODUCT: "DOT",
  EUCLIDEAN: "EUCLIDEAN",
  MANHATTAN: "MANHATTAN",
  HAMMING: "HAMMING",
  EUCLIDEAN_SQUARED: "EUCLIDEAN_SQUARED",
} as const;

export type DistanceStrategy =
  (typeof DistanceStrategy)[keyof typeof DistanceStrategy];

type AddDocumentOptions = Record<string, any>;

function handleError(error: unknown): never {
  // Type guard to check if the error is an object and has 'name' and 'message' properties
  if (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    "message" in error
  ) {
    const err = error as { name: string; message: string }; // Type assertion based on guarded checks

    // Handle specific error types based on the name property
    switch (err.name) {
      case "RuntimeError":
        throw new Error("Database operation failed due to a runtime error.");
      case "ValidationError":
        throw new Error("Operation failed due to a validation error.");
      default:
        throw new Error(
          `An unexpected error occurred during the operation. ${error}`
        );
    }
  }
  throw new Error(`An unknown and unexpected error occurred. ${error}`);
}

function isPool(
  client: oracledb.Connection | oracledb.Pool
): client is oracledb.Pool {
  return "getConnection" in client;
}

function quoteIdentifier(identifier: string) {
  const name = identifier.trim();

  const validateRegex = /^(?:"[^"]+"|[^".]+)(?:\.(?:"[^"]+"|[^".]+))*$/;
  if (!validateRegex.test(name)) {
    throw new Error(`Identifier name ${identifier} is not valid.`);
  }

  // extracts parts of the identifier with quoted and unquoted.
  const matchRegex = /"([^"]+)"|([^".]+)/g;
  const groups = [];

  for (const match of name.matchAll(matchRegex)) {
    groups.push(match[1] || match[2]);
  }
  const quotedParts = groups.map((g) => `"${g}"`);
  return quotedParts.join(".");
}

type TableCustomization = {
  vectorType?: VectorType;
  format?: VectorElementFormat;
  description?: string;
  annotations?: Record<string, string>;
};

type ReturnedEmbedding =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | number[];

const VALID_VECTOR_TYPES = new Set<VectorType>([
  VectorType.DENSE,
  VectorType.SPARSE,
]);

const VALID_VECTOR_FORMAT = new Set<VectorElementFormat>([
  VectorElementFormat.INT8,
  VectorElementFormat.FLOAT32,
  VectorElementFormat.FLOAT64,
  VectorElementFormat.BINARY,
  VectorElementFormat.FLEX,
]);

function normalizeVectorTypeValue(value?: VectorType): VectorType {
  if (!value) return VectorType.DENSE;
  const normalized = value.toUpperCase() as VectorType;
  if (!VALID_VECTOR_TYPES.has(normalized)) {
    throw new Error(`Vector storage type ${value} is not valid. Use DENSE or SPARSE.`);
  }
  return normalized;
}

function normalizeVectorFormat(
  value: VectorElementFormat | undefined,
  vectorType: VectorType,
): VectorElementFormat {
  if (!value) return VectorElementFormat.FLOAT32;
  const normalized = value.toUpperCase() as VectorElementFormat;
  if (!VALID_VECTOR_FORMAT.has(normalized)) {
    throw new Error(
      `Vector format ${value} is not valid. Use INT8, FLOAT32, FLOAT64, BINARY, or *.`,
    );
  }
  if (vectorType === VectorType.SPARSE && normalized === VectorElementFormat.BINARY) {
    throw new Error("BINARY format is not supported for SPARSE vectors.");
  }
  return normalized;
}

function buildVectorColumnDefinition(
  embeddingDim?: number | null,
  customization?: TableCustomization,
): string {
  if (embeddingDim === undefined || embeddingDim === null) {
    throw new Error("Embedding dimension is required to create the vector column.");
  }
  if (!Number.isInteger(embeddingDim) || embeddingDim <= 0) {
    throw new Error("Embedding dimension must be a positive integer.");
  }

  const vectorType = normalizeVectorTypeValue(customization?.vectorType);

  if (customization?.vectorType && customization?.format === undefined) {
    throw new Error("Vector type requires both dimensions and format to be specified.");
  }

  const format = normalizeVectorFormat(customization?.format, vectorType);

  if (format === VectorElementFormat.BINARY && embeddingDim % 8 !== 0) {
    throw new Error("BINARY vector format requires dimensions to be a multiple of 8.");
  }

  const dimensionSegment = String(embeddingDim);
  const formatSegment = format ?? VectorElementFormat.FLOAT32;

  if (vectorType === VectorType.SPARSE) {
    return `VECTOR(${dimensionSegment}, ${formatSegment}, SPARSE)`;
  }

  return `VECTOR(${dimensionSegment}, ${formatSegment}, DENSE)`;
}

function escapeCommentText(comment: string): string {
  return comment.replace(/'/g, "''");
}

export async function createTable(
  connection: oracledb.Connection,
  tableName: string,
  embeddingDim?: number | null,
  customization?: TableCustomization
): Promise<void> {
  const tableIdentifier = quoteIdentifier(tableName);
  const colsDict = {
    id: "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
    external_id: `VARCHAR2(36) UNIQUE`,
    embedding: buildVectorColumnDefinition(embeddingDim, customization),
    text: "CLOB",
    metadata: "JSON",
  };

  try {
    const ddlBody = Object.entries(colsDict)
      .map(([colName, colType]) => `${colName} ${colType}`)
      .join(", ");
    const ddl = `CREATE TABLE IF NOT EXISTS ${tableIdentifier}
                   (
                       ${ddlBody}
                   )`;
    await connection.execute(ddl);

    const tableDescription = customization?.description?.trim();
    const commentStatements: string[] = [];

    if (tableDescription) {
      const escapedDescription = escapeCommentText(tableDescription);
      commentStatements.push(
        `EXECUTE IMMEDIATE 'COMMENT ON TABLE ${tableIdentifier} IS ''${escapedDescription}''';`,
      );
    }

    if (customization?.annotations) {
      for (const [columnName, note] of Object.entries(
        customization.annotations
      )) {
        const trimmedNote = note?.trim();
        if (!trimmedNote) continue;
        const columnKey = columnName.trim().toLowerCase();
        if (!(columnKey in colsDict)) continue;
        const normalizedColumnIdentifier = `${tableIdentifier}.${quoteIdentifier(
          columnKey.toUpperCase()
        )}`;
        const escapedNote = escapeCommentText(trimmedNote);
        commentStatements.push(
          `EXECUTE IMMEDIATE 'COMMENT ON COLUMN ${normalizedColumnIdentifier} IS ''${escapedNote}''';`,
        );
      }
    }

    if (commentStatements.length > 0) {
      const block = `BEGIN\n  ${commentStatements.join("\n  ")}\nEND;`;
      await connection.execute(block);
    }
  } catch (error: unknown) {
    handleError(error);
  }
}

function _getIndexName(baseName: string): string {
  const uniqueId = crypto.randomUUID().replace(/-/g, "");
  return `${baseName}_${uniqueId}`;
}

function packBinaryVector(bits: number[]): Uint8Array {
  const byteLength = Math.ceil(bits.length / 8);
  const buffer = new Uint8Array(byteLength);
  for (let i = 0; i < bits.length; i += 1) {
    const bit = bits[i] > 0 ? 1 : 0;
    if (bit === 1) {
      const byteIndex = Math.floor(i / 8);
      const bitOffset = 7 - (i % 8);
      buffer[byteIndex] |= 1 << bitOffset;
    }
  }
  return buffer;
}

function convertDenseVectorForFormat(
  values: number[],
  format: VectorElementFormat,
): Float32Array | Float64Array | Int8Array | Uint8Array {
  switch (format) {
    case VectorElementFormat.FLOAT32:
    case undefined:
      return new Float32Array(values);
    case VectorElementFormat.FLOAT64:
      return new Float64Array(values);
    case VectorElementFormat.INT8: {
      const clamped = new Int8Array(values.length);
      for (let i = 0; i < values.length; i += 1) {
        const rounded = Math.round(values[i]);
        if (rounded < -128 || rounded > 127) {
          throw new Error("INT8 vector values must be within [-128, 127].");
        }
        clamped[i] = rounded;
      }
      return clamped;
    }
    case VectorElementFormat.BINARY:
      return packBinaryVector(values);
    case VectorElementFormat.FLEX:
      return new Float32Array(values);
    default:
      return new Float32Array(values);
  }
}

function buildSparseVector(
  values: number[],
  format: VectorElementFormat,
  dimension: number,
): oracledb.SparseVector {
  const indices: number[] = [];
  const denseValues: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (value === 0) continue;
    indices.push(i);
    if (format === VectorElementFormat.INT8) {
      const rounded = Math.round(value);
      if (rounded < -128 || rounded > 127) {
        throw new Error("INT8 sparse vector values must be within [-128, 127].");
      }
      denseValues.push(rounded);
    } else {
      denseValues.push(value);
    }
  }

  return new oracledb.SparseVector({
    values: denseValues,
    indices,
    numDimensions: dimension,
  });
}

function unpackBinaryVector(
  buffer: Uint8Array | Buffer,
  dimension: number,
): Float32Array {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const result = new Float32Array(dimension);
  for (let i = 0; i < dimension; i += 1) {
    const byteIndex = Math.floor(i / 8);
    const bitOffset = 7 - (i % 8);
    const byte = bytes[byteIndex];
    result[i] = (byte >> bitOffset) & 1;
  }
  return result;
}

export async function createIndex(
  client: oracledb.Connection,
  vectorStore: OracleVS,
  params?: { [key: string]: any }
): Promise<void> {
  const idxType = params?.idxType || "HNSW";

  if (idxType === "IVF") {
    await createIVFIndex(client, vectorStore, params);
  } else {
    await createHNSWIndex(client, vectorStore, params);
  }
}

async function createHNSWIndex(
  connection: oracledb.Connection,
  oraclevs: OracleVS,
  params?: { [key: string]: any }
): Promise<void> {
  try {
    const defaults: { [key: string]: any } = {
      idxName: "HNSW",
      idxType: "HNSW",
      neighbors: 32,
      efConstruction: 200,
      accuracy: 90,
      parallel: 8,
    };

    // if params then copy params to config
    const config: { [key: string]: any } = params
      ? { ...params }
      : { ...defaults };

    // Ensure compulsory parts are included
    const compulsoryKeys = ["idxName", "parallel"];
    for (const key of compulsoryKeys) {
      if (!(key in config)) {
        if (key === "idxName") {
          config[key] = _getIndexName(defaults[key] as string);
        } else {
          config[key] = defaults[key] as number;
        }
      }
    }

    // Validate keys in config against defaults
    for (const key of Object.keys(config)) {
      if (!(key in defaults)) {
        throw new Error(`Invalid parameter: ${key}`);
      }
    }

    const { idxName } = config;
    const baseSql = `CREATE VECTOR INDEX IF NOT EXISTS ${quoteIdentifier(
      idxName
    )}
                              ON ${oraclevs.tableName}(embedding) 
                              ORGANIZATION INMEMORY NEIGHBOR GRAPH`;
    const accuracyPart = config.accuracy
      ? ` WITH TARGET ACCURACY ${config.accuracy}`
      : "";
    const distancePart = ` DISTANCE ${oraclevs.distanceStrategy}`;

    let parametersPart = "";
    if ("neighbors" in config && "efConstruction" in config) {
      parametersPart = ` parameters (type ${config.idxType}, 
                                     neighbors ${config.neighbors}, 
                                     efConstruction ${config.efConstruction})`;
    } else if ("neighbors" in config && !("efConstruction" in config)) {
      config.efConstruction = defaults.efConstruction;
      parametersPart = ` parameters (type ${config.idxType}, 
                                     neighbors ${config.neighbors}, 
                                     efConstruction ${config.efConstruction})`;
    } else if (!("neighbors" in config) && "efConstruction" in config) {
      config.neighbors = defaults.neighbors;
      parametersPart = ` parameters (type ${config.idxType}, 
                                     neighbors ${config.neighbors}, 
                                     efConstruction ${config.efConstruction})`;
    }

    const parallelPart = ` PARALLEL ${config.parallel}`;
    const ddl =
      baseSql + accuracyPart + distancePart + parametersPart + parallelPart;
    await connection.execute(ddl);
  } catch (error: unknown) {
    handleError(error);
  }
}

async function createIVFIndex(
  connection: oracledb.Connection,
  oraclevs: OracleVS,
  params?: { [key: string]: any }
): Promise<void> {
  try {
    const defaults: { [key: string]: any } = {
      idxName: "IVF",
      idxType: "IVF",
      neighborPart: 32,
      accuracy: 90,
      parallel: 8,
    };

    // Combine defaults with any provided params. Note: params could contain keys not explicitly declared in IndexConfig
    const config: { [key: string]: any } = params
      ? { ...params }
      : { ...defaults };

    // Ensure compulsory parts are included
    const compulsoryKeys = ["idxName", "parallel"];
    for (const key of compulsoryKeys) {
      if (!(key in config)) {
        if (key === "idxName") {
          config[key] = _getIndexName(defaults[key] as string);
        } else {
          config[key] = defaults[key] as number;
        }
      }
    }

    // Validate keys in config against defaults
    for (const key of Object.keys(config)) {
      if (!(key in defaults)) {
        throw new Error(`Invalid parameter: ${key}`);
      }
    }

    // Base SQL statement
    const { idxName } = config;
    const baseSql = `CREATE VECTOR INDEX IF NOT EXISTS ${quoteIdentifier(
      idxName
    )}
                              ON ${oraclevs.tableName}(embedding) 
                              ORGANIZATION NEIGHBOR PARTITIONS`;

    // Optional parts depending on parameters
    const accuracyPart = config.accuracy
      ? ` WITH TARGET ACCURACY ${config.accuracy}`
      : "";
    const distancePart = ` DISTANCE ${oraclevs.distanceStrategy}`;

    let parametersPart = "";
    if ("idxType" in config && "neighborPart" in config) {
      parametersPart = ` PARAMETERS (type ${config.idxType}, 
                         neighbor partitions ${config.neighborPart})`;
    }

    // Always included part for parallel - assuming parallel is compulsory and always included
    const parallelPart = ` PARALLEL ${config.parallel}`;

    // Combine all parts
    const ddl =
      baseSql + accuracyPart + distancePart + parametersPart + parallelPart;
    await connection.execute(ddl);
  } catch (error: unknown) {
    handleError(error);
  }
}

export async function dropTablePurge(
  connection: oracledb.Connection,
  tableName: string
): Promise<void> {
  try {
    const ddl = `DROP TABLE IF EXISTS ${quoteIdentifier(tableName)} PURGE`;
    await connection.execute(ddl);
  } catch (error: unknown) {
    handleError(error);
  }
}

export class OracleVS extends VectorStore {
  declare FilterType: Metadata;

  readonly client: oracledb.Pool | oracledb.Connection;

  embeddingDimension: number | undefined;

  readonly tableName: string;

  readonly distanceStrategy: DistanceStrategy = DistanceStrategy.COSINE;

  filter?: Metadata;

  readonly description?: string;

  readonly annotations?: Record<string, string>;

  readonly vectorType?: VectorType;

  readonly vectorFormat?: VectorElementFormat;

  readonly query: string;

  _vectorstoreType(): string {
    return "oraclevs";
  }

  constructor(embeddings: EmbeddingsInterface, dbConfig: OracleDBVSArgs) {
    super(embeddings, dbConfig);

    try {
      this.client = dbConfig.client;
      this.tableName = quoteIdentifier(dbConfig.tableName);
      this.distanceStrategy =
        dbConfig.distanceStrategy ?? this.distanceStrategy;
      this.query = dbConfig.query;
      this.filter = dbConfig.filter;
      this.description = dbConfig.description;
      this.annotations = dbConfig.annotations
        ? { ...dbConfig.annotations }
        : undefined;
      this.vectorType = dbConfig.vectorType;
      this.vectorFormat = dbConfig.format;
    } catch (error: unknown) {
      handleError(error);
    }
  }

  private getActiveVectorType(): VectorType {
    return this.vectorType ? normalizeVectorTypeValue(this.vectorType) : VectorType.DENSE;
  }

  private ensureEmbeddingDimension(): number {
    if (this.embeddingDimension === undefined || this.embeddingDimension === null) {
      throw new Error("Embedding dimension is not initialized for this vector store.");
    }
    return this.embeddingDimension;
  }

  private prepareVectorForStorage(vector: number[]): any {
    if (this.embeddingDimension === undefined || this.embeddingDimension === null) {
      this.embeddingDimension = vector.length;
    }
    const dimension = this.ensureEmbeddingDimension();
    const vectorType = this.getActiveVectorType();
    const format = this.vectorFormat
      ? normalizeVectorFormat(this.vectorFormat, vectorType)
      : VectorElementFormat.FLOAT32;

    if (vectorType === VectorType.SPARSE) {
      if (vector.length !== dimension) {
        throw new Error("Sparse vectors must supply full-dimension arrays for conversion.");
      }
      return buildSparseVector(vector, format, dimension);
    }

    if (vector.length !== dimension) {
      throw new Error("Vector length does not match the embedding dimension.");
    }

    return convertDenseVectorForFormat(vector, format);
  }

  private prepareQueryVector(vector: number[]): any {
    return this.prepareVectorForStorage(vector);
  }

  private normalizeReturnedEmbedding(value: unknown): ReturnedEmbedding {
    if (value instanceof oracledb.SparseVector) {
      const dense = value.dense?.();
      if (!dense) {
        throw new Error("Unable to expand sparse vector to dense representation.");
      }
      if (dense instanceof Float32Array) {
        return dense;
      }
      if (dense instanceof Float64Array) {
        return new Float32Array(dense);
      }
      if (dense instanceof Int8Array) {
        return Float32Array.from(dense);
      }
      if (Array.isArray(dense)) {
        return Float32Array.from(dense as number[]);
      }
      return Float32Array.from(dense as ArrayLike<number>);
    }
    if (
      value instanceof Float32Array ||
      value instanceof Float64Array ||
      value instanceof Int8Array ||
      value instanceof Uint8Array ||
      Array.isArray(value)
    ) {
      return value as ReturnedEmbedding;
    }
    throw new Error("Received unsupported vector representation from the database.");
  }

  private coerceEmbeddingToFloat32(value: unknown): Float32Array {
    if (value instanceof Float32Array) {
      return value;
    }
    if (value instanceof Float64Array) {
      return new Float32Array(value);
    }
    if (value instanceof Int8Array) {
      return Float32Array.from(value);
    }
    if (value instanceof Uint8Array) {
      const byteLength = value.length;
      const dimension = this.embeddingDimension ?? byteLength * 8;
      return unpackBinaryVector(value, dimension);
    }
    if (Array.isArray(value)) {
      return Float32Array.from(value as number[]);
    }
    return this.coerceEmbeddingToFloat32(this.normalizeReturnedEmbedding(value));
  }

  async getEmbeddingDimension(query: string): Promise<number> {
    const embeddingVector = await this.embeddings.embedQuery(query);
    return embeddingVector.length;
  }

  async initialize(): Promise<void> {
    let connection: oracledb.Connection | null = null;
    try {
      this.embeddingDimension = await this.getEmbeddingDimension(this.query);
      connection = await this.getConnection();
      await createTable(connection, this.tableName, this.embeddingDimension, {
        description: this.description,
        annotations: this.annotations,
        vectorType: this.vectorType,
        format: this.vectorFormat
      });
    } catch (error: unknown) {
      handleError(error);
    } finally {
      if (connection) await this.retConnection(connection);
    }
  }

  public async getConnection(): Promise<oracledb.Connection> {
    try {
      if (isPool(this.client)) {
        return await (this.client as oracledb.Pool).getConnection();
      }
      return this.client as oracledb.Connection;
    } catch (error: unknown) {
      handleError(error);
    }
  }

  // Close connection or return it to the pool
  public async retConnection(connection: oracledb.Connection): Promise<void> {
    try {
      // If the client is a pool, close the connection (return it to the pool)
      if (isPool(this.client)) {
        await connection.close();
      }
    } catch (error) {
      console.error("Error in retConnection:", error);
      throw error;
    }
  }

  /**
   * Method to add vectors to the Oracle database.
   * @param vectors The vectors to add.
   * @param documents The documents associated with the vectors.
   * @param options
   * ** Add { upsert?: boolean } to do upsert
   * @returns Promise that resolves when the vectors have been added.
   */
  public async addVectors(
    vectors: number[][],
    documents: DocumentInterface[],
    options?: AddDocumentOptions
  ): Promise<string[] | undefined> {
    if (vectors.length === 0) {
      throw new Error("Vectors input null. Nothing to add...");
    }

    const inputIds = options?.ids;
    let connection: oracledb.Connection | null = null;

    try {
      // Ensure there are IDs for all documents
      if (inputIds !== undefined && inputIds.length !== vectors.length) {
        throw new Error(
          "The number of ids must match the number of vectors provided."
        );
      }

      connection = await this.getConnection();
      const finalIds: string[] = [];
      const binds: any[] = [];

      for (let index = 0; index < documents.length; index += 1) {
        const doc = documents[index];

        // Generate ID if not provided: Priority (Options -> Metadata -> Random)
        const externalId =
          inputIds?.[index] ?? doc.metadata.id ?? crypto.randomUUID();
        finalIds.push(externalId);
        const preparedEmbedding = this.prepareVectorForStorage(
          vectors[index],
        );
        binds.push({
          ext_id: externalId,
          text: doc.pageContent,
          metadata: doc.metadata,
          embedding: preparedEmbedding,
        });
      }

      const isUpsert = options?.upsert ?? false; // Default to false for better performance
      const insertSql = `
        INSERT INTO ${this.tableName} (external_id, embedding, text, metadata)
        VALUES (:ext_id, :embedding, :text, :metadata)`;
      const mergeSql = `
      MERGE INTO ${this.tableName} t
      USING (
        SELECT :ext_id as external_id, :embedding as embedding,
               :metadata as metadata, :text as text FROM DUAL
      ) s
      ON (t.external_id = s.external_id)
      WHEN MATCHED THEN
        UPDATE SET
          t.embedding = s.embedding,
          t.metadata = s.metadata,
          t.text = s.text
      WHEN NOT MATCHED THEN
        INSERT (external_id, embedding, metadata, text)
        VALUES (s.external_id, s.embedding, s.metadata, s.text)`;

      const sql = isUpsert ? mergeSql : insertSql;
      const executeOptions = {
        bindDefs: {
          ext_id: { type: oracledb.STRING, maxSize: 255 },
          text: { type: oracledb.STRING, maxSize: 10000000 },
          metadata: { type: oracledb.DB_TYPE_JSON },
          embedding: { type: oracledb.DB_TYPE_VECTOR },
        },
        autoCommit: false,
      };

      await connection.executeMany(sql, binds, executeOptions);

      // Commit once all inserts are queued up
      await connection.commit();
      return finalIds;
    } catch (error: any) {
      return handleError(error);
    } finally {
      if (connection) {
        await this.retConnection(connection);
      }
    }
  }

  public async addDocuments(
    documents: DocumentInterface[],
    options?: AddDocumentOptions
  ): Promise<string[] | undefined> {
    const texts = documents.map(({ pageContent }) => pageContent);

    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  /**
   * Method to search for vectors that are similar to a given query vector.
   * @param query The query vector.
   * @param k The number of similar vectors to return.
   * @param filter Optional filter for the search results.
   * @returns Promise that resolves with an array of tuples, each containing a Document and a score.
   */
  public async similaritySearchByVectorReturningEmbeddings(
    query: number[],
    k = 4,
    filter?: this["FilterType"]
  ): Promise<[Document, number, ReturnedEmbedding][]> {
    const docsScoresAndEmbeddings: Array<
      [Document, number, ReturnedEmbedding]
    > = [];

    let connection: oracledb.Connection | null = null;

    try {
      const bindValues: any = [this.prepareQueryVector(query)];

      let sqlQuery = `
      SELECT external_id,
        text,
        metadata,
        vector_distance(embedding, :1, ${this.distanceStrategy}) as distance,
        embedding
      FROM ${this.tableName} `;
      if (filter && Object.keys(filter).length > 0) {
        sqlQuery += ` WHERE ${generateWhereClause(filter, bindValues)}`;
      }
      bindValues.push(k);
      sqlQuery += ` ORDER BY distance FETCH APPROX FIRST :${bindValues.length} ROWS ONLY `;

      // Execute the query
      connection = await this.getConnection();
      const resultSet = await connection.execute(sqlQuery, bindValues, {
        fetchInfo: {
          TEXT: { type: oracledb.STRING },
        },
      } as unknown as oracledb.ExecuteOptions);

      if (Array.isArray(resultSet.rows) && resultSet.rows.length > 0) {
        const rows = resultSet.rows as unknown[][];

        for (let idx = 0; idx < resultSet.rows.length; idx += 1) {
          const row = rows[idx];
          const text = row[1] as string;
          const metadata = row[2] as Metadata;
          const distance = row[3] as number;
          const embedding = this.normalizeReturnedEmbedding(row[4]);
          const document = new Document({
            pageContent: text || "",
            metadata: metadata || {},
            id: row[0] as string,
          });
          docsScoresAndEmbeddings.push([document, distance, embedding]);
        }
      } else {
        // Throw an exception if no rows are found
        throw new Error("No rows found.");
      }
    } finally {
      if (connection) {
        await connection.close();
      }
    }
    return docsScoresAndEmbeddings;
  }

  public async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"]
  ): Promise<[DocumentInterface, number][]> {
    const docsScoresAndEmbeddings =
      await this.similaritySearchByVectorReturningEmbeddings(query, k, filter);
    return docsScoresAndEmbeddings.map(([document, score]) => [
      document,
      score,
    ]);
  }

  /**
   * Return documents selected using the maximal marginal relevance.
   * Maximal marginal relevance optimizes for similarity to the query AND diversity
   * among selected documents.
   *
   * @param {string} query - Text to look up documents similar to.
   * @param options
   * @param {number} options.k - Number of documents to return.
   * @param {number} options.fetchK - Number of documents to fetch before passing to the MMR algorithm.
   * @param {number} options.lambda - Number between 0 and 1 that determines the degree of diversity among the results,
   *                 where 0 corresponds to maximum diversity and 1 to minimum diversity.
   * @param {this["FilterType"]} options.filter - Optional filter
   * @param _callbacks
   *
   * @returns {Promise<DocumentInterface[]>} - List of documents selected by maximal marginal relevance.
   */
  public async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
  ): Promise<Document[]> {
    const embedding = await this.embeddings.embedQuery(query);
    return await this.maxMarginalRelevanceSearchByVector(embedding, options);
  }

  public async maxMarginalRelevanceSearchByVector(
    embedding: number[],
    options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
  ): Promise<Document[]> {
    // Fetch documents and their scores. This calls the previously adapted function.
    const docsAndScores =
      await this.maxMarginalRelevanceSearchWithScoreByVector(
        embedding,
        options
      );

    // Extract and return only the documents from the results
    return docsAndScores.map((ds) => ds.document);
  }

  public async maxMarginalRelevanceSearchWithScoreByVector(
    embedding: number[],
    options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
  ): Promise<Array<{ document: Document; score: number }>> {
    // Fetch documents and their scores.
    const docsScoresEmbeddings =
      await this.similaritySearchByVectorReturningEmbeddings(
        embedding,
        options.fetchK,
        options.filter
      );

    if (!docsScoresEmbeddings.length) {
      return [];
    }

    // Split documents, scores, and embeddings
    const documents: Document[] = docsScoresEmbeddings.map(
      ([document]) => document
    );
    const scores: number[] = docsScoresEmbeddings.map(([, score]) => score);
    const consistentEmbeddings: number[][] = docsScoresEmbeddings.map(([, , emb]) =>
      Array.from(this.coerceEmbeddingToFloat32(emb))
    );
    const queryEmbedding: number[] = Array.from(this.coerceEmbeddingToFloat32(embedding));

    // Ensure lambdaMult has a default value if not provided
    const lambdaMult = options.lambda ?? 0.5;
    const mmrSelectedIndices: number[] = maximalMarginalRelevance(
      queryEmbedding,
      consistentEmbeddings,
      lambdaMult,
      options.k
    );

    // Filter documents based on MMR-selected indices and map scores
    return mmrSelectedIndices.map((index) => ({
      document: documents[index],
      score: scores[index],
    }));
  }

  public async delete(params: {
    ids?: Buffer[];
    deleteAll?: boolean;
  }): Promise<void> {
    let connection: oracledb.Connection | null = null;
    try {
      connection = await this.getConnection();
      const options = { autoCommit: true };
      if (params.ids && params.ids.length > 0) {
        // Dynamically create placeholders
        const placeholders = params.ids
          .map((_, index) => `:${index + 1}`)
          .join(",");
        // Prepare the query
        const query = `DELETE FROM ${this.tableName} WHERE id IN (${placeholders})`;
        // Execute the query with the IDs as bind parameters
        await connection.execute(query, [...params.ids], options);
      } else if (params.deleteAll) {
        await connection.execute(
          `TRUNCATE TABLE ${this.tableName}`,
          [],
          options
        );
      }
    } catch (error: unknown) {
      handleError(error);
    } finally {
      if (connection) await connection.close();
    }
  }

  static async fromDocuments(
    documents: Document[],
    embeddings: EmbeddingsInterface,
    dbConfig: OracleDBVSArgs
  ): Promise<OracleVS> {
    const { client } = dbConfig;
    if (!client) throw new Error("client parameter is required...");

    try {
      const vss = new OracleVS(embeddings, dbConfig);
      await vss.initialize();

      const texts = documents.map(({ pageContent }) => pageContent);
      const vectors = await embeddings.embedDocuments(texts);

      // Assuming a method exists to handle adding texts and metadata appropriately
      await vss.addVectors(vectors, documents);

      return vss;
    } catch (error: unknown) {
      handleError(error);
    }
  }

  /**
   *
   * @returns Promise that resolves when all connections
   * inside the pool are terminated.
   */
  async end(): Promise<void> {
    if (isPool(this.client)) {
      await this.client?.close();
    }
  }
}
