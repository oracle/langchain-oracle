import oracledb from "oracledb";
import { ErrorCode, throwError } from "./errors.js";

export type OracleDBClient = oracledb.Pool | oracledb.Connection;
export type Client = OracleDBClient | OracleDBClientProvider;

// Allows callers to resolve the OracleDB client lazily, for example when the
// pool or connection is created asynchronously or managed by the caller.
export type OracleDBClientProvider = () => Promise<OracleDBClient>;

export async function withConnection<T>(
  dbSource: OracleDBClient | OracleDBClientProvider,
  fn: (Connection: oracledb.Connection) => Promise<T>
): Promise<T> {
  if (dbSource == null) {
    throwError(ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER, "client");
  }
  const client = typeof dbSource === "function" ? await dbSource() : dbSource;
  const isPool = "getConnection" in client;
  const connection = isPool ? await client.getConnection() : client;

  let result: T;

  try {
    result = await fn(connection);
  } catch (error) {
    if (isPool) {
      try {
        await connection.close();
      } catch (closeError) {
        throw new AggregateError(
          [error, closeError],
          "Operation failed, and connection cleanup failed."
        );
      }
    }

    throw error;
  }

  if (isPool) {
    await connection.close();
  }

  return result;
}

export function quoteIdentifier(identifier: string) {
  const name = identifier.trim();

  const validateRegex = /^(?:"[^"]+"|[^".]+)(?:\.(?:"[^"]+"|[^".]+))*$/;
  if (!validateRegex.test(name)) {
    throwError(ErrorCode.VALIDATION_INVALID_IDENTIFIER, identifier);
  }

  // Extract quoted and unquoted identifier parts.
  const matchRegex = /"([^"]+)"|([^".]+)/g;
  const groups = [];

  for (const match of name.matchAll(matchRegex)) {
    groups.push(match[1] || match[2]);
  }
  const quotedParts = groups.map((g) => `"${g}"`);
  return quotedParts.join(".");
}

function identifierParts(name: string): { owner: string | null; name: string } {
  const parts = Array.from(name.matchAll(/"([^"]+)"/g), (match) => match[1]);
  if (parts.length < 1 || parts.length > 2) {
    throwError(
      ErrorCode.VALIDATION_INVALID_INPUT,
      "Expected an optionally schema-qualified identifier"
    );
  }
  return {
    owner: parts.length === 2 ? parts[0] : null,
    name: parts[parts.length - 1],
  };
}

export async function indexExist(
  client: Client,
  indexName: string,
  tableName: string
): Promise<boolean> {
  const index = identifierParts(indexName);
  const table = identifierParts(tableName);

  return withConnection(client, async (connection) => {
    const result = await connection.execute<[string]>(
      `SELECT index_name
       FROM all_indexes
       WHERE index_name = :idx_name
         AND table_name = :table_name
         AND owner = COALESCE(
           :owner, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
         )
         AND table_owner = COALESCE(
           :table_owner, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
         )`,
      {
        idx_name: index.name,
        table_name: table.name,
        owner: index.owner,
        table_owner: table.owner,
      },
      { outFormat: oracledb.OUT_FORMAT_ARRAY }
    );

    return Boolean(result.rows?.length);
  });
}
