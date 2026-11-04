import oracledb from "oracledb";
import { ErrorCode, throwError } from "./errors.js";

export type OracleDBClient = oracledb.Pool | oracledb.Connection;

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
