export const OracleErrorCode = {
  NO_ROWS_FOUND: "NO_ROWS_FOUND",
  RUNTIME_ERROR: "RUNTIME_ERROR",
  VALIDATION_ERROR: "VALIDATION_ERROR",
  UNEXPECTED_ERROR: "UNEXPECTED_ERROR",
  UNKNOWN_ERROR: "UNKNOWN_ERROR",
  INVALID_METADATA_KEY: "INVALID_METADATA_KEY",
  INVALID_FILTER_VALUE: "INVALID_FILTER_VALUE",
  UNSUPPORTED_FILTER_OPERATOR: "UNSUPPORTED_FILTER_OPERATOR",
  INVALID_IDENTIFIER: "INVALID_IDENTIFIER",
  INVALID_VECTOR_CONFIGURATION: "INVALID_VECTOR_CONFIGURATION",
  INVALID_VECTOR_VALUE: "INVALID_VECTOR_VALUE",
  INVALID_INDEX_PARAMETERS: "INVALID_INDEX_PARAMETERS",
  INVALID_STATE: "INVALID_STATE",
  UNSUPPORTED_VECTOR_REPRESENTATION: "UNSUPPORTED_VECTOR_REPRESENTATION",
  INVALID_INPUT: "INVALID_INPUT",
  MISSING_REQUIRED_PARAMETER: "MISSING_REQUIRED_PARAMETER",
  INVALID_PREFERENCES: "INVALID_PREFERENCES",
  INVALID_SQL_IDENTIFIER: "INVALID_SQL_IDENTIFIER",
} as const;

export type OracleErrorCode =
  (typeof OracleErrorCode)[keyof typeof OracleErrorCode];

export class OracleError extends Error {
  code: OracleErrorCode;

  override cause?: unknown;

  constructor(code: OracleErrorCode, message: string, cause?: unknown) {
    super(message);
    this.name = "OracleError";
    this.code = code;
    this.cause = cause;
  }
}

export function createOracleError(
  code: OracleErrorCode,
  message: string,
  cause?: unknown
): OracleError {
  return new OracleError(code, message, cause);
}

export function throwOracleError(
  code: OracleErrorCode,
  message: string,
  cause?: unknown
): never {
  throw createOracleError(code, message, cause);
}
