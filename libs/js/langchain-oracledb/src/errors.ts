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

const errorMessageFactories: Record<OracleErrorCode, (...args: unknown[]) => string> = {
  [OracleErrorCode.NO_ROWS_FOUND]: () => "No rows found.",
  [OracleErrorCode.RUNTIME_ERROR]: () =>
    "Database operation failed due to a runtime error.",
  [OracleErrorCode.VALIDATION_ERROR]: () =>
    "Operation failed due to a validation error.",
  [OracleErrorCode.UNEXPECTED_ERROR]: (error) =>
    `An unexpected error occurred during the operation. ${String(error)}`,
  [OracleErrorCode.UNKNOWN_ERROR]: (error) =>
    `An unknown and unexpected error occurred. ${String(error)}`,
  [OracleErrorCode.INVALID_METADATA_KEY]: (column) =>
    `Invalid metadata key '${String(column)}'. Only letters, numbers, underscores, nesting via '.', and array wildcards '[*]' are allowed.`,
  [OracleErrorCode.INVALID_FILTER_VALUE]: (message) => String(message),
  [OracleErrorCode.UNSUPPORTED_FILTER_OPERATOR]: (operator) =>
    `Unsupported operator: ${String(operator)}`,
  [OracleErrorCode.INVALID_IDENTIFIER]: (identifier) =>
    `Identifier name ${String(identifier)} is not valid.`,
  [OracleErrorCode.INVALID_VECTOR_CONFIGURATION]: (message) => String(message),
  [OracleErrorCode.INVALID_VECTOR_VALUE]: (message) => String(message),
  [OracleErrorCode.INVALID_INDEX_PARAMETERS]: (invalidKeys) =>
    `Invalid parameter(s): ${Array.isArray(invalidKeys) ? invalidKeys.join(", ") : String(invalidKeys)}`,
  [OracleErrorCode.INVALID_STATE]: (message) => String(message),
  [OracleErrorCode.UNSUPPORTED_VECTOR_REPRESENTATION]: (message) =>
    String(message),
  [OracleErrorCode.INVALID_INPUT]: (message) => String(message),
  [OracleErrorCode.MISSING_REQUIRED_PARAMETER]: (parameter) =>
    `${String(parameter)} parameter is required...`,
  [OracleErrorCode.INVALID_PREFERENCES]: (message) => String(message),
  [OracleErrorCode.INVALID_SQL_IDENTIFIER]: () =>
    "Invalid owner, table, or column name",
};

export function throwOracleError(
  code: OracleErrorCode,
  ...args: unknown[]
): never {
  throw createOracleError(code, errorMessageFactories[code](...args));
}
