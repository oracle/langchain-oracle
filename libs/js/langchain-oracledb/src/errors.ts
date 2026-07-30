export const ErrorCode = {
  VALIDATION_INVALID_INPUT: "VALIDATION_INVALID_INPUT",
  VALIDATION_MISSING_REQUIRED_PARAMETER:
    "VALIDATION_MISSING_REQUIRED_PARAMETER",
  VALIDATION_INVALID_IDENTIFIER: "VALIDATION_INVALID_IDENTIFIER",
  FILTER_INVALID_METADATA_KEY: "FILTER_INVALID_METADATA_KEY",
  FILTER_INVALID_VALUE: "FILTER_INVALID_VALUE",
  FILTER_UNSUPPORTED_OPERATOR: "FILTER_UNSUPPORTED_OPERATOR",
  VECTOR_INVALID_CONFIGURATION: "VECTOR_INVALID_CONFIGURATION",
  VECTOR_INVALID_VALUE: "VECTOR_INVALID_VALUE",
  VECTOR_UNSUPPORTED_REPRESENTATION: "VECTOR_UNSUPPORTED_REPRESENTATION",
  VECTOR_INVALID_INDEX_PARAMETERS: "VECTOR_INVALID_INDEX_PARAMETERS",
  STATE_INVALID: "STATE_INVALID",
  QUERY_NO_ROWS_FOUND: "QUERY_NO_ROWS_FOUND",
  SYSTEM_ERROR: "SYSTEM_ERROR",
} as const;

export type ErrorCode =
  (typeof ErrorCode)[keyof typeof ErrorCode];

const LANGCHAIN_ORACLE_ERROR_BRAND = Symbol.for(
  "@oracle/langchain-oracledb/LangChainOracleError"
);

type ErrorArgs = {
  [ErrorCode.VALIDATION_INVALID_INPUT]: [message: string];
  [ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER]: [parameter: string];
  [ErrorCode.VALIDATION_INVALID_IDENTIFIER]: [identifier: string];
  [ErrorCode.FILTER_INVALID_METADATA_KEY]: [column: string];
  [ErrorCode.FILTER_INVALID_VALUE]: [message: string];
  [ErrorCode.FILTER_UNSUPPORTED_OPERATOR]: [operator: string];
  [ErrorCode.VECTOR_INVALID_CONFIGURATION]: [message: string];
  [ErrorCode.VECTOR_INVALID_VALUE]: [message: string];
  [ErrorCode.VECTOR_UNSUPPORTED_REPRESENTATION]: [message: string];
  [ErrorCode.VECTOR_INVALID_INDEX_PARAMETERS]: [invalidKeys: string[]];
  [ErrorCode.STATE_INVALID]: [message: string];
  [ErrorCode.QUERY_NO_ROWS_FOUND]: [];
  [ErrorCode.SYSTEM_ERROR]: [message: string];
};

/**
 * A specialized exception representing domain-specific failures within the
 * LangChain Oracle integration. Extends the native JavaScript `Error` to preserve
 * stack traces. Lower-level driver errors are safely isolated inside the `cause`
 * property to protect against shifting node-oracledb definitions.
 */
export class LangChainOracleError extends Error {
  readonly code: ErrorCode;

  readonly cause?: unknown;

  readonly [LANGCHAIN_ORACLE_ERROR_BRAND] = true;

  constructor(code: ErrorCode, message: string, cause?: unknown) {
    super(message);

    // Hardcode the class name so logs show "LangChainOracleError: ..."
    this.name = "LangChainOracleError";
    this.code = code;

    // Save the raw node-oracledb error (or validation error) for root-cause analysis
    this.cause = cause;
  }
}

export function createError(
  code: ErrorCode,
  message: string,
  cause?: unknown
): LangChainOracleError {
  return new LangChainOracleError(code, message, cause);
}

// Identifies LangChainOracleError instances created by this package, not generic
// node-oracledb or database errors such as NJS- or ORA- codes.
export function isLangChainOracleError(
  error: unknown
): error is LangChainOracleError {
  return (
    typeof error === "object" &&
    error !== null &&
    LANGCHAIN_ORACLE_ERROR_BRAND in error
  );
}

const errorMessageFactories: {
  [Code in ErrorCode]: (...innerArgs: ErrorArgs[Code]) => string;
} = {
  [ErrorCode.VALIDATION_INVALID_INPUT]: (message) => message,
  [ErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER]: (parameter) =>
    `${parameter} parameter is required...`,
  [ErrorCode.VALIDATION_INVALID_IDENTIFIER]: (identifier) =>
    `Identifier name ${identifier} is not valid.`,
  [ErrorCode.FILTER_INVALID_METADATA_KEY]: (column) =>
    `Invalid metadata key '${String(column)}'. Only letters, numbers, underscores, nesting via '.', and array wildcards '[*]' are allowed.`,
  [ErrorCode.FILTER_INVALID_VALUE]: (message) => message,
  [ErrorCode.FILTER_UNSUPPORTED_OPERATOR]: (operator) =>
    `Unsupported operator: ${operator}`,
  [ErrorCode.VECTOR_INVALID_CONFIGURATION]: (message) => message,
  [ErrorCode.VECTOR_INVALID_VALUE]: (message) => message,
  [ErrorCode.VECTOR_UNSUPPORTED_REPRESENTATION]: (message) => message,
  [ErrorCode.VECTOR_INVALID_INDEX_PARAMETERS]: (invalidKeys) =>
    `Invalid parameter(s): ${invalidKeys.join(", ")}`,
  [ErrorCode.STATE_INVALID]: (message) => message,
  [ErrorCode.QUERY_NO_ROWS_FOUND]: () => "No rows found.",
  [ErrorCode.SYSTEM_ERROR]: (message) => message,
};

function getErrorMessage<K extends ErrorCode>(
  code: K,
  ...args: ErrorArgs[K]
): string {
  return errorMessageFactories[code](...args);
}

export function createErrorFromCode<K extends ErrorCode>(
  code: K,
  ...args: ErrorArgs[K]
): LangChainOracleError {
  return createError(code, getErrorMessage(code, ...args));
}

export function createErrorFromCodeWithCause<K extends ErrorCode>(
  code: K,
  cause: unknown,
  ...args: ErrorArgs[K]
): LangChainOracleError {
  return createError(code, getErrorMessage(code, ...args), cause);
}

export function throwError<K extends ErrorCode>(
  code: K,
  ...args: ErrorArgs[K]
): never {
  throw createErrorFromCode(code, ...args);
}
