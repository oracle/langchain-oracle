export const OracleErrorCode = {
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

export type OracleErrorCode =
  (typeof OracleErrorCode)[keyof typeof OracleErrorCode];

type OracleErrorArgs = {
  [OracleErrorCode.VALIDATION_INVALID_INPUT]: [message: string];
  [OracleErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER]: [parameter: string];
  [OracleErrorCode.VALIDATION_INVALID_IDENTIFIER]: [identifier: string];
  [OracleErrorCode.FILTER_INVALID_METADATA_KEY]: [column: string];
  [OracleErrorCode.FILTER_INVALID_VALUE]: [message: string];
  [OracleErrorCode.FILTER_UNSUPPORTED_OPERATOR]: [operator: string];
  [OracleErrorCode.VECTOR_INVALID_CONFIGURATION]: [message: string];
  [OracleErrorCode.VECTOR_INVALID_VALUE]: [message: string];
  [OracleErrorCode.VECTOR_UNSUPPORTED_REPRESENTATION]: [message: string];
  [OracleErrorCode.VECTOR_INVALID_INDEX_PARAMETERS]: [invalidKeys: string[]];
  [OracleErrorCode.STATE_INVALID]: [message: string];
  [OracleErrorCode.QUERY_NO_ROWS_FOUND]: [];
  [OracleErrorCode.SYSTEM_ERROR]: [message: string];
};

export class OracleError extends Error {
  readonly code: OracleErrorCode;

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

function getOracleErrorMessage<K extends OracleErrorCode>(
  code: K,
  ...args: OracleErrorArgs[K]
): string {
  const errorMessageFactories: {
    [Code in OracleErrorCode]: (...innerArgs: OracleErrorArgs[Code]) => string;
  } = {
    [OracleErrorCode.VALIDATION_INVALID_INPUT]: (message) => message,
    [OracleErrorCode.VALIDATION_MISSING_REQUIRED_PARAMETER]: (parameter) =>
      `${parameter} parameter is required...`,
    [OracleErrorCode.VALIDATION_INVALID_IDENTIFIER]: (identifier) =>
      `Identifier name ${identifier} is not valid.`,
    [OracleErrorCode.FILTER_INVALID_METADATA_KEY]: (column) =>
    `Invalid metadata key '${String(column)}'. Only letters, numbers, underscores, nesting via '.', and array wildcards '[*]' are allowed.`,
    [OracleErrorCode.FILTER_INVALID_VALUE]: (message) => message,
    [OracleErrorCode.FILTER_UNSUPPORTED_OPERATOR]: (operator) =>
      `Unsupported operator: ${operator}`,
    [OracleErrorCode.VECTOR_INVALID_CONFIGURATION]: (message) => message,
    [OracleErrorCode.VECTOR_INVALID_VALUE]: (message) => message,
    [OracleErrorCode.VECTOR_UNSUPPORTED_REPRESENTATION]: (message) => message,
    [OracleErrorCode.VECTOR_INVALID_INDEX_PARAMETERS]: (invalidKeys) =>
      `Invalid parameter(s): ${invalidKeys.join(", ")}`,
    [OracleErrorCode.STATE_INVALID]: (message) => message,
    [OracleErrorCode.QUERY_NO_ROWS_FOUND]: () => "No rows found.",
    [OracleErrorCode.SYSTEM_ERROR]: (message) => message,
  };

  return errorMessageFactories[code](...args);
}

export function createOracleErrorFromCode<K extends OracleErrorCode>(
  code: K,
  args: OracleErrorArgs[K],
  cause?: unknown
): OracleError {
  return createOracleError(code, getOracleErrorMessage(code, ...args), cause);
}

export function throwOracleError<K extends OracleErrorCode>(
  code: K,
  ...args: OracleErrorArgs[K]
): never {
  throw createOracleErrorFromCode(code, args);
}
