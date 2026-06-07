import { describe, expect, test } from "vitest";

import {
  OracleError,
  OracleErrorCode,
  generateWhereClause,
} from "../vectorstores.js";

describe("generateWhereClause", () => {
  test("binds scalar values instead of interpolating them into SQL", () => {
    const bindValues: unknown[] = [];

    const clause = generateWhereClause(
      { author: "Robert'); DROP TABLE docs; --" },
      bindValues
    );

    expect(clause).toContain("JSON_EXISTS");
    expect(clause).not.toContain("DROP TABLE");
    expect(bindValues).toEqual(["Robert'); DROP TABLE docs; --"]);
  });

  test("rejects metadata keys containing injection payloads", () => {
    expect(() =>
      generateWhereClause({ ["author') OR 1=1 --"]: "alice" }, [])
    ).toThrow(/Invalid metadata key/);
  });
});

describe("OracleError", () => {
  test("preserves the no rows found message while exposing a stable code", () => {
    const error = new OracleError(
      OracleErrorCode.NO_ROWS_FOUND,
      "No rows found."
    );

    expect(error.message).toBe("No rows found.");
    expect(error.code).toBe(OracleErrorCode.NO_ROWS_FOUND);
    expect(error.name).toBe("OracleError");
  });
});
