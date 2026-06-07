import { describe, expect, test } from "vitest";

import { generateWhereClause } from "../vectorstores.js";

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

  test("rejects metadata keys containing SQL injection payloads", () => {
    expect(() =>
      generateWhereClause({ ["author') OR 1=1 --"]: "alice" }, [])
    ).toThrow(/Invalid metadata key/);
  });
});
