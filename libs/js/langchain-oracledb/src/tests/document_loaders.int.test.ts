/* eslint-disable no-process-env */
import { describe, test, expect } from "vitest";
import * as url from "node:url";
import * as path from "node:path";
import oracledb from "oracledb";
import { Document } from "@langchain/core/documents";
import { OracleDocLoader, OracleAutonomousDatabaseLoader } from "../index.js";

describe("OracleDocLoader", () => {
  test("Test loading PDF from file", async () => {
    const filePath = path.resolve(
      path.dirname(url.fileURLToPath(import.meta.url)),
      "./example_data/1706.03762.pdf"
    );
    const pref = { file: filePath };
    const connection = await oracledb.getConnection({
      user: process.env.ORACLE_USERNAME,
      password: process.env.ORACLE_PASSWORD,
      connectString: process.env.ORACLE_DSN,
    });
    const loader = new OracleDocLoader(connection, pref);
    const docs = await loader.load();
    await connection.close();

    expect(docs.length).toBe(1);
    expect(docs[0].pageContent).toContain("Transformer");
  });

  test("Test loading from directory", async () => {
    const filePath = path.resolve(
      path.dirname(url.fileURLToPath(import.meta.url)),
      "./example_data"
    );
    const connection = await oracledb.getConnection({
      user: process.env.ORACLE_USERNAME,
      password: process.env.ORACLE_PASSWORD,
      connectString: process.env.ORACLE_DSN,
    });
    const pref = {
      dir: filePath,
    };
    const loader = new OracleDocLoader(connection, pref);
    const docs = await loader.load();
    await connection.close();

    expect(docs.length).toBeGreaterThanOrEqual(1);
  });

  test("Test loading from table", async () => {
    const tableName = "LC_DOC_LOADER_TEST";
    const connection = await oracledb.getConnection({
      user: process.env.ORACLE_USERNAME,
      password: process.env.ORACLE_PASSWORD,
      connectString: process.env.ORACLE_DSN,
    });

    try {
      await connection.execute(`DROP TABLE IF EXISTS ${tableName} PURGE`);
      await connection.execute(`CREATE TABLE ${tableName} (text CLOB)`);
      await connection.execute(
        `INSERT INTO ${tableName} (text) VALUES (:content)`,
        {
          content: "Oracle document loader table integration test content.",
        },
        { autoCommit: true }
      );

      const pref = {
        owner: process.env.ORACLE_USERNAME,
        tablename: tableName,
        colname: "TEXT",
      };
      const loader = new OracleDocLoader(connection, pref);
      const docs = await loader.load();

      expect(docs.length).toBeGreaterThanOrEqual(1);
      expect(docs[0].pageContent).toContain("Oracle document loader");
    } finally {
      await connection.execute(`DROP TABLE IF EXISTS ${tableName} PURGE`);
      await connection.close();
    }
  });
});

describe("OracleAutonomousDatabaseLoader", () => {
  test("creates one document for each query row", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query: "SELECT level AS n FROM DUAL CONNECT BY level <= 5",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
    });

    const docs = await loader.load();
    expect(docs).toHaveLength(5);
    // eslint-disable-next-line no-instanceof/no-instanceof
    expect(docs.every((doc) => doc instanceof Document)).toBe(true);
  });

  test("serializes all row columns into pageContent", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query: "SELECT 'hello' AS greeting, 'world' AS target FROM DUAL",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
    });

    const docs = await loader.load();
    expect(docs[0].pageContent).toContain("hello");
    expect(docs[0].pageContent).toContain("world");
  });

  test("materializes CLOB columns into pageContent", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query:
        "SELECT TO_CLOB('CLOB integration test content') AS content FROM DUAL",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
    });

    const docs = await loader.load();

    expect(docs).toHaveLength(1);
    expect(JSON.parse(docs[0].pageContent)).toEqual({
      CONTENT: "CLOB integration test content",
    });
  });

  test("includes only configured columns in document metadata", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query: "SELECT 1 AS id, 'Alice' AS name, 'ENG' AS dept FROM DUAL",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
      metadata: ["ID", "DEPT"],
    });

    const docs = await loader.load();
    expect(docs[0].metadata).toEqual({ ID: 1, DEPT: "ENG" });
    expect("NAME" in docs[0].metadata).toBe(false);
  });

  test("returns no documents when the query returns no rows", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query: "SELECT 1 AS n FROM DUAL WHERE 1=0",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
    });

    expect(await loader.load()).toEqual([]);
  });

  test("returns equivalent documents across repeated load", async () => {
    const loader = new OracleAutonomousDatabaseLoader({
      query: "SELECT 42 AS val FROM DUAL",
      user: process.env.ORACLE_USERNAME!,
      password: process.env.ORACLE_PASSWORD!,
      connectString: process.env.ORACLE_DSN,
    });

    const docs1 = await loader.load();
    const docs2 = await loader.load();

    expect(docs1.length).toEqual(docs2.length);
    expect(docs1).toEqual(docs2);
  });
});
