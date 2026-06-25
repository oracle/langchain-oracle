/* eslint-disable no-process-env */
import { test, expect } from "vitest";
import * as url from "node:url";
import * as path from "node:path";
import oracledb from "oracledb";
import { OracleDocLoader } from "../index.js";

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
