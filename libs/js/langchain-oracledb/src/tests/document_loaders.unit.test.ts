import oracledb from "oracledb";
import { afterEach, describe, expect, test, vi } from "vitest";
import {
  OracleAutonomousDatabaseLoader,
  type OracleADBLoaderParams,
} from "../document_loaders.js";

function makeLoader(
  overrides: Partial<OracleADBLoaderParams> = {}
): OracleAutonomousDatabaseLoader {
  return new OracleAutonomousDatabaseLoader({
    query: "SELECT * FROM T",
    user: "test_user",
    password: "test_password",
    connectString: "test_db",
    ...overrides,
  });
}

function mockDatabase(rows: Record<string, unknown>[] = []) {
  const execute = vi.fn().mockResolvedValue({ rows });
  const close = vi.fn().mockResolvedValue(undefined);
  const connection = {
    execute,
    close,
  } as unknown as oracledb.Connection;
  const getConnection = vi
    .spyOn(oracledb, "getConnection")
    .mockImplementation(() => connection);

  return { connection, execute, close, getConnection };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("OracleAutonomousDatabaseLoader", () => {
  describe("constructor", () => {
    test("stores required connection options", () => {
      const loader = makeLoader({
        query: "SELECT 1 FROM DUAL",
        user: "hr",
        password: "secret",
        connectString: "adb_high",
      });

      expect(loader.query).toBe("SELECT 1 FROM DUAL");
      expect(loader.user).toBe("hr");
      expect(loader.password).toBe("secret");
      expect(loader.connectString).toBe("adb_high");
    });

    test("leaves optional settings undefined by default", () => {
      const loader = makeLoader();

      expect(loader.schema).toBeUndefined();
      expect(loader.configDir).toBeUndefined();
      expect(loader.walletLocation).toBeUndefined();
      expect(loader.walletPassword).toBeUndefined();
      expect(loader.metadata).toBeUndefined();
      expect(loader.parameter).toBeUndefined();
    });

    test("stores optional settings", () => {
      const parameter = { id: 42 };
      const loader = makeLoader({
        schema: "HR",
        configDir: "/etc/oracle/config",
        walletLocation: "/wallet",
        walletPassword: "wallet-password",
        metadata: ["ID", "NAME"],
        parameter,
      });

      expect(loader.schema).toBe("HR");
      expect(loader.configDir).toBe("/etc/oracle/config");
      expect(loader.walletLocation).toBe("/wallet");
      expect(loader.walletPassword).toBe("wallet-password");
      expect(loader.metadata).toEqual(["ID", "NAME"]);
      expect(loader.parameter).toEqual(parameter);
    });
  });

  describe("connection handling", () => {
    test("forwards optional connection settings and applies the schema", async () => {
      const loader = makeLoader({
        schema: "HR",
        configDir: "/etc/oracle/config",
        walletLocation: "/wallet",
        walletPassword: "wallet-password",
      });
      const { connection, getConnection } = mockDatabase();

      await loader._fetchRows();

      expect(getConnection).toHaveBeenCalledWith({
        user: "test_user",
        password: "test_password",
        connectString: "test_db",
        configDir: "/etc/oracle/config",
        walletLocation: "/wallet",
        walletPassword: "wallet-password",
      });
      expect(connection.currentSchema).toBe("HR");
    });

    test("omits optional connection settings when incomplete", async () => {
      const loader = makeLoader({ walletLocation: "/wallet" });
      const { connection, getConnection } = mockDatabase();

      await loader._fetchRows();

      const connectOptions = getConnection.mock.calls[0][0];
      expect(connectOptions).not.toHaveProperty("configDir");
      expect(connectOptions).not.toHaveProperty("walletLocation");
      expect(connectOptions).not.toHaveProperty("walletPassword");
      expect(connection).not.toHaveProperty("currentSchema");
    });

    test("closes the connection after a successful query", async () => {
      const loader = makeLoader();
      const { close } = mockDatabase();

      await loader._fetchRows();

      expect(close).toHaveBeenCalledOnce();
    });
  });

  describe("query execution", () => {
    test("passes named bind parameters to execute", async () => {
      const loader = makeLoader({
        query: "SELECT * FROM T WHERE ID = :id",
        parameter: { id: 1 },
      });
      const { execute } = mockDatabase();

      await loader._fetchRows();

      expect(execute).toHaveBeenCalledWith(
        "SELECT * FROM T WHERE ID = :id",
        { id: 1 },
        { outFormat: oracledb.OUT_FORMAT_OBJECT }
      );
    });

    test("passes positional bind parameters to execute", async () => {
      const loader = makeLoader({
        query: "SELECT * FROM T WHERE ID = :1",
        parameter: [42],
      });
      const { execute } = mockDatabase();

      await loader._fetchRows();

      expect(execute).toHaveBeenCalledWith(
        "SELECT * FROM T WHERE ID = :1",
        [42],
        { outFormat: oracledb.OUT_FORMAT_OBJECT }
      );
    });

    test("resolves LOB values with getData", async () => {
      const loader = makeLoader();
      const getData = vi.fn().mockResolvedValue("LOB content");
      mockDatabase([{ ID: 1, TEXT: { getData } }]);

      await expect(loader._fetchRows()).resolves.toEqual([
        { ID: 1, TEXT: "LOB content" },
      ]);
      expect(getData).toHaveBeenCalledOnce();
    });
  });

  describe("document creation", () => {
    test("serializes each row and selects the configured metadata", async () => {
      const loader = makeLoader({ metadata: ["ID", "CATEGORY"] });
      mockDatabase([{ ID: 7, TEXT: "hello", CATEGORY: "docs", IGNORED: true }]);

      const documents = await loader.load();

      expect(documents).toHaveLength(1);
      expect(documents[0].pageContent).toBe(
        '{"ID":7,"TEXT":"hello","CATEGORY":"docs","IGNORED":true}'
      );
      expect(documents[0].metadata).toEqual({ ID: 7, CATEGORY: "docs" });
    });
  });
});
