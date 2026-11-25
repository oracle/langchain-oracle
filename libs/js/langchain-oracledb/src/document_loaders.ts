/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { Document } from "@langchain/core/documents";
import fs from "node:fs";
import path from "node:path";
import oracledb from "oracledb";
import * as htmlparser2 from "htmlparser2";
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";
import { ErrorCode, handleError, throwError } from "./errors.js";
import type { BindParameters } from "oracledb";

function* listDir(dir: string): Generator<string> {
  const files = fs.readdirSync(dir, { withFileTypes: true });

  for (const file of files) {
    if (file.isDirectory()) {
      yield* listDir(path.join(dir, file.name));
    } else {
      yield path.join(dir, file.name);
    }
  }
}

/**
 * Load documents from a file, a directory, or a table
 * If the document isn't a plain text file such as a PDF,
 * a plain text version will be extracted using utl_to_text
 * @example
 * ```typescript
 * const loader = new OracleDocLoader(conn, params);
 * const docs = await loader.load();
 * ```
 */
export class OracleDocLoader extends BaseDocumentLoader {
  protected conn: oracledb.Connection;

  protected pref: Record<string, any>;

  constructor(conn: oracledb.Connection, pref: Record<string, any>) {
    super();
    this.conn = conn;
    this.pref = pref;
  }

  /**
   * A method that loads the text file or blob and returns a promise that
   * resolves to an array of `Document` instances. It reads the text from
   * the file or blob using the `readFile` function from the
   * `node:fs/promises` module or the `text()` method of the blob. It then
   * parses the text using the `parse()` method and creates a `Document`
   * instance for each parsed page. The metadata includes the source of the
   * text (file path or blob) and, if there are multiple pages, the line
   * number of each page.
   * @returns A promise that resolves to an array of `Document` instances.
   */
  async load(): Promise<Document[]> {
    const docs: Document[] = [];

    if ("file" in this.pref) {
      const doc = await this._loadFromFile(this.pref.file);
      if (doc != null) {
        docs.push(doc);
      }
    } else if ("dir" in this.pref) {
      for (const file of listDir(this.pref.dir)) {
        const doc = await this._loadFromFile(file);
        if (doc != null) {
          docs.push(doc);
        }
      }
    } else if ("tablename" in this.pref) {
      if (!("owner" in this.pref) || !("colname" in this.pref)) {
        throwError(
          ErrorCode.VALIDATION_INVALID_INPUT,
          "Invalid preferences: missing owner or colname"
        );
      }
      docs.push(
        ...(await this._loadFromTable(
          this.pref.owner,
          this.pref.tablename,
          this.pref.colname
        ))
      );
    } else {
      throwError(
        ErrorCode.VALIDATION_INVALID_INPUT,
        "Invalid preferences: missing file, dir, or tablename"
      );
    }

    return docs;
  }

  // load from file
  private async _loadFromFile(filename: string): Promise<Document | null> {
    let doc = null;

    // don't specify an encoding to use binary
    const data = fs.readFileSync(filename);

    const result = await this.conn.execute(
      <string>`select dbms_vector_chain.utl_to_text(:content, :pref) text,\
                        dbms_vector_chain.utl_to_text(:content, json('{"plaintext": "false"}')) metadata from dual`,
      <oracledb.BindParameters>{
        content: { val: data, dir: oracledb.BIND_IN, type: oracledb.BLOB },
        pref: { val: this.pref, type: oracledb.DB_TYPE_JSON },
      },
      <oracledb.ExecuteOptions>(<unknown>{
        resultSet: true, // return a ResultSet (default is false)
        fetchInfo: {
          TEXT: { type: oracledb.STRING },
          METADATA: { type: oracledb.STRING },
        },
      })
    );
    const resultSet = result.resultSet;
    try {
      if (resultSet) {
        for await (const row of resultSet as unknown as Array<any>) {
          const [plain_text, metadata] = await this._extract(row);
          doc = new Document({
            pageContent: <string>plain_text,
            metadata: <Record<string, any>>metadata,
          });
        }
      }
    } finally {
      if (resultSet) {
        await resultSet.close();
      }
    }

    return doc;
  }

  // load from table
  private async _loadFromTable(
    owner: string,
    table: string,
    col: string
  ): Promise<Document[]> {
    const docs = [];

    // Check if names are invalid
    const qn = `${owner}.${table}`;
    try {
      const sql = `select sys.dbms_assert.simple_sql_name(:col),\
                            sys.dbms_assert.qualified_sql_name(:qn) from dual`;
      const binds = [col, qn];
      await this.conn.execute(sql, binds);
    } catch {
      throwError(
        ErrorCode.VALIDATION_INVALID_IDENTIFIER,
        `${owner}.${table}.${col}`
      );
    }

    const result = await this.conn.execute(
      <string>(
        `select dbms_vector_chain.utl_to_text(t.${this.pref.colname}, :pref) text,\
                  dbms_vector_chain.utl_to_text(t.${this.pref.colname}, json('{"plaintext": "false"}')) metadata\
                  from ${owner}.${table} t`
      ),
      <oracledb.BindParameters>{
        pref: { val: this.pref, type: oracledb.DB_TYPE_JSON },
      },
      <oracledb.ExecuteOptions>(<unknown>{
        resultSet: true, // return a ResultSet (default is false)
        fetchInfo: {
          TEXT: { type: oracledb.STRING },
          METADATA: { type: oracledb.STRING },
        },
      })
    );

    const resultSet = result.resultSet;
    try {
      if (resultSet) {
        for await (const row of resultSet as unknown as Array<any>) {
          const [plain_text, metadata] = await this._extract(row);
          docs.push(
            new Document({
              pageContent: <string>plain_text,
              metadata: <Record<string, any>>metadata,
            })
          );
        }
      }
    } finally {
      if (resultSet) {
        await resultSet.close();
      }
    }

    return docs;
  }

  // extract plain text and metadata from a row
  private async _extract(row: any): Promise<[string, Record<string, string>]> {
    const [text, htmlMetadata] = row;
    const metadata: Record<string, string> = {};

    const parser = new htmlparser2.Parser({
      onopentag(name, attrs) {
        if (name === "meta" && attrs.name) {
          metadata[attrs.name] = attrs.content;
        }
      },
    });

    parser.write(htmlMetadata);
    parser.end();

    return [text, metadata];
  }
}

export interface OracleADBLoaderParams {
  query: string;
  user: string;
  password: string;

  schema?: string;
  connectString?: string;
  configDir?: string;
  walletLocation?: string;
  walletPassword?: string;
  metadata?: string[];
  parameter?: BindParameters;
}

export class OracleAutonomousDatabaseLoader extends BaseDocumentLoader {
  readonly query: string;
  readonly user: string;
  readonly password: string;

  readonly schema?: string;
  readonly connectString?: string;
  readonly configDir?: string;
  readonly walletLocation?: string;
  readonly walletPassword?: string;
  readonly metadata?: string[];
  readonly parameter?: BindParameters;

  constructor(options: OracleADBLoaderParams) {
    super();

    this.query = options.query;
    this.user = options.user;
    this.password = options.password;
    this.schema = options.schema;
    this.configDir = options.configDir;
    this.walletLocation = options.walletLocation;
    this.walletPassword = options.walletPassword;
    this.metadata = options.metadata;
    this.parameter = options.parameter;
    this.connectString = options.connectString;
  }

  async _fetchRows(): Promise<Record<string, unknown>[]> {
    let connection: oracledb.Connection | undefined = undefined;

    const connectParams: oracledb.ConnectionAttributes = {
      user: this.user,
      password: this.password,
      connectString: this.connectString,
    };

    if (this.configDir) {
      connectParams.configDir = this.configDir;
    }

    if (this.walletLocation && this.walletPassword) {
      connectParams.walletLocation = this.walletLocation;
      connectParams.walletPassword = this.walletPassword;
    }

    try {
      connection = await oracledb.getConnection(connectParams);
      let data: Record<string, unknown>[] = [];
      if (this.schema) {
        connection.currentSchema = this.schema;
      }

      const result = await connection.execute(
        this.query,
        this.parameter ?? [],
        { outFormat: oracledb.OUT_FORMAT_OBJECT }
      );

      data = (result.rows ?? []) as Record<string, unknown>[];

      for (const row of data) {
        for (const [column, value] of Object.entries(row)) {
          if (
            typeof value === "object" &&
            value !== null &&
            "getData" in value &&
            typeof value.getData === "function"
          ) {
            row[column] = await value.getData();
          }
        }
      }

      return data;
    } catch (error) {
      return handleError(error);
    } finally {
      if (connection) {
        await connection.close();
      }
    }
  }

  async load(): Promise<Document[]> {
    const data = await this._fetchRows();
    const metadataColumns = this.metadata ? this.metadata : [];
    const metadataColumnSet = new Set(metadataColumns);

    const documents = data.map(
      (row: Record<string, unknown>) =>
        new Document({
          pageContent: JSON.stringify(row),
          metadata: Object.fromEntries(
            Object.entries(row).filter(([key]) => metadataColumnSet.has(key))
          ),
        })
    );

    return documents;
  }
}
