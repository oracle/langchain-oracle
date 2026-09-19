import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import oracledb from "oracledb";

import { Document } from "@langchain/core/documents";
import { PromptTemplate } from "@langchain/core/prompts";

import {
  OracleDocLoader,
  OracleTextSplitter,
  OracleEmbeddings,
  DistanceStrategy,
  createIndex,
  dropTablePurge,
  OracleVS,
} from "@oracle/langchain-oracledb";
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

type OciChatConfig = {
  compartmentId: string;
  modelId: string;
  endpoint: string;
  configFile: string;
  profile: string;
  authType: OciGenAiNewClientAuthType;
};

type RagConfig = {
  tableName: string;
  documentsFolder: string;
  embeddingModel: string;
  resetVectorStore: boolean;
};

function resolvePath(filePath: string): string {
  // Expand only "~" or "~/" rather than paths such as "~my-folder".
  if (/^~(?=$|\/|\\)/.test(filePath)) {
    return path.join(os.homedir(), filePath.slice(1));
  }

  return path.resolve(filePath);
}

/**
 * Map OCI_AUTH_TYPE to the langchain-oci auth type. "ConfigFile" (default) is
 * API-key auth from the config file; "Session" uses the profile's
 * security_token_file (e.g. a profile created by `oci session authenticate`).
 */
function resolveAuthType(value: string | undefined): OciGenAiNewClientAuthType {
  const normalized = (value ?? "ConfigFile").trim().toLowerCase();
  if (normalized === "configfile" || normalized === "config_file") {
    return OciGenAiNewClientAuthType.ConfigFile;
  }
  if (normalized === "session" || normalized === "security_token") {
    return OciGenAiNewClientAuthType.Session;
  }
  throw new Error(
    `Unsupported OCI_AUTH_TYPE "${value}". Use "ConfigFile" or "Session".`,
  );
}

function getOciClient(config: OciChatConfig): OciGenAiGenericChat {
  return new OciGenAiGenericChat({
    compartmentId: config.compartmentId,
    onDemandModelId: config.modelId,
    newClientParams: {
      authType: config.authType,
      serviceEndpoint: config.endpoint,
      authParams: {
        clientConfigFilePath: resolvePath(config.configFile),
        clientProfile: config.profile,
      },
    },
  });
}

async function chatWithOci(
  client: OciGenAiGenericChat,
  prompt: string,
): Promise<string> {
  const response = await client.invoke(prompt, {
    requestParams: {
      temperature: 0.2,
      topP: 0.9,
      maxTokens: Number(process.env.OCI_MAX_TOKENS ?? 1000),
    },
  });

  if (typeof response.content !== "string" || !response.content.trim()) {
    throw new Error("OCI returned an empty chat response");
  }

  return response.content.trim();
}

async function ingestDocuments(
  conn: oracledb.Connection,
  pool: oracledb.Pool,
  config: RagConfig,
): Promise<OracleVS> {
  console.log("\n-------------------------------------------------------");
  console.log("DOCUMENT INGESTION");
  console.log("-------------------------------------------------------");

  const splitter = new OracleTextSplitter(conn, {
    by: "words",
    max: 200,
    overlap: 20,
    normalize: "all",
    split: "recursively",
  });

  const embedder = new OracleEmbeddings(conn, {
    provider: "database",
    model: config.embeddingModel,
  });

  console.log(`📄 Loading documents from: ${config.documentsFolder}`);

  // Load each file individually so its real filename can be attached as
  // metadata. OracleDocLoader's directory mode extracts text inside the
  // database, which never sees the file path, so it cannot provide one.
  const files = fs
    .readdirSync(config.documentsFolder, {
      recursive: true,
      withFileTypes: true,
    })
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(entry.parentPath, entry.name));

  const rawDocs: Document[] = [];

  for (const file of files) {
    const loader = new OracleDocLoader(conn, { file });
    const [doc] = await loader.load();

    if (doc) {
      doc.metadata.source = file;
      rawDocs.push(doc);
    }
  }

  if (rawDocs.length === 0) {
    throw new Error(
      `No documents were found in DOCUMENTS_FOLDER: ${config.documentsFolder}`,
    );
  }

  const chunks: Document[] = [];

  for (const doc of rawDocs) {
    const textParts = await splitter.splitText(doc.pageContent);

    const sourcePath = String(doc.metadata.source ?? "doc");
    const fileName = sourcePath.split(/[\\/]/).pop() || "doc";

    textParts
      .map((part) => part.trim())
      .filter(Boolean)
      .forEach((part, idx) => {
        const chunkId = idx + 1;

        chunks.push(
          new Document({
            pageContent: part,
            metadata: {
              ...doc.metadata,
              sourceFile: fileName,
              chunkId,
            },
          }),
        );
      });
  }

  if (chunks.length === 0) {
    throw new Error("No non-empty chunks were produced from the documents");
  }

  console.log(
    `📚 Loaded ${rawDocs.length} documents and created ${chunks.length} chunks.`,
  );

  console.log(`🧹 Resetting vector table: ${config.tableName}`);

  await dropTablePurge(conn, config.tableName);

  console.log("🧠 Generating embeddings and populating OracleVS...");

  // The table was just dropped, so let OracleVS generate the row IDs.
  const vectorStore = await OracleVS.fromDocuments(chunks, embedder, {
    client: pool,
    tableName: config.tableName,
    distanceStrategy: DistanceStrategy.COSINE,
    query: "Initialization query",
  });

  console.log("✅ Oracle vector store populated.");

  // ---------------------------------------------------------------------
  // Optional IVF vector index
  //
  // IVF is used here rather than HNSW because this is an introductory
  // sample intended to run across a broad range of Oracle environments.
  //
  // HNSW uses Oracle's Vector Memory Pool for its in-memory neighbor graph
  // and can therefore introduce additional database-specific memory
  // requirements. This sample is intended to demonstrate the RAG and
  // OracleVS APIs rather than vector-index tuning.
  //
  // If index creation is unavailable or fails, the sample continues with
  // vector retrieval rather than treating index creation as a prerequisite
  // for demonstrating RAG.
  // ---------------------------------------------------------------------

  try {
    await createIndex(conn, vectorStore, {
      idxName: "IDX_FULL_RAG_IVF",
      idxType: "IVF",
      accuracy: 90,
    });

    console.log("✅ IVF vector index created.");
  } catch (err) {
    console.warn(
      "⚠️ IVF index creation failed; continuing with vector retrieval.",
    );
    console.warn(err);
  }

  return vectorStore;
}

async function getOrInitVectorStore(
  conn: oracledb.Connection,
  pool: oracledb.Pool,
  config: RagConfig,
): Promise<OracleVS> {
  const embedder = new OracleEmbeddings(conn, {
    provider: "database",
    model: config.embeddingModel,
  });

  if (!config.resetVectorStore) {
    console.log(
      `📦 Connecting to existing vector store: ${config.tableName}`,
    );

    // Reuse the existing OracleVS table without reading documents
    // or generating embeddings again.
    const vectorStore = new OracleVS(embedder, {
      client: pool,
      tableName: config.tableName,
      query: "Initialization query",
      distanceStrategy: DistanceStrategy.COSINE,
    });

    return vectorStore;
  }

  return ingestDocuments(conn, pool, config);
}

async function answerQuestion(
  vectorStore: OracleVS,
  ociClient: OciGenAiGenericChat,
  question: string,
): Promise<string> {
  // ---------------------------------------------------------------------
  // STEP 1: Retrieval
  // ---------------------------------------------------------------------

  console.log("\n-------------------------------------------------------");
  console.log("1. RETRIEVAL");
  console.log("-------------------------------------------------------");

  console.log(`🔎 USER QUERY: "${question}"\n`);

  const searchResults =
    await vectorStore.similaritySearchWithScore(question, 5);

  if (searchResults.length === 0) {
    throw new Error("No documents were retrieved for the query");
  }

  const tableRows = searchResults.map(([doc, score], idx) => ({
    Rank: idx + 1,
    "Vector Distance": Number(score).toFixed(4),
    Source: `${doc.metadata.sourceFile} (Chunk #${doc.metadata.chunkId})`,
    "Content Snippet": doc.pageContent
      .replace(/\s+/g, " ")
      .slice(0, 300),
  }));

  console.table(tableRows);

  // ---------------------------------------------------------------------
  // STEP 2: Prompt construction
  // ---------------------------------------------------------------------

  console.log("\n-------------------------------------------------------");
  console.log("2. PROMPT CONSTRUCTION");
  console.log("-------------------------------------------------------");

  const retrievedContext = searchResults
    .map(
      ([doc], index) =>
        `[Context ${index + 1} | ` +
        `${doc.metadata.sourceFile}, ` +
        `Chunk ${doc.metadata.chunkId}]\n` +
        doc.pageContent,
    )
    .join("\n\n");

  const promptTemplate = PromptTemplate.fromTemplate(`
You are a question-answering assistant.

Answer the user's question using only the supplied context.
Do not use outside knowledge.

If the context does not contain enough information to answer the question,
say exactly:

"I don't have enough information in the retrieved documents to answer that."

When making a factual claim, cite the context number that supports it,
for example [Context 1].

Do not invent citations.

Context:
{context}

Question:
{question}

Answer:
`);

  const formattedPrompt = await promptTemplate.format({
    context: retrievedContext,
    question,
  });

  console.log(formattedPrompt.trim());

  // ---------------------------------------------------------------------
  // STEP 3: Generation
  // ---------------------------------------------------------------------

  console.log("\n-------------------------------------------------------");
  console.log("3. LLM GENERATION");
  console.log("-------------------------------------------------------");

  console.log("🤖 Querying OCI Generative AI...\n");

  return chatWithOci(ociClient, formattedPrompt);
}

async function runCompleteRagPipeline() {
  const {
    ORACLEDB_USER,
    ORACLEDB_PASSWORD,
    ORACLEDB_CONNECTION_STRING,
    EMBEDDING_ONNX_MODEL,
    DOCUMENTS_FOLDER,
    OCI_COMPARTMENT_ID,
  } = process.env;

  const ociConfig: OciChatConfig = {
    compartmentId: OCI_COMPARTMENT_ID ?? "",
    modelId: process.env.OCI_MODEL_ID || "xai.grok-4.3",
    endpoint:
      process.env.OCI_ENDPOINT ||
      "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com",
    configFile: process.env.OCI_CONFIG_FILE || "~/.oci/config",
    profile: process.env.OCI_CONFIG_PROFILE || "DEFAULT",
    authType: resolveAuthType(process.env.OCI_AUTH_TYPE),
  };

  const ragConfig: RagConfig = {
    tableName: "FULL_RAG_DEMO",
    documentsFolder: DOCUMENTS_FOLDER ?? "",
    embeddingModel: EMBEDDING_ONNX_MODEL ?? "",
    resetVectorStore:
      (process.env.RESET_VECTOR_STORE ?? "true").toLowerCase() === "true",
  };

  // ---------------------------------------------------------------------
  // Configuration validation
  // ---------------------------------------------------------------------

  if (
    !ORACLEDB_USER ||
    !ORACLEDB_PASSWORD ||
    !ORACLEDB_CONNECTION_STRING ||
    !EMBEDDING_ONNX_MODEL ||
    !DOCUMENTS_FOLDER ||
    !OCI_COMPARTMENT_ID
  ) {
    throw new Error(
      "Missing required environment variables.\n" +
        "Ensure the following are set:\n" +
        "  ORACLEDB_USER\n" +
        "  ORACLEDB_PASSWORD\n" +
        "  ORACLEDB_CONNECTION_STRING\n" +
        "  EMBEDDING_ONNX_MODEL\n" +
        "  DOCUMENTS_FOLDER\n" +
        "  OCI_COMPARTMENT_ID",
    );
  }

  let pool: oracledb.Pool | undefined;
  let conn: oracledb.Connection | undefined;
  let ociClient: OciGenAiGenericChat | undefined;

  try {
    console.log("\n=======================================================");
    console.log("  🚀 END-TO-END ORACLE AI + LANGCHAIN RAG PIPELINE");
    console.log("=======================================================\n");

    console.log(
      `🔧 Vector store mode: ${
        ragConfig.resetVectorStore ? "REBUILD" : "REUSE"
      }`,
    );

    // ---------------------------------------------------------------------
    // Database connection
    // ---------------------------------------------------------------------

    pool = await oracledb.createPool({
      user: ORACLEDB_USER,
      password: ORACLEDB_PASSWORD,
      connectString: ORACLEDB_CONNECTION_STRING,
    });

    conn = await pool.getConnection();

    // ---------------------------------------------------------------------
    // OCI client
    // ---------------------------------------------------------------------

    ociClient = getOciClient(ociConfig);

    // ---------------------------------------------------------------------
    // Vector store
    //
    // REBUILD:
    //   Documents are loaded, split, embedded, and stored.
    //
    // REUSE:
    //   Existing vector table is opened directly. No document loading
    //   or re-embedding occurs.
    // ---------------------------------------------------------------------

    const vectorStore = await getOrInitVectorStore(
      conn,
      pool,
      ragConfig,
    );

    // ---------------------------------------------------------------------
    // Query
    // ---------------------------------------------------------------------

    const userQuery =
      process.argv.slice(2).join(" ") ||
      "How do Transformer models use attention masks?";

    const generatedResponse = await answerQuestion(
      vectorStore,
      ociClient,
      userQuery,
    );

    console.log("\n-------------------------------------------------------");
    console.log("FINAL ANSWER");
    console.log("-------------------------------------------------------");

    console.log(generatedResponse);

    console.log("\n✅ RAG pipeline completed successfully.");
    console.log(
      `📦 Vector store '${ragConfig.tableName}' remains available.`,
    );
  } catch (err) {
    console.error("\n❌ Pipeline Error:", err);
    throw err;
  } finally {
    console.log("\n🧹 Closing database resources...");

    // Deliberately do NOT drop the vector table here.
    //
    // Use RESET_VECTOR_STORE=true when you explicitly want to rebuild it.

    if (conn) {
      try {
        await conn.close();
      } catch (err) {
        console.warn(
          "⚠️ Error closing database connection:",
          err,
        );
      }
    }

    if (pool) {
      try {
        await pool.close(10);
      } catch (err) {
        console.warn(
          "⚠️ Error closing connection pool:",
          err,
        );
      }
    }

    if (ociClient) {
      await ociClient.close();
    }

    console.log("✨ Execution complete.\n");
  }
}

runCompleteRagPipeline().catch((err) => {
  console.error("Fatal error:", err);
  process.exitCode = 1;
});
