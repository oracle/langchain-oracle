import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";

import oracledb from "oracledb";
import * as common from "oci-common";
import * as generative_ai_inference from "oci-generativeaiinference";

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

type OciChatConfig = {
  compartmentId: string;
  modelId: string;
  endpoint: string;
  configFile: string;
  profile: string;
};

type RagConfig = {
  tableName: string;
  documentsFolder: string;
  embeddingModel: string;
  resetVectorStore: boolean;
};

function resolvePath(filePath: string): string {
  if (filePath.startsWith("~")) {
    return path.join(os.homedir(), filePath.slice(1));
  }

  return path.resolve(filePath);
}

function makeStableChunkId(source: string, chunkId: number): string {
  return createHash("sha256")
    .update(`${source}:${chunkId}`)
    .digest("hex")
    .slice(0, 36);
}

function getOciClient(
  config: OciChatConfig,
): generative_ai_inference.GenerativeAiInferenceClient {
  const provider = new common.ConfigFileAuthenticationDetailsProvider(
    resolvePath(config.configFile),
    config.profile,
  );

  const client = new generative_ai_inference.GenerativeAiInferenceClient({
    authenticationDetailsProvider: provider,
  });

  client.endpoint = config.endpoint;

  return client;
}

async function chatWithOci(
  client: generative_ai_inference.GenerativeAiInferenceClient,
  prompt: string,
  config: OciChatConfig,
): Promise<string> {
  const response = await client.chat({
    chatDetails: {
      compartmentId: config.compartmentId,
      servingMode: {
        servingType: "ON_DEMAND",
        modelId: config.modelId,
      },
      chatRequest: {
        apiFormat: "GENERIC",
        messages: [
          {
            role: "USER",
            content: [
              {
                type: "TEXT",
                text: prompt,
              } as generative_ai_inference.models.TextContent,
            ],
          },
        ],
        temperature: 0.2,
        topP: 0.9,
        maxTokens: Number(process.env.OCI_MAX_TOKENS ?? 1000),
        isStream: false,
      },
    },
  });

  const chatResponse = response?.chatResult?.chatResponse;

  if (!chatResponse) {
    throw new Error("OCI returned no valid chat response");
  }

  if (!("choices" in chatResponse)) {
    throw new Error("OCI returned a non-GENERIC chat response");
  }

  const choice = chatResponse.choices?.[0];

  if (!choice?.message?.content) {
    throw new Error("OCI returned no generated message content");
  }

  const text = choice.message.content
    .filter(
      (content): content is generative_ai_inference.models.TextContent =>
        content.type === "TEXT" && "text" in content,
    )
    .map((content) => content.text ?? "")
    .join("");

  if (!text.trim()) {
    throw new Error("OCI returned an empty chat response");
  }

  return text.trim();
}

async function ingestDocuments(
  conn: oracledb.Connection,
  pool: oracledb.Pool,
  config: RagConfig,
): Promise<OracleVS> {
  console.log("\n-------------------------------------------------------");
  console.log("DOCUMENT INGESTION");
  console.log("-------------------------------------------------------");

  const loader = new OracleDocLoader(conn, {
    dir: config.documentsFolder,
  });

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

  console.log("📄 Loading documents...");

  const rawDocs = await loader.load();

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

  // ---------------------------------------------------------------------
  // Prepare vector store
  // ---------------------------------------------------------------------

  if (config.resetVectorStore) {
    console.log(`🧹 Resetting vector table ${config.tableName}...`);
    await dropTablePurge(conn, config.tableName);
  }

  const deterministicIds = chunks.map((chunk) => {
    const sourcePath = String(chunk.metadata.source ?? "doc");
    const chunkId = Number(chunk.metadata.chunkId ?? 0);

    return makeStableChunkId(sourcePath, chunkId);
  });

  console.log("🧠 Generating embeddings and populating OracleVS...");

  const vectorStore = await OracleVS.fromDocuments(
    chunks,
    embedder,
    {
      client: pool,
      tableName: config.tableName,
      distanceStrategy: DistanceStrategy.COSINE,
      query: "Initialization query",
    },
    {
      ids: deterministicIds,
      mutateOnDuplicate: true,
    },
  );

  console.log("✅ Oracle vector store is ready.");

  // ---------------------------------------------------------------------
  // Optional vector index
  // ---------------------------------------------------------------------
  //
  // IVF is used in this basic sample instead of HNSW for portability.
  // HNSW uses Oracle's Vector Memory Pool for its in-memory neighbor graph
  // and can therefore require additional database-specific memory
  // configuration.
  //
  // This example is intended to demonstrate the basic OracleVS/RAG APIs,
  // not database vector-index tuning. The vector index is therefore an
  // optimization and is not allowed to prevent the core RAG example from
  // running.
  //
  // Production applications should benchmark IVF and HNSW for their
  // particular data size, latency, recall, and database configuration.
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

async function answerQuestion(
  vectorStore: OracleVS,
  ociClient: generative_ai_inference.GenerativeAiInferenceClient,
  ociConfig: OciChatConfig,
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
    "Content Snippet": doc.pageContent.replace(/\s+/g, " ").slice(0, 300),
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

  const answer = await chatWithOci(
    ociClient,
    formattedPrompt,
    ociConfig,
  );

  return answer;
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

  try {
    console.log("\n=======================================================");
    console.log("  🚀 END-TO-END ORACLE AI + LANGCHAIN RAG PIPELINE");
    console.log("=======================================================\n");

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

    const ociClient = getOciClient(ociConfig);

    // ---------------------------------------------------------------------
    // Ingest documents and create/reuse the vector store
    // ---------------------------------------------------------------------

    const vectorStore = await ingestDocuments(
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
      ociConfig,
      userQuery,
    );

    console.log("\n-------------------------------------------------------");
    console.log("FINAL ANSWER");
    console.log("-------------------------------------------------------");
    console.log(generatedResponse);

    console.log("\n✅ RAG pipeline completed successfully.");

    if (!ragConfig.resetVectorStore) {
      console.log(
        `📦 Vector store ${ragConfig.tableName} was preserved.`,
      );
    } else {
      console.log(
        `📦 Vector store ${ragConfig.tableName} remains available after this run.`,
      );
    }
  } catch (err) {
    console.error("\n❌ Pipeline Error:", err);
    throw err;
  } finally {
    console.log("\n🧹 Closing database resources...");

    // Intentionally do NOT drop the vector table here.
    //
    // The table is reusable between runs. Set RESET_VECTOR_STORE=true
    // when you explicitly want to rebuild the demo vector store.

    if (conn) {
      try {
        await conn.close();
      } catch (err) {
        console.warn("⚠️ Error closing database connection:", err);
      }
    }

    if (pool) {
      try {
        await pool.close(10);
      } catch (err) {
        console.warn("⚠️ Error closing connection pool:", err);
      }
    }

    console.log("✨ Execution complete.\n");
  }
}

runCompleteRagPipeline().catch((err) => {
  console.error("Fatal error:", err);
  process.exitCode = 1;
});