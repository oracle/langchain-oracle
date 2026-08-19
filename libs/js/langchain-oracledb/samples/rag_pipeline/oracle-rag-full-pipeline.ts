import oracledb from "oracledb";
import * as common from "oci-common";
import * as generative_ai_inference from "oci-generativeaiinference";
import { Document } from "@langchain/core/documents";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  OracleDocLoader,
  OracleTextSplitter,
  OracleEmbeddings,
  OracleSummary,
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

async function chatWithOci(prompt: string, config: OciChatConfig): Promise<string> {
  const provider = new common.ConfigFileAuthenticationDetailsProvider(
    config.configFile,
    config.profile
  );
  const client = new generative_ai_inference.GenerativeAiInferenceClient({
    authenticationDetailsProvider: provider,
  });
  client.endpoint = config.endpoint;

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
              { type: "TEXT", text: prompt } as generative_ai_inference.models.TextContent,
            ],
          },
        ],
        temperature: 0.2,
        topP: 0.9,
        maxTokens: 500,
        isStream: false,
      },
    },
  });

  if (!response || response instanceof ReadableStream) {
    throw new Error("OCI returned no non-streaming chat response");
  }

  const chatResponse = response.chatResult.chatResponse;
  if (!("choices" in chatResponse)) {
    throw new Error("OCI returned a non-GENERIC chat response");
  }

  const text = (chatResponse.choices[0]?.message.content ?? [])
    .filter(
      (content): content is generative_ai_inference.models.TextContent =>
        content.type === "TEXT" && "text" in content
    )
    .map((content) => content.text ?? "")
    .join("");

  if (!text) {
    throw new Error("OCI returned an empty chat response");
  }

  return text;
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

  if (
    !ORACLEDB_USER ||
    !ORACLEDB_PASSWORD ||
    !ORACLEDB_CONNECTION_STRING ||
    !EMBEDDING_ONNX_MODEL ||
    !DOCUMENTS_FOLDER ||
    !OCI_COMPARTMENT_ID
  ) {
    throw new Error(
      "Missing required environment variables. Ensure ORACLEDB_USER, ORACLEDB_PASSWORD, " +
        "ORACLEDB_CONNECTION_STRING, EMBEDDING_ONNX_MODEL, DOCUMENTS_FOLDER, and OCI_COMPARTMENT_ID are set."
    );
  }

  const TABLE_NAME = "FULL_RAG_DEMO";
  const pool = await oracledb.createPool({
    user: ORACLEDB_USER,
    password: ORACLEDB_PASSWORD,
    connectString: ORACLEDB_CONNECTION_STRING,
  });

  const conn = await pool.getConnection();

  try {
    console.log("\n=======================================================");
    console.log("  🚀 END-TO-END ORACLE AI + LANGCHAIN RAG PIPELINE");
    console.log("=======================================================\n");

    // ---------------------------------------------------------------------
    // STEP 1: Load, Split & Embed Documents
    // ---------------------------------------------------------------------
    const loader = new OracleDocLoader(conn, { dir: DOCUMENTS_FOLDER });
    const splitter = new OracleTextSplitter(conn, {
      by: "words",
      max: 30,
      overlap: 5,
      normalize: "all",
    });
    const embedder = new OracleEmbeddings(conn, {
      provider: "database",
      model: EMBEDDING_ONNX_MODEL,
    });
    const summarizer = new OracleSummary(conn, {
      provider: "database",
      gLevel: "P",
    });

    console.log("📄 Processing and indexing documents into Oracle AI Vector Search...");
    const rawDocs = await loader.load();
    const chunks: Document[] = [];

    for (const doc of rawDocs) {
      const summary = await summarizer.getSummary(doc.pageContent);
      const textParts = await splitter.splitText(doc.pageContent);
      const fileName = doc.metadata.source?.split("/").pop() || "doc";

      textParts.forEach((part, idx) => {
        chunks.push(
          new Document({
            pageContent: part.trim(),
            metadata: {
              sourceFile: fileName,
              chunkId: idx + 1,
              summarySnippet: summary ? `${summary.slice(0, 60)}...` : "N/A",
            },
          })
        );
      });
    }

    // Prepare table & populate vector store
    await dropTablePurge(conn, TABLE_NAME);
    const deterministicIds = chunks.map(
      (c) => `${c.metadata.sourceFile.replace(".", "_")}_chk_${c.metadata.chunkId}`
    );

    const vectorStore = await OracleVS.fromDocuments(
      chunks,
      embedder,
      {
        client: pool,
        tableName: TABLE_NAME,
        distanceStrategy: DistanceStrategy.COSINE,
        query: "Initialization query",
      },
      {
        ids: deterministicIds,
        mutateOnDuplicate: true,
      }
    );

    // Build Inverted File (IVF) index
    await createIndex(conn, vectorStore, {
      idxName: "IDX_FULL_RAG_IVF",
      idxType: "IVF",
      neighborPart: 32,
      accuracy: 90,
    });

    console.log(`✅ Loaded ${rawDocs.length} documents into ${chunks.length} indexed chunks.\n`);

    // ---------------------------------------------------------------------
    // STEP 2: Retrieval (The "R" in RAG)
    // ---------------------------------------------------------------------
    const userQuery = "How do Transformer models use attention masks?";
    console.log(`🔎 USER QUERY: "${userQuery}"\n`);

    console.log("-------------------------------------------------------");
    console.log("1. RETRIEVED CONTEXT CHUNKS (Top 2 Vector Matches):");
    console.log("-------------------------------------------------------");

    const searchResults = await vectorStore.similaritySearchWithScore(userQuery, 2);

    const tableRows = searchResults.map(([doc, score], idx) => ({
      Rank: idx + 1,
      "Relevance Match": `${Math.max(0, (1 - Math.abs(score)) * 100).toFixed(1)}%`,
      Source: `${doc.metadata.sourceFile} (Chunk #${doc.metadata.chunkId})`,
      "Content Snippet": doc.pageContent.replace(/\n/g, " "),
    }));

    console.table(tableRows);

    // ---------------------------------------------------------------------
    // STEP 3: Prompt Construction (The "A" in RAG)
    // ---------------------------------------------------------------------
    const retrievedContext = searchResults
      .map(([doc], i) => `[Context ${i + 1}]: ${doc.pageContent}`)
      .join("\n\n");

    const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful AI assistant answering user questions using only the provided context.
If the context does not contain enough information to answer, state that clearly.

Context:
{context}

Question: {question}

Answer:`);

    const formattedPrompt = await promptTemplate.format({
      context: retrievedContext,
      question: userQuery,
    });

    console.log("\n-------------------------------------------------------");
    console.log("2. CONSTRUCTED PROMPT FOR LLM COMPLETION:");
    console.log("-------------------------------------------------------");
    console.log(formattedPrompt.trim());

    // ---------------------------------------------------------------------
    // STEP 4: LLM Generation (The "G" in RAG)
    // ---------------------------------------------------------------------
    console.log("\n-------------------------------------------------------");
    console.log("3. LLM GENERATION:");
    console.log("-------------------------------------------------------");
    console.log("🤖 Querying LLM with constructed prompt...\n");

    const generatedResponse = await chatWithOci(formattedPrompt, ociConfig);

    console.log("📝 FINAL ANSWER:");
    console.log(generatedResponse);

  } catch (err) {
    console.error("❌ Pipeline Error:", err);
  } finally {
    console.log("\n🧹 Cleaning up database resources...");
    try {
      await dropTablePurge(conn, TABLE_NAME);
      await conn.close();
      await pool.close(10);
    } catch {
      /* ignore cleanup errors */
    }
    console.log("✨ Execution complete.\n");
  }
}

runCompleteRagPipeline().catch(console.error);
