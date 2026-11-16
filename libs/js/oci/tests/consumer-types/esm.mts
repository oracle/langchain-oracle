import type { Embeddings } from "@langchain/core/embeddings";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  OciGenAiEmbeddings,
  OciGenAiGenericChat,
} from "@oracle/langchain-oci";

declare const embeddings: OciGenAiEmbeddings;
declare const chat: OciGenAiGenericChat;

const embeddingsCheck: Embeddings = embeddings;
const chatCheck: BaseChatModel = chat;

export { embeddingsCheck, chatCheck };
