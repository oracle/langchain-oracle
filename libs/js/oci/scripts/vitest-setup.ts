import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";
import { afterAll, vi } from "vitest";

afterAll(awaitAllCallbacks);

if (process.env.DISABLE_CONSOLE_LOGS === "true") {
  console.log = vi.fn();
}
