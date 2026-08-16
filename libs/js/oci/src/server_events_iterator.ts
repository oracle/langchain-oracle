import { IterableReadableStream } from "@langchain/core/utils/stream";

/**
 * Converts OCI's byte-oriented SSE response into validated JSON event objects.
 * It deliberately keeps transport chunking separate from SSE event framing.
 */
export class JsonServerEventsIterator {
  static readonly _DATA_FIELD = "data:";

  static readonly _DONE_SENTINEL = "[DONE]";

  // Guard against an upstream stream that never emits an SSE delimiter. This
  // is a JavaScript string length, measured in UTF-16 code units, not bytes.
  static readonly _MAX_BUFFERED_TEXT_LENGTH = 1024 * 1024;

  _eventsStream: IterableReadableStream<Uint8Array>;

  _textDecoder: TextDecoder = new TextDecoder();

  _textBuffer: string = "";

  constructor(sourceStream: ReadableStream<Uint8Array>) {
    this._eventsStream =
      IterableReadableStream.fromReadableStream(sourceStream);
  }

  async *[Symbol.asyncIterator](): AsyncIterator<unknown> {
    for await (const eventRawData of this._eventsStream) {
      // A network chunk is not an SSE message boundary. Streaming decoding also
      // retains incomplete UTF-8 sequences (for example, a split emoji).
      this._textBuffer += this._textDecoder.decode(eventRawData, {
        stream: true,
      });
      yield* this._parseAvailableMessages();
      this._assertBufferLength();
    }

    // Flush a final buffered UTF-8 sequence before parsing the remaining data.
    this._textBuffer += this._textDecoder.decode();
    yield* this._parseAvailableMessages();

    // The SSE parsing algorithm dispatches a final event at EOF even when it
    // is not followed by a blank line.
    if (this._textBuffer.trim() !== "") {
      yield* this._parseFinalMessage();
    }
  }

  private *_parseAvailableMessages(): Generator<unknown> {
    while (true) {
      // Consume every complete event while retaining a trailing partial event
      // for the next transport chunk.
      const delimiter = this._findEventDelimiter();
      if (!delimiter) {
        return;
      }

      const eventText = this._textBuffer.slice(0, delimiter.index);
      this._textBuffer = this._textBuffer.slice(
        delimiter.index + delimiter.length
      );

      if (eventText.trim() !== "") {
        const event = this._parseMessage(eventText);
        if (event !== undefined) {
          yield event;
        }
      }
    }
  }

  private _findEventDelimiter(): { index: number; length: number } | undefined {
    // An SSE blank line is two complete line endings. Match the longest
    // alternatives first so a CRLF is never mistaken for separate CR and LF
    // endings when a server mixes newline styles.
    const delimiter = this._textBuffer.match(
      /\r\n\r\n|\r\n\r|\r\n\n|\r\r\n|\n\r\n|\r\r|\n\r|\n\n/
    );
    if (!delimiter || delimiter.index === undefined) {
      return undefined;
    }
    return { index: delimiter.index, length: delimiter[0].length };
  }

  private _parseMessage(eventText: string): unknown | undefined {
    // SSE permits multiple data lines; join them according to the SSE format
    // before treating their contents as the OCI JSON payload. OCI chat
    // streaming consumes only data payloads, so event, id, and retry fields
    // are intentionally ignored.
    const dataLines = eventText
      .split(/\r\n|\r|\n/)
      .filter((line) => line.startsWith(JsonServerEventsIterator._DATA_FIELD));

    if (dataLines.length === 0) {
      // Comments, keepalives, and control-only SSE events do not dispatch data.
      return undefined;
    }

    const jsonText = dataLines
      .map((line) => {
        const data = line.substring(
          JsonServerEventsIterator._DATA_FIELD.length
        );
        // The optional single space after `data:` is excluded from the value.
        return data.startsWith(" ") ? data.substring(1) : data;
      })
      .join("\n");
    // OCI's native stream is JSON-only, but compatible SSE gateways may use
    // the conventional OpenAI-style terminal sentinel.
    if (jsonText.trim() === JsonServerEventsIterator._DONE_SENTINEL) {
      return undefined;
    }
    return this._tryParseTextToJson(jsonText);
  }

  private *_parseFinalMessage(): Generator<unknown> {
    try {
      const event = this._parseMessage(this._textBuffer);
      this._textBuffer = "";
      if (event !== undefined) {
        yield event;
      }
    } catch (error) {
      // Do not silently discard a partial terminal frame: callers need a
      // distinct error when a connection ends during an SSE JSON payload.
      throw new Error("Stream ended with an incomplete server-sent event", {
        cause: error,
      });
    }
  }

  private _assertBufferLength(): void {
    if (
      this._textBuffer.length >
      JsonServerEventsIterator._MAX_BUFFERED_TEXT_LENGTH
    ) {
      throw new Error("Server-sent event exceeds maximum buffered text length");
    }
  }

  private _tryParseTextToJson(jsonText: string): unknown {
    const parsedJson: unknown = this._parseTextToJson(jsonText);
    this._assertParsedJson(parsedJson);
    return parsedJson;
  }

  private _parseTextToJson(jsonText: string): unknown {
    try {
      return JSON.parse(jsonText);
    } catch {
      throw new Error("Could not parse event data as JSON");
    }
  }

  private _assertParsedJson(parsedJson: unknown): asserts parsedJson is object {
    if (typeof parsedJson !== "object" || parsedJson === null) {
      throw new Error("Event data could not be parsed into an object");
    }
  }
}
