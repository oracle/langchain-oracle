import { IterableReadableStream } from "@langchain/core/utils/stream";

/**
 * Converts OCI's byte-oriented SSE response into validated JSON event objects.
 * It deliberately keeps transport chunking separate from SSE event framing.
 */
export class JsonServerEventsIterator {
  static readonly _DATA_FIELD = "data:";

  // Guard against an upstream stream that never emits an SSE event delimiter.
  static readonly _MAX_BUFFER_LENGTH = 1024 * 1024;

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
      const event = this._parseMessage(this._textBuffer);
      this._textBuffer = "";
      if (event !== undefined) {
        yield event;
      }
    }
  }

  private *_parseAvailableMessages(): Generator<unknown> {
    while (true) {
      // Consume every complete event while retaining a trailing partial event
      // for the next transport chunk.
      const delimiterIndex = this._findEventDelimiter();
      if (delimiterIndex === -1) {
        return;
      }

      const delimiterLength = this._getDelimiterLength(delimiterIndex);
      const eventText = this._textBuffer.slice(0, delimiterIndex);
      this._textBuffer = this._textBuffer.slice(
        delimiterIndex + delimiterLength
      );

      if (eventText.trim() !== "") {
        const event = this._parseMessage(eventText);
        if (event !== undefined) {
          yield event;
        }
      }
    }
  }

  private _findEventDelimiter(): number {
    const lfIndex = this._textBuffer.indexOf("\n\n");
    const crlfIndex = this._textBuffer.indexOf("\r\n\r\n");
    const crIndex = this._textBuffer.indexOf("\r\r");

    const delimiterIndexes = [lfIndex, crlfIndex, crIndex].filter(
      (index) => index !== -1
    );
    return delimiterIndexes.length > 0 ? Math.min(...delimiterIndexes) : -1;
  }

  private _getDelimiterLength(index: number): number {
    if (this._textBuffer.startsWith("\r\n\r\n", index)) {
      return 4;
    }
    return 2;
  }

  private _parseMessage(eventText: string): unknown | undefined {
    // SSE permits multiple data lines; join them according to the SSE format
    // before treating their contents as the OCI JSON payload.
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
    return this._tryParseTextToJson(jsonText);
  }

  private _assertBufferLength(): void {
    if (this._textBuffer.length > JsonServerEventsIterator._MAX_BUFFER_LENGTH) {
      throw new Error("Server-sent event exceeds maximum buffered size");
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
