import { IterableReadableStream } from "@langchain/core/utils/stream";

export class JsonServerEventsIterator {
  static readonly _DATA_PREFIX: string = "data: ";

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
    }

    // Flush a final buffered UTF-8 sequence before parsing the remaining data.
    this._textBuffer += this._textDecoder.decode();
    yield* this._parseAvailableMessages();

    if (this._textBuffer.trim() !== "") {
      throw new Error("Incomplete server-sent event at end of stream");
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
        yield this._parseMessage(eventText);
      }
    }
  }

  private _findEventDelimiter(): number {
    const lfIndex = this._textBuffer.indexOf("\n\n");
    const crlfIndex = this._textBuffer.indexOf("\r\n\r\n");

    if (lfIndex === -1) {
      return crlfIndex;
    }
    if (crlfIndex === -1) {
      return lfIndex;
    }
    return Math.min(lfIndex, crlfIndex);
  }

  private _getDelimiterLength(index: number): number {
    return this._textBuffer.startsWith("\r\n\r\n", index) ? 4 : 2;
  }

  private _parseMessage(eventText: string): unknown {
    // SSE permits multiple data lines; join them according to the SSE format
    // before treating their contents as the OCI JSON payload.
    const dataLines = eventText
      .split(/\r?\n/)
      .filter((line) => line.startsWith(JsonServerEventsIterator._DATA_PREFIX));

    if (dataLines.length === 0) {
      throw new Error("Event text is empty, too short or malformed");
    }

    const jsonText = dataLines
      .map((line) =>
        line.substring(JsonServerEventsIterator._DATA_PREFIX.length)
      )
      .join("\n");
    return this._tryParseTextToJson(jsonText);
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
