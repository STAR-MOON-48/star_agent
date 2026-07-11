const DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1";
const MAX_AUDIO_BASE64_SIZE = 10 * 1024 * 1024;

export interface MimoVoiceConfig {
  apiKey: string;
  baseUrl?: string;
  asrModel?: string;
  ttsModel?: string;
  voice?: string;
  style?: string;
  language?: "auto" | "zh" | "en";
}

export interface VoiceCapabilities {
  available: boolean;
  provider: "xiaomi-mimo";
  asrModel: string;
  ttsModel: string;
  voice: string;
  inputSampleRate: number;
  outputSampleRate: number;
  reason?: string;
}

export type FetchLike = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>;

export class MimoVoiceService {
  readonly inputSampleRate = 16_000;
  readonly outputSampleRate = 24_000;
  readonly asrModel: string;
  readonly ttsModel: string;
  readonly voice: string;

  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly style: string;
  private readonly language: "auto" | "zh" | "en";

  constructor(
    config: MimoVoiceConfig,
    private readonly fetchImpl: FetchLike = fetch,
  ) {
    this.apiKey = config.apiKey.trim();
    this.baseUrl = (config.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, "");
    this.asrModel = config.asrModel || "mimo-v2.5-asr";
    this.ttsModel = config.ttsModel || "mimo-v2.5-tts";
    this.voice = config.voice || "冰糖";
    this.style = config.style || "自然、清澈、真诚，语速适中，像面对面持续交谈。";
    this.language = config.language || "auto";
  }

  get available(): boolean {
    return this.apiKey.length > 0;
  }

  capabilities(): VoiceCapabilities {
    return {
      available: this.available,
      provider: "xiaomi-mimo",
      asrModel: this.asrModel,
      ttsModel: this.ttsModel,
      voice: this.voice,
      inputSampleRate: this.inputSampleRate,
      outputSampleRate: this.outputSampleRate,
      ...(!this.available
        ? { reason: "未配置 MIMO_API_KEY 或 XIAOMI_API_KEY" }
        : {}),
    };
  }

  async transcribePcm(pcmBytes: Uint8Array, sampleRate = this.inputSampleRate): Promise<string> {
    this.assertAvailable();
    if (pcmBytes.byteLength < sampleRate * 2 * 0.2) {
      throw new Error("录音太短，无法识别");
    }
    const audioData = wavDataUri(pcmBytes, sampleRate);
    if (audioData.length > MAX_AUDIO_BASE64_SIZE) {
      throw new Error("本轮录音超过 MiMo ASR 10 MB 限制");
    }
    const response = await this.fetchImpl(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        model: this.asrModel,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "input_audio",
                input_audio: { data: audioData },
              },
            ],
          },
        ],
        asr_options: { language: this.language },
      }),
    });
    const completion = await responseJson(response, "MiMo ASR");
    const transcript = completionText(completion);
    if (!transcript) throw new Error("MiMo ASR 没有返回文字");
    return transcript;
  }

  async streamSpeech(
    text: string,
    onChunk: (pcmBase64: string) => void | Promise<void>,
  ): Promise<number> {
    this.assertAvailable();
    const content = text.trim();
    if (!content) return 0;
    const response = await this.fetchImpl(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        model: this.ttsModel,
        messages: [
          { role: "user", content: this.style },
          { role: "assistant", content },
        ],
        audio: { format: "pcm16", voice: this.voice },
        stream: true,
      }),
    });
    if (!response.ok) await throwResponseError(response, "MiMo TTS");

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const completion = await response.json() as unknown;
      const encoded = audioData(completion);
      if (!encoded) throw new Error("MiMo TTS 没有返回音频");
      await onChunk(encoded);
      return base64ByteLength(encoded);
    }
    if (!response.body) throw new Error("MiMo TTS 流没有响应体");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let totalBytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const parsed = splitSseBuffer(buffer, done);
      buffer = parsed.remainder;
      for (const payload of parsed.payloads) {
        if (payload === "[DONE]") continue;
        let chunk: unknown;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue;
        }
        const encoded = audioData(chunk);
        if (!encoded) continue;
        totalBytes += base64ByteLength(encoded);
        await onChunk(encoded);
      }
      if (done) break;
    }
    if (!totalBytes) throw new Error("MiMo TTS 流中没有音频数据");
    return totalBytes;
  }

  private headers(): Record<string, string> {
    return {
      authorization: `Bearer ${this.apiKey}`,
      "content-type": "application/json",
    };
  }

  private assertAvailable(): void {
    if (!this.available) {
      throw new Error("未配置 MIMO_API_KEY 或 XIAOMI_API_KEY");
    }
  }
}

export function wavDataUri(pcmBytes: Uint8Array, sampleRate: number): string {
  const wav = pcm16ToWav(pcmBytes, sampleRate);
  return `data:audio/wav;base64,${Buffer.from(wav).toString("base64")}`;
}

export function pcm16ToWav(pcmBytes: Uint8Array, sampleRate: number): Uint8Array {
  const output = Buffer.allocUnsafe(44 + pcmBytes.byteLength);
  output.write("RIFF", 0, "ascii");
  output.writeUInt32LE(36 + pcmBytes.byteLength, 4);
  output.write("WAVE", 8, "ascii");
  output.write("fmt ", 12, "ascii");
  output.writeUInt32LE(16, 16);
  output.writeUInt16LE(1, 20);
  output.writeUInt16LE(1, 22);
  output.writeUInt32LE(sampleRate, 24);
  output.writeUInt32LE(sampleRate * 2, 28);
  output.writeUInt16LE(2, 32);
  output.writeUInt16LE(16, 34);
  output.write("data", 36, "ascii");
  output.writeUInt32LE(pcmBytes.byteLength, 40);
  Buffer.from(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength).copy(output, 44);
  return output;
}

function completionText(completion: unknown): string {
  const choices = record(completion).choices;
  if (!Array.isArray(choices) || choices.length === 0) return "";
  const content = record(record(choices[0]).message).content;
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    const value = record(part);
    return typeof value.text === "string"
      ? value.text
      : typeof value.content === "string"
        ? value.content
        : "";
  }).join("").trim();
}

function audioData(completion: unknown): string | null {
  const choices = record(completion).choices;
  if (!Array.isArray(choices) || choices.length === 0) return null;
  const choice = record(choices[0]);
  for (const holder of [record(choice.delta), record(choice.message)]) {
    const data = record(holder.audio).data;
    if (typeof data === "string" && data.length > 0) return data;
  }
  return null;
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

async function responseJson(response: Response, label: string): Promise<unknown> {
  if (!response.ok) await throwResponseError(response, label);
  return response.json();
}

async function throwResponseError(response: Response, label: string): Promise<never> {
  const body = (await response.text()).slice(0, 600);
  throw new Error(`${label} 请求失败 (${response.status})${body ? `: ${body}` : ""}`);
}

function splitSseBuffer(
  source: string,
  flush: boolean,
): { payloads: string[]; remainder: string } {
  const normalized = source.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const remainder = flush ? "" : blocks.pop() || "";
  const payloads = blocks.flatMap((block) => block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter(Boolean));
  if (flush && blocks.length === 0 && normalized.trim().startsWith("data:")) {
    payloads.push(normalized.trim().slice(5).trim());
  }
  return { payloads, remainder };
}

function base64ByteLength(encoded: string): number {
  const padding = encoded.endsWith("==") ? 2 : encoded.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor(encoded.length * 3 / 4) - padding);
}
