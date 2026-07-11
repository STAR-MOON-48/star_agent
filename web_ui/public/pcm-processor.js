class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16_000;
    this.frameSize = Math.max(128, Math.round(sampleRate * 0.03));
    this.pending = [];
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let index = 0; index < input.length; index += 1) {
      this.pending.push(input[index]);
    }
    while (this.pending.length >= this.frameSize) {
      const frame = this.pending.splice(0, this.frameSize);
      const ratio = sampleRate / this.targetSampleRate;
      const outputLength = Math.max(1, Math.floor(frame.length / ratio));
      const pcm = new Int16Array(outputLength);
      for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
        const start = Math.floor(outputIndex * ratio);
        const end = Math.min(frame.length, Math.max(start + 1, Math.floor((outputIndex + 1) * ratio)));
        let sum = 0;
        for (let sourceIndex = start; sourceIndex < end; sourceIndex += 1) sum += frame[sourceIndex];
        const sample = Math.max(-1, Math.min(1, sum / (end - start)));
        pcm[outputIndex] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
