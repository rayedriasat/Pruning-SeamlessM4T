// AudioWorklet: takes whatever buffer size the browser hands us and emits
// fixed 512-sample int16 frames suitable for Silero VAD on the server.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = 512;
    this.buf = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;

    // Append new samples to the rolling buffer
    const merged = new Float32Array(this.buf.length + ch.length);
    merged.set(this.buf, 0);
    merged.set(ch, this.buf.length);
    this.buf = merged;

    // Emit complete frames
    while (this.buf.length >= this.frameSize) {
      const frame = this.buf.subarray(0, this.frameSize);
      this.buf = this.buf.subarray(this.frameSize);

      // Float32 [-1,1] -> Int16 little-endian
      const out = new Int16Array(this.frameSize);
      let peak = 0;
      for (let i = 0; i < this.frameSize; i++) {
        let s = frame[i];
        if (s > 1) s = 1; else if (s < -1) s = -1;
        const a = s < 0 ? -s : s;
        if (a > peak) peak = a;
        out[i] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7fff) | 0;
      }
      // Send PCM and a cheap level meter value.
      this.port.postMessage({ pcm: out.buffer, peak }, [out.buffer]);
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
