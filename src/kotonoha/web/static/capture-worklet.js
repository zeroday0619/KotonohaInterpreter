// Microphone capture worklet.
//
// The worklet thread posts raw blocks to the page rather than resampling here:
// an AudioWorklet runs on the audio thread, where an allocation or a long
// computation is a dropout. The page batches blocks before sending, because one
// WebSocket frame per 128 samples is roughly 375 frames a second.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.enabled = false;
    this.port.onmessage = (message) => {
      if (message.data && typeof message.data.enabled === "boolean") {
        this.enabled = message.data.enabled;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!this.enabled || !input || input.length === 0) {
      return true;
    }
    const channel = input[0];
    if (!channel || channel.length === 0) {
      return true;
    }
    // Copy: the render quantum buffer is reused by the audio thread.
    this.port.postMessage(new Float32Array(channel));
    return true;
  }
}

registerProcessor("kotonoha-capture", CaptureProcessor);
