"use strict";

class MonoRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const frameLength = input[0].length;
    const mono = new Float32Array(frameLength);
    for (let channel = 0; channel < input.length; channel += 1) {
      const channelData = input[channel];
      for (let index = 0; index < frameLength; index += 1) {
        mono[index] += channelData[index] / input.length;
      }
    }
    this.port.postMessage(mono, [mono.buffer]);
    return true;
  }
}

registerProcessor("mono-recorder", MonoRecorderProcessor);
