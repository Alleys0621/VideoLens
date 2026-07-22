/**
 * AudioWorklet processor: 把麦克风输入转成 16kHz mono PCM 16-bit chunks.
 *
 * 浏览器原生 AudioContext 默认采样率 48000Hz, 我们要 resample 到 16kHz
 * 给 DashScope paraformer-realtime-v2 (要求 16kHz mono PCM s16).
 *
 * 每个 process() 调用拿到 128 samples (约 2.67ms @ 48kHz),
 * 累积到 ~320 samples (10ms @ 16kHz 等价) 后通过 port.postMessage 发出,
 * 主线程再转 s16 + WebSocket 上传.
 *
 * 性能: AudioWorklet 在独立线程跑, 不阻塞主线程 UI.
 */
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 48000 → 16000 downsample 比例
    this._ratio = 48000 / 16000;
    // 累积 buffer (Float32, downsampled), 一次性发出 ~200ms 的数据
    this._buffer = new Float32Array(0);
    // 每 200ms 发一次 (16kHz * 0.2s = 3200 samples)
    this._flushThreshold = 3200;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono Float32Array, 128 samples @ 48kHz

    // 简单 downsample: 每 ratio 个样本取一个 (线性插值可选, 这里 nearest)
    // 48000 / 16000 = 3, 即每 3 个样本取 1 个
    const outLength = Math.floor(channelData.length / this._ratio);
    const downsampled = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      downsampled[i] = channelData[Math.floor(i * this._ratio)];
    }

    // 累积到 buffer
    const merged = new Float32Array(this._buffer.length + downsampled.length);
    merged.set(this._buffer, 0);
    merged.set(downsampled, this._buffer.length);

    // 达到阈值就发给主线程
    while (merged.length >= this._flushThreshold) {
      const chunk = merged.slice(0, this._flushThreshold);
      this.port.postMessage(chunk);
      // 剩下的留到下一次
      this._buffer = merged.slice(this._flushThreshold);
      return true;
    }

    this._buffer = merged;
    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
