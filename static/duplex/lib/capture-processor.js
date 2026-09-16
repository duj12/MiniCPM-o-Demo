/**
 * capture-processor.js — AudioWorklet processor for mixing + capturing audio
 *
 * Runs on the audio rendering thread. Receives mixed audio from the graph
 * (browser auto-sums all connected inputs), passes it through to output
 * (for MediaStreamDestination + monitor), and posts 1-second PCM chunks
 * to the main thread via MessagePort.
 *
 * Commands via port.postMessage:
 *   { command: 'start' }  — begin accumulating and emitting chunks
 *   { command: 'stop' }   — stop accumulating, flush buffer
 *
 * Emits via port.postMessage:
 *   { type: 'chunk', audio: Float32Array, t0: number, epoch: number }
 *     — PCM chunk (Transferable). ``t0`` 是**本块首采样**在 AudioContext 上的
 *       时刻（秒），``epoch`` 是 context 代号。
 *
 * ## 为什么每个块要带 t0（不要删）
 *
 * 服务端的 AEC 参考轨必须落在麦克风时钟上的**实际播出**时刻。而"实际播出"
 * 只有浏览器知道（`AudioContext.currentTime`）。服务端曾经用
 * 「收到音频时的会话位置 + 提前量」去**预测**，误差含网络往返、本线程阻塞
 * （主线程同时在跑 25fps 抓帧）、mic 在途积压，**逐句变化** —— 固定常量 D
 * 吸收不了，表现为"第一句回声消得掉、后面就失效"。
 *
 * 带 t0 之后服务端能拟合出「ctx 时钟 ↔ 会话采样」的**精确**映射（两个时钟
 * 域数的是同一路流），从而把浏览器承诺的起播时刻直接换算过去。
 * 采集与播放共用同一个 AudioContext，所以两者在同一时间轴上 —— 这是整条
 * 链路能对齐的前提。
 */
class CaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const po = options.processorOptions || {};
        this._chunkSize = po.chunkSize || 16000;
        this._epoch = po.epoch || 0;
        this._buffer = new Float32Array(0);
        this._active = false;
        // 当前缓冲里**首个采样**的 AudioContext 时刻。开始采集时锚定，
        // 之后每发出一个块就按块长推进 —— 纯整数/浮点记账，无累计漂移
        // （用 currentTime 反推才会漂）。
        this._t0 = 0;

        this.port.onmessage = (e) => {
            const { command } = e.data;
            if (command === 'start') {
                this._active = true;
                this._buffer = new Float32Array(0);
                this._t0 = currentTime;
            } else if (command === 'stop') {
                if (this._buffer.length > 0) {
                    const remaining = this._buffer.slice(0);
                    this.port.postMessage(
                        { type: 'chunk', audio: remaining, t0: this._t0,
                          epoch: this._epoch, final: true },
                        [remaining.buffer]
                    );
                }
                this._active = false;
                this._buffer = new Float32Array(0);
            }
        };
    }

    process(inputs, outputs) {
        const input = inputs[0]?.[0];
        const output = outputs[0]?.[0];

        // Always pass-through: enables MediaStreamDestination + monitor downstream
        if (input && output) {
            output.set(input);
        }

        if (!this._active || !input || input.length === 0) {
            return true;
        }

        // Accumulate input samples
        const newBuf = new Float32Array(this._buffer.length + input.length);
        newBuf.set(this._buffer);
        newBuf.set(input, this._buffer.length);
        this._buffer = newBuf;

        // Emit full chunks。⚠️ ``t0`` 必须在推进之前取 —— 取的是**本块**
        // 首采样的时刻。
        while (this._buffer.length >= this._chunkSize) {
            const t0 = this._t0;
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);
            this._t0 = t0 + this._chunkSize / sampleRate;
            this.port.postMessage(
                { type: 'chunk', audio: chunk, t0: t0, epoch: this._epoch },
                [chunk.buffer]
            );
        }

        return true;
    }
}

registerProcessor('capture-processor', CaptureProcessor);
