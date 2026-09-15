/**
 * Orchestrator 会话客户端 —— 替代 RealtimeSession 的角色。
 *
 * **为什么不复用 RealtimeSession 协议**：
 *   1. 它与其 gateway 生命周期不可分（queue_done → session.init →
 *      session.created.active_model），Orchestrator 没有这些，伪造会让
 *      前端永远带着死握手逻辑
 *   2. 它的 input.append 把音频和视频**捆在一条消息**里 —— 结构上装不下
 *      「25fps 人脸 + 1fps OmniLLM + 100ms 音频」三条节奏
 *   3. 没有容纳播放回执的位置，而云端 AEC 的参考时钟依赖它
 *
 * 保留的**类形态**与 RealtimeSession 一致：connect / start / sendChunk /
 * stop，以及 onEvent 回调，便于渐进替换。
 *
 * 音频在线保持 float32（与采集层一致）：云端 AEC 与 OmniLLM 都要 float32，
 * 只有 ASR 要 int16 —— 服务端转一次优于浏览器转了再转回来。
 * 带宽：100ms float32 16k 单声道 = 6.4KB → base64 8.5KB × 10/s = 85KB/s。
 */
(function (global) {
  'use strict';

  class OrchestratorSession {
    constructor(options) {
      const opts = options || {};
      this.url = opts.url || null;
      this.onEvent = opts.onEvent || null;
      /** 会话级回调 */
      this.onOpen = opts.onOpen || null;
      this.onClose = opts.onClose || null;
      this.onError = opts.onError || null;

      this.ws = null;
      this.sessionId = null;
      this.ready = false;
      this.closed = false;

      /** TTS 播放器（由外部注入，需实现 playChunk/stop） */
      this.player = opts.player || null;
      /** 回执发送节流：每个 response 只在起播时报一次 started */
      this._playbackStarted = {};
      /** 统计 */
      this.stats = {
        audioSent: 0,
        faceFramesSent: 0,
        omniFramesSent: 0,
        bytesSent: 0,
        ttsFrames: 0,
        ttsBytes: 0,
      };
    }

    // ------------------------------------------------------------------ //

    _resolveUrl() {
      if (this.url) return this.url;
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const base = `${proto}//${location.host}/v1/orchestrator`;
      // 复用既有的客户端身份标记（与 MiniCPM 前端一致）
      if (global.ClientIdentity && global.ClientIdentity.appendToUrl) {
        return global.ClientIdentity.appendToUrl(base);
      }
      return base;
    }

    _emit(ev) {
      try {
        if (this.onEvent) this.onEvent(ev);
      } catch (e) {
        console.error('[orch] onEvent 抛错', e);
      }
    }

    _send(obj) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
      const s = JSON.stringify(obj);
      this.ws.send(s);
      this.stats.bytesSent += s.length;
      return true;
    }

    // ------------------------------------------------------------------ //

    /** 建连并等 session.ready。 */
    start(payload) {
      const self = this;
      return new Promise(function (resolve, reject) {
        const url = self._resolveUrl();
        let ws;
        try {
          ws = new WebSocket(url);
        } catch (e) {
          reject(e);
          return;
        }
        self.ws = ws;
        let settled = false;

        const timer = setTimeout(function () {
          if (!settled) {
            settled = true;
            reject(new Error('session.ready 超时'));
          }
        }, 15000);

        ws.onopen = function () {
          self._send(
            Object.assign(
              { type: 'session.start' },
              payload || {}
            )
          );
        };

        ws.onmessage = function (ev) {
          let msg;
          try {
            msg = JSON.parse(ev.data);
          } catch (e) {
            return;
          }
          if (msg.type === 'session.ready') {
            self.sessionId = msg.session_id;
            self.ready = true;
            if (!settled) {
              settled = true;
              clearTimeout(timer);
              resolve(msg);
            }
            if (self.onOpen) self.onOpen(msg);
            return;
          }
          self._handle(msg);
        };

        ws.onerror = function (e) {
          if (self.onError) self.onError(e);
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            reject(new Error('WebSocket 连接失败'));
          }
        };

        ws.onclose = function (e) {
          self.closed = true;
          self.ready = false;
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            reject(new Error('连接在 ready 前关闭 code=' + e.code));
          }
          if (self.onClose) self.onClose(e);
          self._emit({ type: 'session.closed', code: e.code });
        };
      });
    }

    /** 处理服务端消息。 */
    _handle(msg) {
      const t = msg.type;
      if (t === 'tts.start') {
        if (this.player && this.player.beginResponse) {
          this.player.beginResponse(msg.response_id, msg.sample_rate);
        }
        this._playbackStarted[msg.response_id] = false;
      } else if (t === 'tts.audio') {
        this.stats.ttsFrames += 1;
        const raw = atob(msg.audio_base64 || '');
        this.stats.ttsBytes += raw.length;
        if (this.player) {
          const ok = this.player.playBase64
            ? this.player.playBase64(msg.audio_base64, msg.sample_rate)
            : null;
          // 一旦起播就回报一次 started（驱动云端 AEC 的参考时钟）
          if (!this._playbackStarted[msg.response_id]) {
            this._playbackStarted[msg.response_id] = true;
            this._sendPlayback(msg.response_id, 'started', 0);
          }
        }
      } else if (t === 'tts.end') {
        if (this.player && this.player.endResponse) {
          this.player.endResponse(msg.response_id);
        }
        this._sendPlayback(msg.response_id, 'ended', 0);
      } else if (t === 'tts.cancel') {
        if (this.player && this.player.stop) this.player.stop();
        // 取消必须立即回执 —— 云端据此截断 AEC 参考轨
        this._sendPlayback(msg.response_id, 'cancelled', 0);
      }
      this._emit(msg);
    }

    /** 发送播放回执。ctx_time 用 AudioContext 当前时间。 */
    _sendPlayback(responseId, phase, sampleOffset) {
      const ctxTime =
        this.player && this.player.now ? this.player.now() : 0;
      this._send({
        type: 'playback',
        response_id: responseId,
        phase: phase,
        ctx_time: ctxTime,
        seq: 0,
        sample_offset: sampleOffset || 0,
      });
    }

    // ------------------------------------------------------------------ //

    /** 送一个采集块。音频为 Float32Array（16kHz 单声道）。 */
    sendChunk(chunk) {
      if (!this.ready || this.closed) return;
      const audio = chunk.audio;
      const buf = new ArrayBuffer(audio.length * 4);
      new Float32Array(buf).set(audio);
      this._send({
        type: 'audio',
        audio_base64: arrayBufferToBase64(buf),
        t_ms: 0,
      });
      this.stats.audioSent += 1;

      if (chunk.frameBase64) {
        this._send({ type: 'video_face', frame_base64: chunk.frameBase64, t_ms: 0 });
        this.stats.faceFramesSent += 1;
      }
      if (chunk.frameOmniBase64) {
        this._send({ type: 'video_omni', frame_base64: chunk.frameOmniBase64, t_ms: 0 });
        this.stats.omniFramesSent += 1;
      }
    }

    /** 主动请求中断当前播报（barge-in）。 */
    cancel(reason) {
      this._send({ type: 'cancel', reason: reason || 'bargein' });
    }

    /** 发送任意控制消息（校准、切换 AEC 模式等）。 */
    send(obj) {
      return this._send(obj);
    }

    /** 优雅停止：先让服务端收尾（等 ASR 最终结果、TTS 送达），再关闭。 */
    stop(reason) {
      if (this.closed) return Promise.resolve();
      const self = this;
      return new Promise(function (resolve) {
        if (!self.ready) {
          if (self.ws) self.ws.close();
          resolve();
          return;
        }
        try {
          self._send({ type: 'session.stop' });
        } catch (e) {
          /* ignore */
        }
        // 给服务端收尾时间（它要等 ASR 最终结果并推完 TTS 音频）
        setTimeout(function () {
          if (self.ws) self.ws.close(1000, reason || 'client_stop');
          resolve();
        }, 3000);
      });
    }
  }

  /** 与 duplex-utils 的 arrayBufferToBase64 等价（避免强依赖加载顺序）。 */
  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + chunk)
      );
    }
    return btoa(binary);
  }

  global.OrchestratorSession = OrchestratorSession;
})(window);
