/**
 * lib/session-video-recorder.js — Video session recorder (WebM/MP4)
 *
 * Records a duplex session into a video file with stereo audio:
 *   Video   = captured from <video> element (camera or file playback)
 *   Audio L = user input mix (16kHz from CaptureProcessor)
 *   Audio R = AI response audio (resampled to 16kHz)
 *
 * Cross-platform video capture strategy (3-tier fallback):
 *   1. videoEl.captureStream()        — Chrome, Firefox, Android
 *   2. videoEl.srcObject video track  — iOS Live mode (camera stream)
 *   3. Canvas drawImage loop          — iOS File mode (universal fallback)
 *
 * Subtitle compositing: when enabled, forces canvas path and renders AI response
 * text as burned-in captions at the bottom of each frame (auto-wrapped, semi-transparent bg).
 *
 * Codec selection: VP9/VP8 (WebM) → H.264 (MP4, Safari) → audio-only fallback.
 * Memory-efficient: encoded chunks flushed every 1 second via timeslice.
 *
 * API is compatible with SessionRecorder (pushLeft/pushRight/pause/resume/stop).
 */

import { resampleAudio } from './duplex-utils.js';

export class SessionVideoRecorder {
    /**
     * @param {HTMLVideoElement} videoEl - The video element to capture
     * @param {number} [inputSR=16000] - Sample rate of left channel (CaptureProcessor output)
     * @param {number} [aiSR=24000] - Sample rate of AI audio (AudioPlayer raw output)
     */
    constructor(videoEl, inputSR = 16000, aiSR = 24000) {
        this._videoEl = videoEl;
        this._inputSR = inputSR;
        this._aiSR = aiSR;
        this._recording = false;
        this._paused = false;
        this._mediaRecorder = null;
        /** @type {Blob[]} */
        this._chunks = [];
        this._recCtx = null;
        this._stereoNode = null;
        this._startTime = 0;

        // Canvas compositing state (subtitle overlay or iOS fallback)
        this._canvasComposite = null; // { canvas, ctx, animFrame }

        // Subtitle state
        this._subtitleEnabled = false;
        /** @type {Array<{text: string, active: boolean}>} */
        this._subtitleMessages = [];
        this._subtitleHeight = 25;          // % of canvas height from bottom
        this._subtitleOpacityBottom = 0.9;
        this._subtitleOpacityTop = 0.15;
    }

    get recording() { return this._recording; }

    /**
     * Start recording. Call after media provider is started (video element active).
     * @param {Object} [settings] - Optional recording settings from RecordingSettings.getSettings()
     * @param {string} [settings.mimeType='auto'] - MIME type ('auto' uses built-in detection)
     * @param {number} [settings.videoBitsPerSecond=2000000] - Video bitrate in bps
     * @param {boolean} [settings.subtitle=false] - Burn AI subtitles into video via canvas compositing
     * @param {number} [settings.subtitleHeight=25] - Subtitle area as % of video height from bottom
     * @param {number} [settings.subtitleOpacityBottom=0.9] - Bottom message opacity (0-1)
     * @param {number} [settings.subtitleOpacityTop=0.15] - Top message opacity (0-1)
     * @param {Object} [leftAudioOptions] - Direct left audio source (bypasses 1s CaptureProcessor delay)
     * @param {MediaStream} [leftAudioOptions.leftAudioStream] - Real-time audio stream (Live/mic/mixed modes)
     * @param {Float32Array} [leftAudioOptions.leftAudioBuffer] - Pre-decoded full audio (video-only File mode)
     */
    async start(settings = {}, leftAudioOptions = {}) {
        const {
            mimeType: requestedMime = 'auto',
            videoBitsPerSecond = 2_000_000,
            subtitle = false,
            subtitleHeight = 25,
            subtitleOpacityBottom = 0.9,
            subtitleOpacityTop = 0.15,
        } = settings;
        const { leftAudioStream = null, leftAudioBuffer = null } = leftAudioOptions;

        this._subtitleEnabled = subtitle;
        this._subtitleMessages = [];
        this._subtitleHeight = subtitleHeight;
        this._subtitleOpacityBottom = subtitleOpacityBottom;
        this._subtitleOpacityTop = subtitleOpacityTop;

        // 1. Acquire video track (force canvas when subtitle is enabled for compositing)
        const { track: videoTrack, method } = subtitle
            ? this._acquireCanvasTrack()
            : this._acquireVideoTrack();

        // 2. Create recording AudioContext + stereo mixing worklet
        this._recCtx = new AudioContext({ sampleRate: this._inputSR });
        if (this._recCtx.state === 'suspended') await this._recCtx.resume();

        await this._recCtx.audioWorklet.addModule('/static/duplex/lib/stereo-recorder-processor.js');
        this._stereoNode = new AudioWorkletNode(this._recCtx, 'stereo-recorder-processor', {
            numberOfInputs: 2,    // input 0: left channel, input 1: right channel
            outputChannelCount: [2],
        });

        // 2b. Connect direct left audio source (zero-delay, bypasses CaptureProcessor 1s chunking)
        this._directLeft = false;
        this._leftSource = null;
        this._leftBufSource = null;

        if (leftAudioStream) {
            this._leftSource = this._recCtx.createMediaStreamSource(leftAudioStream);
            this._leftSource.connect(this._stereoNode);
            this._stereoNode.port.postMessage({ command: 'useInputLeft' });
            this._directLeft = true;
        } else if (leftAudioBuffer && leftAudioBuffer.length > 0) {
            const buf = this._recCtx.createBuffer(1, leftAudioBuffer.length, this._inputSR);
            buf.getChannelData(0).set(leftAudioBuffer);
            this._leftBufSource = this._recCtx.createBufferSource();
            this._leftBufSource.buffer = buf;
            this._leftBufSource.connect(this._stereoNode);
            this._stereoNode.port.postMessage({ command: 'useInputLeft' });
            this._directLeft = true;
        }

        // 2c. Setup right channel: sample-accurate AudioBufferSourceNode scheduling
        //     Mirrors AudioPlayer's scheduling — no postMessage jitter, no queue-drain gaps.
        this._rightNextTime = 0;
        this._rightSources = [];
        this._stereoNode.port.postMessage({ command: 'useInputRight' });

        // 3. Connect to MediaStreamDestination for stereo audio track
        const dest = this._recCtx.createMediaStreamDestination();
        this._stereoNode.connect(dest);

        // 4. Combine video + stereo audio into one MediaStream
        const tracks = [...dest.stream.getAudioTracks()];
        if (videoTrack) tracks.unshift(videoTrack);
        const combinedStream = new MediaStream(tracks);

        // 5. Select codec: user-specified or auto-detect
        const mimeType = (requestedMime === 'auto')
            ? this._selectMimeType(!!videoTrack)
            : requestedMime;

        // 6. Create MediaRecorder
        const options = { mimeType };
        if (videoTrack) options.videoBitsPerSecond = videoBitsPerSecond;
        this._mediaRecorder = new MediaRecorder(combinedStream, options);

        this._chunks = [];
        this._mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this._chunks.push(e.data);
        };

        this._mediaRecorder.start(1000); // 1-second timeslice for memory efficiency

        // Start pre-buffered audio playback (video-only File mode)
        if (this._leftBufSource) {
            this._leftBufSource.start();
        }

        this._recording = true;
        this._paused = false;
        this._startTime = performance.now();
        this._rightNextTime = this._recCtx.currentTime;

        const leftMode = leftAudioStream ? 'stream' : leftAudioBuffer ? 'buffer' : 'pushLeft';
        console.log(`[SessionVideoRecorder] started (${mimeType}, video=${method}, left=${leftMode}, right=scheduled, ${(videoBitsPerSecond / 1_000_000).toFixed(0)} Mbps)`);
    }

    /**
     * Push left-channel chunk (user input mix from CaptureProcessor).
     * No-op when direct audio mode is active (leftAudioStream/leftAudioBuffer).
     * @param {Float32Array} pcm16k
     */
    pushLeft(pcm16k) {
        if (!this._recording || this._paused || this._directLeft) return;
        this._stereoNode?.port.postMessage({ channel: 'left', audio: pcm16k });
    }

    /**
     * Push right-channel chunk (AI response from AudioPlayer.onRawAudio).
     * Uses AudioBufferSourceNode scheduling for sample-accurate placement,
     * mirroring AudioPlayer's gap-handling logic. Eliminates postMessage
     * jitter and queue-drain silence gaps.
     *
     * @param {Float32Array} samples - Raw PCM at aiSR
     * @param {number} sampleRate - Source sample rate (typically 24000)
     * @param {number} timestamp - performance.now() timestamp (kept for API compat)
     */
    pushRight(samples, sampleRate, timestamp) {
        if (!this._recording || this._paused || !this._recCtx) return;
        const resampled = resampleAudio(samples, sampleRate, this._inputSR);

        // Create AudioBufferSourceNode for sample-accurate scheduling
        const audioBuf = this._recCtx.createBuffer(1, resampled.length, this._inputSR);
        audioBuf.getChannelData(0).set(resampled);
        const src = this._recCtx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(this._stereoNode, 0, 1); // output 0 → stereoNode input 1 (right)

        // Schedule: gapless back-to-back; shift forward on underrun (mirrors AudioPlayer)
        const now = this._recCtx.currentTime;
        if (this._rightNextTime < now) {
            this._rightNextTime = now;
        }
        src.start(this._rightNextTime);
        this._rightNextTime += audioBuf.duration;

        // Auto-cleanup when chunk finishes
        this._rightSources.push(src);
        src.onended = () => {
            const idx = this._rightSources.indexOf(src);
            if (idx >= 0) this._rightSources.splice(idx, 1);
        };
    }

    /**
     * Update subtitle text for the current (active) turn.
     * Creates a new message entry on first call after finalize/start.
     * Subsequent calls update the same entry (streaming text).
     * @param {string} text
     */
    setSubtitleText(text) {
        if (!this._subtitleEnabled) return;
        const msgs = this._subtitleMessages;
        const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
        if (last && last.active) {
            last.text = text;   // 同一轮内流式更新
        } else {
            // 新一轮开始：清掉上一轮的消息，只显示当前这轮（避免多轮堆叠挤压字号）
            msgs.length = 0;
            msgs.push({ text, active: true });
        }
    }

    /**
     * Finalize current subtitle turn — text stays visible and gradually fades
     * as new messages push it upward (like fullscreen chat overlay).
     * Call from session.onSpeakEnd.
     */
    finalizeSubtitle() {
        if (!this._subtitleEnabled) return;
        const msgs = this._subtitleMessages;
        const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
        if (last && last.active) {
            last.active = false;
        }
    }

    /**
     * Pause recording. MediaRecorder pauses encoding; AudioContext suspended
     * to keep AudioBufferSource playback position aligned with recording timeline.
     */
    pause() {
        if (!this._recording || this._paused) return;
        this._paused = true;
        if (this._mediaRecorder?.state === 'recording') {
            this._mediaRecorder.pause();
        }
        if (this._recCtx?.state === 'running') {
            this._recCtx.suspend();
        }
        console.log('[SessionVideoRecorder] paused');
    }

    /**
     * Resume recording after pause.
     */
    resume() {
        if (!this._recording || !this._paused) return;
        this._paused = false;
        if (this._recCtx?.state === 'suspended') {
            this._recCtx.resume().then(() => {
                if (this._mediaRecorder?.state === 'paused') {
                    this._mediaRecorder.resume();
                }
            });
        } else {
            if (this._mediaRecorder?.state === 'paused') {
                this._mediaRecorder.resume();
            }
        }
        console.log('[SessionVideoRecorder] resumed');
    }

    /**
     * Stop recording and return video blob.
     * @returns {Promise<{blob: Blob, durationSec: number}>}
     */
    stop() {
        if (!this._recording) {
            return Promise.resolve({ blob: new Blob(), durationSec: 0 });
        }
        this._recording = false;
        this._paused = false;

        // Stop canvas draw loop
        if (this._canvasComposite) {
            cancelAnimationFrame(this._canvasComposite.animFrame);
            this._canvasComposite = null;
        }

        return new Promise((resolve) => {
            this._mediaRecorder.onstop = () => {
                const mimeType = this._mediaRecorder.mimeType || 'video/webm';
                const blob = new Blob(this._chunks, { type: mimeType });
                const durationSec = (performance.now() - this._startTime) / 1000;

                console.log(`[SessionVideoRecorder] encoded: ${durationSec.toFixed(1)}s, ` +
                    `${(blob.size / 1024 / 1024).toFixed(1)} MB (${mimeType})`);

                // Cleanup
                if (this._leftSource) { this._leftSource.disconnect(); this._leftSource = null; }
                if (this._leftBufSource) { try { this._leftBufSource.stop(); } catch (_) {} this._leftBufSource = null; }
                this._directLeft = false;
                for (const src of this._rightSources) {
                    try { src.stop(); } catch (_) {}
                    try { src.disconnect(); } catch (_) {}
                }
                this._rightSources = [];
                this._stereoNode?.disconnect();
                this._stereoNode = null;
                this._recCtx?.close().catch(() => {});
                this._recCtx = null;
                this._chunks = [];
                this._mediaRecorder = null;

                resolve({ blob, durationSec });
            };

            // Flush any remaining encoded data
            try { this._mediaRecorder.requestData(); } catch (_) {}
            this._mediaRecorder.stop();
        });
    }

    // ==================== Video Track Acquisition (3-tier fallback) ====================

    /**
     * Acquire a video MediaStreamTrack from the video element.
     *
     * Strategy (ordered by preference):
     *   1. captureStream()  — native, works on Chrome/Firefox/Android
     *   2. srcObject clone   — iOS Live mode (camera already is a MediaStream)
     *   3. Canvas fallback   — iOS File mode (drawImage loop + canvas.captureStream)
     *
     * @returns {{ track: MediaStreamTrack|null, method: string }}
     */
    _acquireVideoTrack() {
        // Tier 1: HTMLMediaElement.captureStream() — Chrome, Firefox, Android
        const captureFn = this._videoEl.captureStream || this._videoEl.mozCaptureStream;
        if (captureFn) {
            try {
                const stream = captureFn.call(this._videoEl);
                const track = stream.getVideoTracks()[0];
                if (track) return { track, method: 'captureStream' };
            } catch (e) {
                console.warn('[SessionVideoRecorder] captureStream failed:', e.message);
            }
        }

        // Tier 2: srcObject video track — iOS Live mode (camera MediaStream)
        if (this._videoEl.srcObject instanceof MediaStream) {
            const track = this._videoEl.srcObject.getVideoTracks()[0];
            if (track) {
                // Clone to avoid stopping the original camera track on recorder stop
                return { track: track.clone(), method: 'srcObject' };
            }
        }

        // Tier 3: Canvas fallback — iOS File mode / universal last resort
        return this._acquireCanvasTrack();
    }

    /**
     * Canvas-based video track acquisition. Used as:
     *   - Fallback for iOS File mode (Tier 3)
     *   - Primary path when subtitle compositing is enabled
     *
     * Draws video + optional subtitle text to an offscreen canvas at ~30fps,
     * then captures via canvas.captureStream().
     * @returns {{ track: MediaStreamTrack|null, method: string }}
     */
    _acquireCanvasTrack() {
        if (typeof HTMLCanvasElement.prototype.captureStream !== 'function') {
            console.warn('[SessionVideoRecorder] No canvas.captureStream, audio-only');
            return { track: null, method: 'none' };
        }

        const canvas = document.createElement('canvas');
        // Initial size from video; auto-resizes when actual dimensions become available
        canvas.width = this._videoEl.videoWidth || 1;
        canvas.height = this._videoEl.videoHeight || 1;
        const ctx = canvas.getContext('2d');

        let fontSize = this._calcSubtitleFontSize(canvas.width, canvas.height);

        // Store composite state first so drawLoop condition works on first frame
        this._canvasComposite = { canvas, ctx, animFrame: 0 };

        const drawLoop = () => {
            if (!this._canvasComposite) return;
            // Dynamic resize: adapt canvas to video's native dimensions (handles metadata race)
            const vw = this._videoEl.videoWidth;
            const vh = this._videoEl.videoHeight;
            if (vw > 0 && vh > 0 && (canvas.width !== vw || canvas.height !== vh)) {
                canvas.width = vw;
                canvas.height = vh;
                fontSize = this._calcSubtitleFontSize(vw, vh);
                console.log(`[SessionVideoRecorder] canvas resized to ${vw}×${vh}`);
            }
            // Draw video frame (canvas matches video dimensions — no stretching)
            ctx.drawImage(this._videoEl, 0, 0, canvas.width, canvas.height);
            // Draw subtitle overlay if enabled and messages exist
            if (this._subtitleEnabled && this._subtitleMessages.length > 0) {
                this._drawSubtitleStack(ctx, canvas.width, canvas.height, fontSize);
            }
            this._canvasComposite.animFrame = requestAnimationFrame(drawLoop);
        };
        drawLoop();

        const stream = canvas.captureStream(30);
        const track = stream.getVideoTracks()[0];
        if (track) {
            return { track, method: this._subtitleEnabled ? 'canvas+subtitle' : 'canvas' };
        }
        return { track: null, method: 'none' };
    }

    /**
     * Calculate adaptive subtitle font size based on canvas dimensions.
     * Considers both dimensions so portrait mode doesn't get oversized text.
     * @param {number} w - Canvas width
     * @param {number} h - Canvas height
     * @returns {number}
     */
    _calcSubtitleFontSize(w, h) {
        return Math.max(14, Math.min(Math.round(h / 25), Math.round(w / 18)));
    }

    // ==================== Subtitle Rendering ====================

    /**
     * Draw stacked subtitle messages from bottom to top with gradient opacity.
     * Mimics fullscreen chat overlay: newest message at bottom (opaque),
     * older messages pushed upward and fading toward transparent.
     *
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} w - Canvas width
     * @param {number} h - Canvas height
     * @param {number} fontSize
     */
    _drawSubtitleStack(ctx, w, h, fontSize) {
        const msgs = this._subtitleMessages;
        if (msgs.length === 0) return;

        // 字幕区（从底部 subtitleHeight% 处向上）—— 长文本完整显示都装在这里
        const bottomMargin = Math.round(fontSize * 0.6);
        const subtitleFloor = h - bottomMargin;
        const subtitleCeiling = h - Math.round(h * this._subtitleHeight / 100);
        const areaHeight = Math.max(10, subtitleFloor - subtitleCeiling);

        // 只显示当前轮：取最后一条（setSubtitleText 已清空旧轮，msgs 只有当前这轮）。
        // 若最后一轮已固化（说话结束到下一轮间隙），仍保留显示直到新轮到来。
        const texts = [];
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].text) {
                texts.push({ text: msgs[i].text, active: !!msgs[i].active });
                break;   // 只显示最后一条（当前轮）
            }
        }
        if (texts.length === 0) return;

        // ── 字号自适应：保证最长文本能在字幕区完整装下 ──
        // 先用基础字号测每行能放多少字，若总行数超出字幕区可容纳行数，等比缩小字号。
        // 逐级估算避免多次 measureText（成本高）。中文/英文混合按平均字宽近似。
        let size = fontSize;
        const longest = texts.reduce((a, b) => (b.text.length > a.text.length ? b : a), texts[0]);
        // 最多重试几次缩小，防止极端长文本缩到过小（下限 11px）
        for (let attempt = 0; attempt < 5; attempt++) {
            const pad = Math.round(size * 0.5);
            const lineH = Math.round(size * 1.4);
            const maxTextW = w - pad * 6;
            ctx.font = `${size}px -apple-system, "Segoe UI", Roboto, sans-serif`;
            // 估算每行字符数：CJK 每字≈size，拉丁≈size*0.55，取保守 0.65
            const charsPerLine = Math.max(1, Math.floor(maxTextW / (size * 0.65)));
            // 换行（用真实 measure 更准，但几百字逐字测成本高；折中用每行 cap）
            const lineCount = texts.reduce((acc, t) => {
                const rows = Math.max(1, Math.ceil(t.text.length / charsPerLine));
                return acc + rows;
            }, 0);
            const totalBlockH = texts.length * lineH * 0 + lineCount * lineH + pad * 2 * texts.length;
            if (totalBlockH <= areaHeight || size <= 11) break;
            size = Math.max(11, Math.round(size * areaHeight / Math.max(1, totalBlockH)));
        }

        // 用最终字号做真实换行（不限行数，完整显示）
        const padding = Math.round(size * 0.5);
        const lineHeight = Math.round(size * 1.4);
        const maxTextWidth = w - padding * 6;
        const msgGap = Math.round(size * 0.35);
        const radius = Math.round(size * 0.4);
        const maxBgWidth = Math.min(w - padding * 2, maxTextWidth + padding * 3);
        ctx.font = `${size}px -apple-system, "Segoe UI", Roboto, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // 每条换行（不限行数，完整显示几百字）
        const blocks = [];
        for (const t of texts) {
            const lines = this._wrapText(ctx, t.text, maxTextWidth, 0);   // maxLines=0 不截断
            blocks.push({ lines, blockH: lines.length * lineHeight + padding * 2, active: t.active });
        }

        // 从底部往上画，超出字幕区天花板则停止（不再硬截断单条文本）
        const opacityAtY = (y) => {
            const tt = areaHeight > 0 ? Math.max(0, Math.min(1, (subtitleFloor - y) / areaHeight)) : 0;
            return this._subtitleOpacityBottom + tt * (this._subtitleOpacityTop - this._subtitleOpacityBottom);
        };

        let curY = subtitleFloor;
        for (let bi = blocks.length - 1; bi >= 0; bi--) {
            const block = blocks[bi];
            const blockTop = curY - block.blockH;
            if (blockTop < subtitleCeiling) break;   // 超出字幕区则不画（但单条已完整换行不会截断）

            const bgX = (w - maxBgWidth) / 2;
            const textStartY = blockTop + padding;
            const midY = blockTop + block.blockH / 2;
            const bgOpacity = opacityAtY(midY);
            ctx.fillStyle = `rgba(0, 0, 0, ${(0.55 * bgOpacity).toFixed(2)})`;
            this._roundRect(ctx, bgX, blockTop, maxBgWidth, block.blockH, radius);
            ctx.fill();

            for (let li = 0; li < block.lines.length; li++) {
                const lineY = textStartY + li * lineHeight;
                const lineOpacity = opacityAtY(lineY);
                ctx.fillStyle = `rgba(255, 255, 255, ${lineOpacity.toFixed(2)})`;
                ctx.fillText(block.lines[li], w / 2, lineY);
            }
            curY = blockTop - msgGap;
        }
    }

    /**
     * Wrap text into lines that fit within maxWidth.
     * Handles both CJK (character-by-character) and Latin (word-by-word) wrapping.
     * @param {CanvasRenderingContext2D} ctx
     * @param {string} text
     * @param {number} maxWidth
     * @returns {string[]}
     */
    _wrapText(ctx, text, maxWidth, maxLines = 0) {
        const lines = [];
        let current = '';

        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            const test = current + ch;
            if (ctx.measureText(test).width > maxWidth && current.length > 0) {
                lines.push(current);
                current = ch;
            } else {
                current = test;
            }
        }
        if (current) lines.push(current);

        // maxLines>0 时截断到最近 maxLines 行（旧逻辑 4 行硬截断会丢长文本内容）；
        // maxLines=0 表示不限行数（供长字幕完整显示，配合字号自适应控制高度）。
        if (maxLines > 0 && lines.length > maxLines) {
            return lines.slice(lines.length - maxLines);
        }
        return lines;
    }

    /**
     * Draw a rounded rectangle path.
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     * @param {number} r
     */
    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    // ==================== MIME Type Selection ====================

    /**
     * Select the best supported MIME type for MediaRecorder.
     * Tries WebM (VP9/VP8) first, then MP4 (H.264, Safari), then audio-only.
     * @param {boolean} hasVideo
     * @returns {string}
     */
    _selectMimeType(hasVideo) {
        // Safari UA contains "Safari" but NOT "Chrome"/"Chromium".
        // Chrome UA contains both "Chrome" and "Safari" (legacy compat).
        const isSafari = /Safari/.test(navigator.userAgent)
            && !/Chrome/.test(navigator.userAgent)
            && !/Chromium/.test(navigator.userAgent);

        if (hasVideo) {
            if (isSafari) {
                // Safari/iOS: prefer MP4 for native playback (Photos, Files, QuickTime).
                // WebM records fine on modern Safari but cannot be played natively on Apple platforms.
                if (MediaRecorder.isTypeSupported('video/mp4')) return 'video/mp4';
            }
            // Chrome/Firefox/Android: prefer WebM (native, proven stereo audio support)
            const webmCandidates = [
                'video/webm;codecs=vp9,opus',
                'video/webm;codecs=vp8,opus',
                'video/webm',
            ];
            for (const mime of webmCandidates) {
                if (MediaRecorder.isTypeSupported(mime)) return mime;
            }
            // Last resort: try mp4 anyway
            if (MediaRecorder.isTypeSupported('video/mp4')) return 'video/mp4';
        }
        // Audio-only fallback
        if (isSafari && MediaRecorder.isTypeSupported('audio/mp4')) return 'audio/mp4';
        const audioCandidates = ['audio/webm;codecs=opus', 'audio/webm'];
        for (const mime of audioCandidates) {
            if (MediaRecorder.isTypeSupported(mime)) return mime;
        }
        if (MediaRecorder.isTypeSupported('audio/mp4')) return 'audio/mp4';
        return '';
    }
}
