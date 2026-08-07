(function (global) {
  "use strict";

  class RealtimePlaybackController {
    constructor(options = {}) {
      this.sampleRate = options.sampleRate || 24000;
      this.createAudioContext =
        options.createAudioContext ||
        (() => new (global.AudioContext || global.webkitAudioContext)());
      this.decodeBase64 =
        options.decodeBase64 ||
        ((encoded) => {
          const binary = global.atob(encoded);
          return Uint8Array.from(binary, (value) => value.charCodeAt(0));
        });
      this.context = null;
      this.nextPlaybackTime = 0;
      this.activeResponseId = null;
      this.playbackResponseId = null;
      this.cancelledResponseIds = new Set();
      this.sources = new Set();
    }

    ensureContext() {
      if (!this.context || this.context.state === "closed") {
        this.context = this.createAudioContext();
        this.nextPlaybackTime = 0;
      }
      if (this.context.state === "suspended") this.context.resume();
      return this.context;
    }

    beginResponse(responseId) {
      this.activeResponseId = responseId || null;
    }

    finishResponse(responseId, status) {
      if (status !== "completed") {
        this.rejectResponse(responseId);
      }
      if (responseId && responseId === this.activeResponseId) {
        this.activeResponseId = null;
      }
    }

    rejectResponse(responseId) {
      if (!responseId) return;
      this.cancelledResponseIds.add(responseId);
      if (responseId === this.activeResponseId) this.activeResponseId = null;
      if (responseId === this.playbackResponseId) this.flush();
      while (this.cancelledResponseIds.size > 64) {
        this.cancelledResponseIds.delete(
          this.cancelledResponseIds.values().next().value,
        );
      }
    }

    interrupt() {
      const responseIds = new Set(
        [this.activeResponseId, this.playbackResponseId].filter(Boolean),
      );
      responseIds.forEach((responseId) => this.rejectResponse(responseId));
      this.activeResponseId = null;
      this.flush();
      return [...responseIds];
    }

    queueAudioDelta(encoded, responseId) {
      if (!responseId || this.cancelledResponseIds.has(responseId)) return false;
      const bytes = this.decodeBase64(encoded);
      if (bytes.byteLength % 2 !== 0) {
        throw new Error("Received an odd-length PCM16 audio delta");
      }
      if (this.cancelledResponseIds.has(responseId)) return false;

      const context = this.ensureContext();
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const samples = new Float32Array(bytes.byteLength / 2);
      for (let index = 0; index < samples.length; index++) {
        samples[index] = view.getInt16(index * 2, true) / 0x8000;
      }
      const buffer = context.createBuffer(1, samples.length, this.sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => {
        this.sources.delete(source);
        if (
          this.sources.size === 0 &&
          this.playbackResponseId === responseId
        ) {
          this.playbackResponseId = null;
        }
      };

      const startAt = Math.max(
        context.currentTime + 0.02,
        this.nextPlaybackTime,
      );
      this.playbackResponseId = responseId;
      this.sources.add(source);
      source.start(startAt);
      this.nextPlaybackTime = startAt + buffer.duration;
      return true;
    }

    flush() {
      this.sources.forEach((source) => {
        try {
          source.stop();
        } catch (_) {}
      });
      this.sources.clear();
      if (this.context) {
        this.context.close();
        this.context = null;
      }
      this.playbackResponseId = null;
      this.nextPlaybackTime = 0;
    }
  }

  class RealtimeTurnTracker {
    constructor() {
      this.pendingItemIds = [];
      this.respondingItemId = null;
      this.activeResponseId = null;
      this.responseItems = new Map();
      this.staleResponseIds = new Set();
    }

    commit(itemId) {
      if (itemId) this.pendingItemIds.push(itemId);
    }

    hasPendingResponse() {
      return this.pendingItemIds.length > 0;
    }

    beginResponse(responseId, interrupted = false) {
      const itemId = this.pendingItemIds.shift() || null;
      if (!responseId || !itemId) {
        if (responseId) this.staleResponseIds.add(responseId);
        return { itemId, interrupted: true };
      }
      this.responseItems.set(responseId, itemId);
      if (interrupted) {
        this.staleResponseIds.add(responseId);
      } else {
        this.activeResponseId = responseId;
        this.respondingItemId = itemId;
      }
      while (this.responseItems.size > 64) {
        this.responseItems.delete(this.responseItems.keys().next().value);
      }
      return { itemId, interrupted };
    }

    interruptResponse(responseId) {
      if (!responseId) return null;
      this.staleResponseIds.add(responseId);
      const itemId = this.responseItems.get(responseId) || null;
      if (responseId === this.activeResponseId) {
        this.activeResponseId = null;
        this.respondingItemId = null;
      }
      return itemId;
    }

    ownsResponse(responseId) {
      return Boolean(responseId) && responseId === this.activeResponseId;
    }

    finishResponse(responseId) {
      const itemId = this.responseItems.get(responseId) || null;
      const interrupted = this.staleResponseIds.has(responseId);
      if (responseId === this.activeResponseId) {
        this.activeResponseId = null;
        this.respondingItemId = null;
      }
      this.staleResponseIds.delete(responseId);
      return { itemId, interrupted };
    }

    clear() {
      this.pendingItemIds.length = 0;
      this.respondingItemId = null;
      this.activeResponseId = null;
      this.responseItems.clear();
      this.staleResponseIds.clear();
    }
  }

  global.RealtimePlaybackController = RealtimePlaybackController;
  global.RealtimeTurnTracker = RealtimeTurnTracker;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { RealtimePlaybackController, RealtimeTurnTracker };
  }
})(typeof window !== "undefined" ? window : globalThis);
