const test = require("node:test");
const assert = require("node:assert/strict");

const {
  RealtimePlaybackController,
  RealtimeTurnTracker,
} = require("./playback.js");

class FakeSource {
  constructor() {
    this.stopped = false;
    this.onended = null;
  }

  connect() {}
  start() {}
  stop() {
    this.stopped = true;
  }
}

class FakeContext {
  constructor() {
    this.state = "running";
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
    this.closed = false;
  }

  createBuffer(_channels, length, sampleRate) {
    return { duration: length / sampleRate, copyToChannel() {} };
  }

  createBufferSource() {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }

  close() {
    this.state = "closed";
    this.closed = true;
  }

  resume() {}
}

function controller() {
  const contexts = [];
  const playback = new RealtimePlaybackController({
    createAudioContext: () => {
      const context = new FakeContext();
      contexts.push(context);
      return context;
    },
    decodeBase64: () => new Uint8Array([0, 0, 1, 0]),
  });
  return { playback, contexts };
}

test("interrupt stops playback and rejects late audio", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("response-a");
  assert.equal(playback.queueAudioDelta("audio", "response-a"), true);

  assert.deepEqual(playback.interrupt(), ["response-a"]);
  assert.equal(contexts[0].sources[0].stopped, true);
  assert.equal(contexts[0].closed, false);
  assert.equal(playback.queueAudioDelta("late", "response-a"), false);
});

test("speech can interrupt playback after response generation completed", () => {
  const { playback } = controller();
  playback.beginResponse("response-a");
  playback.queueAudioDelta("audio", "response-a");
  playback.finishResponse("response-a", "completed");

  assert.deepEqual(playback.interrupt(), ["response-a"]);
});

test("a new response can play after an interrupted response", () => {
  const { playback, contexts } = controller();
  playback.beginResponse("response-a");
  playback.queueAudioDelta("audio", "response-a");
  playback.interrupt();
  playback.beginResponse("response-b");

  assert.equal(playback.queueAudioDelta("audio", "response-b"), true);
  assert.equal(contexts.length, 1);
});

test("explicit close releases the audio context", () => {
  const { playback, contexts } = controller();
  playback.ensureContext();

  playback.close();

  assert.equal(contexts[0].closed, true);
});

test("pending-start interruption marks the later response stale", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");

  const binding = turns.beginResponse("response-a", true);

  assert.deepEqual(binding, { itemId: "item-a", interrupted: true });
  assert.equal(turns.ownsResponse("response-a"), false);
  assert.deepEqual(turns.finishResponse("response-a"), {
    itemId: "item-a",
    interrupted: true,
  });
});

test("a response delayed until speech stops remains live", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");

  assert.deepEqual(turns.beginResponse("response-a"), {
    itemId: "item-a",
    interrupted: false,
  });
  assert.equal(turns.ownsResponse("response-a"), true);
});

test("active interruption retains response-to-turn correlation", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-a");
  turns.beginResponse("response-a");

  assert.equal(turns.interruptResponse("response-a"), "item-a");
  assert.deepEqual(turns.finishResponse("response-a"), {
    itemId: "item-a",
    interrupted: true,
  });
});

test("a stale terminal cannot finish a newer live response", () => {
  const turns = new RealtimeTurnTracker();
  turns.commit("item-live");
  turns.beginResponse("response-live");

  assert.deepEqual(turns.finishResponse("response-stale"), {
    itemId: null,
    interrupted: false,
  });
  assert.equal(turns.ownsResponse("response-live"), true);
});
