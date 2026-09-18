import assert from "node:assert/strict";

import {
  ReviewPanelsController,
  nearestSample,
} from "../../robot_sf/render/web_assets/components/review_panels/review_panels.js";

class FakeElement {
  constructor(ownerDocument, tagName) {
    this.ownerDocument = ownerDocument;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.textContent = "";
    this.value = "";
    this.currentTime = 0;
    this.paused = true;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = children;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    this.listeners.set(type, listeners.filter((candidate) => candidate !== listener));
  }

  emit(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) listener({ target: this, ...event });
  }

  setAttribute() {}

  remove() {}

  getContext() {
    return new Proxy(
      {},
      {
        get: () => () => {},
      },
    );
  }

  play() {
    this.paused = false;
    return Promise.resolve();
  }

  pause() {
    this.paused = true;
  }
}

class FakeDocument {
  constructor() {
    this.listeners = new Map();
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    this.listeners.set(type, listeners.filter((candidate) => candidate !== listener));
  }
}

function findAll(element, predicate) {
  const result = [];
  if (predicate(element)) result.push(element);
  for (const child of element.children || []) result.push(...findAll(child, predicate));
  return result;
}

function model() {
  const sceneSamples = [
    { time_s: 0, value: { robot: { position: [0, 0] } } },
    { time_s: 1, value: { robot: { position: [1, 0] } } },
    { time_s: 2, value: { robot: { position: [2, 0] } } },
    { time_s: 4, value: { robot: { position: [4, 0] } } },
  ];
  const metricSamples = [
    { time_s: 0, value: { value: 1.0 } },
    { time_s: 2, value: { value: 3.0 } },
  ];
  return {
    schema_version: "review-panels.v1",
    time: { origin_s: 0, terminal_s: 4, cursor: { time_s: 0 } },
    context: { episode_id: "episode-1", execution_id: "execution-1", actor_id: "ego-1" },
    source_identity: { scene: { uri: "scene.json" }, video: { uri: "clip.mp4" } },
    scene_surface: { schema_version: "threejs-viewer.v1", map: { width: 5, height: 5 } },
    goal_geometry: {
      point: { status: "available", value: [4, 0] },
      completion_boundary: { status: "unavailable", value: null },
    },
    streams: {
      scene: { status: "available", resolution_s: 1, samples: sceneSamples, surface: {} },
      video: {
        status: "available",
        resolution_s: 1,
        media_uri: "clip.mp4",
        samples: [
          { time_s: 0, value: { pts_s: 0 } },
          { time_s: 2, value: { pts_s: 1 } },
          { time_s: 4, value: { pts_s: 2 } },
        ],
      },
      "metric:clearance": { status: "available", resolution_s: 1, samples: metricSamples },
    },
    metrics: {
      clearance: {
        metric_id: "clearance",
        label: "Clearance",
        unit: "m",
        visible: true,
        stream: { samples: metricSamples },
        current: { sample_time_s: 0 },
      },
    },
    events: [],
    intervals: [],
    controls: { default_speed: 1, speeds: [0.5, 1, 2] },
  };
}

let now = 0;
const scheduled = new Map();
const cancelled = new Set();
let nextHandle = 1;
const scheduler = {
  request(callback) {
    const handle = nextHandle++;
    scheduled.set(handle, callback);
    return handle;
  },
  cancel(handle) {
    cancelled.add(handle);
    scheduled.delete(handle);
  },
};

const playback = new ReviewPanelsController(model(), null, {
  now: () => now,
  scheduler,
});
assert.equal(nearestSample(playback.model.streams.scene, 0.5).sample_time_s, 0);
playback.dispatch({ type: "toggle-play" });
assert.equal(playback.state.playing, true);
now = 1000;
playback.tick(now);
assert.equal(playback.state.cursorTimeS, 1);
playback.dispatch({ type: "set-speed", speed: 2 });
now = 1500;
playback.tick(now);
assert.equal(playback.state.cursorTimeS, 2);

now = 0;
const controlClock = new ReviewPanelsController(model(), null, {
  now: () => now,
  scheduler,
});
controlClock.dispatch({ type: "toggle-play" });
now = 1000;
controlClock.tick(now);
assert.equal(controlClock.state.cursorTimeS, 1);
now = 1500;
controlClock.dispatch({ type: "set-speed", speed: 2 });
now = 2500;
controlClock.tick(now);
assert.equal(controlClock.state.cursorTimeS, 3);
now = 3000;
controlClock.dispatch({ type: "metric-seek", metric_id: "clearance", time_s: 1 });
now = 3500;
controlClock.tick(now);
assert.equal(controlClock.state.cursorTimeS, 2);
controlClock.dispatch({ type: "step", delta: -1 });
now = 4000;
controlClock.tick(now);
assert.equal(controlClock.state.cursorTimeS, 2);
controlClock.unmount();

const eventClockModel = model();
eventClockModel.events = [{ event_id: "near-miss", start_s: 2, end_s: 2.5, seek_time_s: 2 }];
now = 4000;
const eventClock = new ReviewPanelsController(eventClockModel, null, {
  now: () => now,
  scheduler,
});
eventClock.dispatch({ type: "toggle-play" });
now = 5000;
eventClock.tick(now);
assert.equal(eventClock.state.cursorTimeS, 1);
now = 5500;
eventClock.dispatch({ type: "event-seek", event_id: "near-miss" });
now = 6000;
eventClock.tick(now);
assert.equal(eventClock.state.cursorTimeS, 2.5);
eventClock.unmount();

playback.dispatch({ type: "toggle-play" });
assert.equal(playback.state.playing, false);
assert.ok(cancelled.size >= 1);
playback.dispatch({ type: "toggle-play" });
now = 2500;
playback.tick(now);
now = 3500;
playback.tick(now);
assert.equal(playback.state.cursorTimeS, 4);
assert.equal(playback.state.playing, false);

const intervalPlayback = new ReviewPanelsController(
  { ...model(), intervals: [{ interval_id: "short", start_s: 0, end_s: 1 }] },
  null,
  { now: () => now, scheduler },
);
intervalPlayback.dispatch({ type: "seek", time_s: 3 });
intervalPlayback.dispatch({ type: "toggle-play" });
intervalPlayback.dispatch({ type: "set-interval", interval_id: "short" });
assert.equal(intervalPlayback.state.cursorTimeS, 1);
assert.equal(intervalPlayback.state.playing, false);

const documentRef = new FakeDocument();
const rootA = new FakeElement(documentRef, "main");
const rootB = new FakeElement(documentRef, "main");
const controllerA = new ReviewPanelsController(model(), rootA, {
  now: () => now,
  scheduler,
});
const controllerB = new ReviewPanelsController(model(), rootB, {
  now: () => now,
  scheduler,
});
assert.equal(documentRef.listeners.get("keydown").length, 2);
assert.equal(findAll(rootA, (element) => element.tagName === "CANVAS").length, 1);
assert.equal(findAll(rootA, (element) => element.tagName === "VIDEO").length, 1);
const videoElement = findAll(rootA, (element) => element.tagName === "VIDEO")[0];
controllerA.dispatch({ type: "seek", time_s: 2, source: "probe" });
assert.equal(findAll(rootA, (element) => element.tagName === "VIDEO")[0], videoElement);
assert.equal(videoElement.currentTime, 1);
controllerA.dispatch({ type: "seek", time_s: 3, source: "probe" });
const metricRow = findAll(rootA, (element) => element.className === "metric-row")[0];
metricRow.emit("click", { target: metricRow });
assert.equal(controllerA.state.cursorTimeS, 2);
const sampleButtons = findAll(rootA, (element) => element.className === "metric-sample");
assert.equal(sampleButtons.length, 2);
sampleButtons[1].emit("click", { stopPropagation() {} });
assert.equal(controllerA.state.cursorTimeS, 2);
assert.equal(findAll(rootA, (element) => element.textContent.includes("[object Object]")).length, 0);
controllerA.dispatch({ type: "toggle-metric", metric_id: "clearance" });
assert.equal(controllerA.snapshot().metrics.clearance.visible, false);
assert.equal(findAll(rootA, (element) => element.className === "metric-sample").length, 0);
controllerA.unmount();
assert.equal(documentRef.listeners.get("keydown").length, 1);
assert.equal(rootA.children.length, 0);
controllerA.mount(rootA);
assert.equal(documentRef.listeners.get("keydown").length, 2);
assert.equal(findAll(rootA, (element) => element.tagName === "CANVAS").length, 1);
assert.equal(findAll(rootA, (element) => element.tagName === "VIDEO").length, 1);
const staleControl = findAll(rootA, (element) => element.tagName === "BUTTON")[0];
controllerA.unmount();
assert.equal(rootA.children.length, 0);
assert.equal(documentRef.listeners.get("keydown").length, 1);
staleControl.emit("click");
assert.equal(controllerA.state.playing, false);

const intervalModel = model();
intervalModel.context = {
  ...intervalModel.context,
  interval_id: "short",
  cursor: { time_s: 3, context_revision: 4 },
};
intervalModel.time.interval = { interval_id: "short", start_s: 1, end_s: 2 };
intervalModel.intervals = [{ interval_id: "short", start_s: 1, end_s: 2 }];
const intervalController = new ReviewPanelsController(intervalModel, null, {
  now: () => now,
  scheduler,
});
assert.equal(intervalController.state.cursorTimeS, 2);
assert.equal(intervalController.state.start, 1);
assert.equal(intervalController.state.end, 2);
const restartRevision = intervalController.state.contextRevision;
intervalController.dispatch({ type: "toggle-play" });
assert.equal(intervalController.state.cursorTimeS, 1);
assert.equal(intervalController.state.contextRevision, restartRevision + 1);
intervalController.dispatch({ type: "toggle-play" });

const endStep = new ReviewPanelsController(model(), null, { now: () => now, scheduler });
endStep.dispatch({ type: "toggle-play" });
endStep.dispatch({ type: "step", delta: 99 });
assert.equal(endStep.state.cursorTimeS, 4);
assert.equal(endStep.state.playing, false);
const endMetric = new ReviewPanelsController(model(), null, { now: () => now, scheduler });
endMetric.dispatch({ type: "toggle-play" });
endMetric.dispatch({ type: "metric-seek", metric_id: "clearance", time_s: 99 });
assert.equal(endMetric.state.cursorTimeS, 4);
assert.equal(endMetric.state.playing, false);
const endEventModel = model();
endEventModel.events = [{ event_id: "terminal", start_s: 4, end_s: 4 }];
const endEvent = new ReviewPanelsController(endEventModel, null, { now: () => now, scheduler });
endEvent.dispatch({ type: "toggle-play" });
endEvent.dispatch({ type: "event-seek", event_id: "terminal" });
assert.equal(endEvent.state.cursorTimeS, 4);
assert.equal(endEvent.state.playing, false);
endStep.unmount();
endMetric.unmount();
endEvent.unmount();

for (const uri of [
  "file:///tmp/clip.mp4",
  "//host/clip.mp4",
  "ftp://host/clip.mp4",
  "javascript:alert(1)",
  "https://host/clip.mp4",
  "/tmp/clip.mp4",
  "../clip.mp4",
  "%2e%2e/clip.mp4",
  "clip.mp4?download=1",
  "clip.mp4#fragment",
  "folder\\clip.mp4",
]) {
  const unsafeRoot = new FakeElement(documentRef, "main");
  const unsafeModel = model();
  unsafeModel.streams.video.media_uri = uri;
  const unsafeController = new ReviewPanelsController(unsafeModel, unsafeRoot, {
    now: () => now,
    scheduler,
  });
  assert.equal(findAll(unsafeRoot, (element) => element.tagName === "VIDEO").length, 0);
  unsafeController.unmount();
}
const unavailableMappingRoot = new FakeElement(documentRef, "main");
const unavailableMappingModel = model();
unavailableMappingModel.streams.video.status = "unavailable";
const unavailableMappingController = new ReviewPanelsController(
  unavailableMappingModel,
  unavailableMappingRoot,
  { now: () => now, scheduler },
);
assert.equal(findAll(unavailableMappingRoot, (element) => element.tagName === "VIDEO").length, 0);
unavailableMappingController.unmount();

const malformedSceneRoot = new FakeElement(documentRef, "main");
const malformedSceneModel = model();
malformedSceneModel.scene_surface = {
  map: { bounds: { malformed: true }, obstacles: "malformed", robot_goal_zones: {} },
};
const malformedSceneController = new ReviewPanelsController(
  malformedSceneModel,
  malformedSceneRoot,
  { now: () => now, scheduler },
);
assert.equal(findAll(malformedSceneRoot, (element) => element.tagName === "CANVAS").length, 1);
malformedSceneController.unmount();

const unmountProbe = new ReviewPanelsController(model(), rootA, {
  now: () => now,
  scheduler,
});
unmountProbe.dispatch({ type: "toggle-play" });
const unmountHandle = unmountProbe._frameHandle;
assert.notEqual(unmountHandle, null);
unmountProbe.unmount();
assert.ok(cancelled.has(unmountHandle));
assert.equal(documentRef.listeners.get("keydown").length, 1);
controllerB.unmount();
assert.equal(documentRef.listeners.get("keydown").length, 0);

console.log("review_panels_runtime: ok");
