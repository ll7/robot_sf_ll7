import * as THREE from "https://unpkg.com/three@0.164.1/build/three.module.js";

const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");
const timeline = document.querySelector("#timeline");
const markers = document.querySelector("#markers");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111827);

const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
camera.up.set(0, 0, -1);
camera.position.set(0, 55, 0);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
root.appendChild(renderer.domElement);

const world = new THREE.Group();
scene.add(world);
scene.add(new THREE.AmbientLight(0xffffff, 1));

const robot = makeAgent(0x22c55e, 0.42);
const egoPedestrian = makeAgent(0xf59e0b, 0.34);
const pedestrianGroup = new THREE.Group();
const trajectory = new THREE.Group();
const rays = new THREE.Group();
world.add(robot, egoPedestrian, pedestrianGroup, trajectory, rays);

let payload = null;
let currentFrame = 0;
let playing = true;
let lastFrameTime = 0;

fetch("./scene.json")
  .then((response) => response.json())
  .then((scenePayload) => {
    payload = scenePayload;
    buildStaticMap(scenePayload.map);
    timeline.max = Math.max(scenePayload.frames.length - 1, 0);
    fitCamera(scenePayload.map);
    buildTimelineMarkers(scenePayload);
    drawFrame(0);
    animate();
  })
  .catch((error) => {
    hud.textContent = `Failed to load scene.json: ${error}`;
  });

timeline.addEventListener("input", () => {
  playing = false;
  drawFrame(Number(timeline.value));
});

window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    playing = !playing;
  }
});

window.addEventListener("resize", resize);
resize();

function buildStaticMap(map) {
  const origin = map.origin || [0, 0];
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(map.width, map.height),
    new THREE.MeshBasicMaterial({ color: 0x243041 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(origin[0] + map.width / 2, -0.02, origin[1] + map.height / 2);
  world.add(floor);

  map.bounds.forEach((line) => world.add(makeLine(line, 0xe5e7eb)));
  map.obstacles.forEach((obstacle) => {
    obstacle.lines.forEach((line) => world.add(makeLine(line, 0xef4444)));
  });
  addZones(map.robot_spawn_zones, 0x60a5fa);
  addZones(map.robot_goal_zones, 0x22c55e);
  addZones(map.ped_spawn_zones, 0xa78bfa);
  addZones(map.ped_goal_zones, 0xfbbf24);
}

function addZones(zones, color) {
  zones.forEach((zone) => {
    const shape = new THREE.Shape(zone.map(([x, y]) => new THREE.Vector2(x, y)));
    const mesh = new THREE.Mesh(
      new THREE.ShapeGeometry(shape),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.28, side: THREE.DoubleSide })
    );
    mesh.rotation.x = -Math.PI / 2;
    world.add(mesh);
  });
}

function buildTimelineMarkers(scenePayload) {
  const numFrames = scenePayload.frames.length;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(numFrames, 2);
  canvas.height = 40;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let hasMarkers = false;

  for (let i = 0; i < numFrames; i++) {
    const frame = scenePayload.frames[i];
    if (frame.event && i > 0 && frame.event !== scenePayload.frames[i - 1].event) {
      ctx.fillStyle = "#fbbf24";
      ctx.fillRect(i, 0, 1, 40);
      hasMarkers = true;
    }
    if (frame.event_id) {
      ctx.fillStyle = "#60a5fa";
      ctx.fillRect(i, 30, 1, 4);
      hasMarkers = true;
    }
    if (frame.planner_action) {
      ctx.fillStyle = "#a78bfa";
      ctx.fillRect(i, 34, 1, 4);
      hasMarkers = true;
    }
  }

  const annotations = scenePayload.annotations || [];
  const annotationColors = ["#f97316", "#ec4899", "#14b8a6", "#eab308", "#8b5cf6"];
  annotations.forEach((ann, idx) => {
    const [start, end] = annotationFrameRange(scenePayload, ann);
    if (start === null || end === null) return;
    const color = annotationColors[idx % annotationColors.length];
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.fillRect(start, 0, end - start + 1, 40);
    ctx.globalAlpha = 1.0;
    ctx.fillRect(start, 24, end - start + 1, 2);
    hasMarkers = true;
  });

  if (!hasMarkers) return;

  markers.style.backgroundImage = `url(${canvas.toDataURL()})`;
  markers.style.backgroundRepeat = "no-repeat";
  markers.style.backgroundSize = "100% 40px";

  const slider = document.querySelector("#timeline");
  slider.style.marginTop = "0";
  markers.style.display = "block";
}

function drawFrame(index) {
  if (!payload) return;
  currentFrame = Math.max(0, Math.min(index, payload.frames.length - 1));
  const frame = payload.frames[currentFrame];
  timeline.value = currentFrame;

  setAgent(robot, frame.robot);
  setAgent(egoPedestrian, frame.ego_pedestrian);
  replaceChildren(pedestrianGroup, frame.pedestrians.map((ped) => {
    const marker = makeAgent(0x38bdf8, 0.24);
    setAgent(marker, { position: ped.position, heading: 0 });
    return marker;
  }));
  const trajectoryPoints = (payload.trajectory || []).slice(0, currentFrame + 1);
  replaceChildren(trajectory, [makePolyline(trajectoryPoints, 0x86efac)]);
  replaceChildren(rays, (frame.rays || []).map((ray) => makeLine([ray[0][0], ray[1][0], ray[0][1], ray[1][1]], 0xf8fafc)));

  const parts = [];
  if (payload.source) parts.push(`src=${payload.source}`);
  const traceId = payload.trace_id || payload.episode_id || "";
  if (traceId) parts.push(`Trace ${traceId}`);
  if (payload.metadata?.source?.scenario_id) {
    parts.push(`scenario=${payload.metadata.source.scenario_id}`);
  }
  if (payload.metadata?.source?.planner_id) {
    parts.push(`planner=${payload.metadata.source.planner_id}`);
  }
  if (payload.metadata?.source?.seed !== undefined) {
    parts.push(`seed=${payload.metadata.source.seed}`);
  }
  if (payload.metadata?.diagnostic_only) {
    parts.push("[diagnostic-only]");
  }
  parts.push(`frame ${currentFrame + 1}/${payload.frames.length}`);
  parts.push(`step=${frame.timestep != null ? frame.timestep : currentFrame}`);
  if (frame.time_s != null) parts.push(`t=${frame.time_s.toFixed(2)}s`);

  if (frame.event) parts.push(`event=${frame.event}`);
  if (frame.event_id) parts.push(`id=${frame.event_id}`);

  if (frame.planner_action) {
    const a = frame.planner_action;
    parts.push(`v=${a.linear_velocity.toFixed(3)} w=${a.angular_velocity.toFixed(3)}`);
  }

  const annotations = payload.annotations || [];
  const activeAnnotations = annotations.filter((ann) => {
    const step = frame.timestep != null ? frame.timestep : currentFrame;
    return step >= ann.anchor.frame_start && step <= ann.anchor.frame_end;
  });
  if (activeAnnotations.length > 0) {
    parts.push("");
    activeAnnotations.forEach((ann) => {
      parts.push(`[${ann.category}] ${ann.summary}`);
    });
  }

  hud.textContent = parts.join(" | ");
}

function annotationFrameRange(scenePayload, annotation) {
  const startStep = annotation.anchor.frame_start;
  const endStep = annotation.anchor.frame_end;
  const frames = scenePayload.frames;
  if (!frames.length) return [null, null];
  const firstStep = frames[0].timestep ?? 0;
  const lastStep = frames[frames.length - 1].timestep ?? frames.length - 1;
  if (endStep < firstStep || startStep > lastStep) return [null, null];
  let start = frames.findIndex((frame) => frame.timestep === startStep);
  let end = frames.findIndex((frame) => frame.timestep === endStep);
  if (start < 0) {
    start = frames.findIndex((frame) => (frame.timestep ?? 0) >= startStep);
  }
  if (end < 0) {
    end = frames.findIndex((frame) => (frame.timestep ?? 0) > endStep) - 1;
  }
  if (end < 0) end = frames.length - 1;
  if (start < 0 || end < start) return [null, null];
  return [start, end];
}

function animate() {
  requestAnimationFrame(animate);
  if (playing && payload?.frames.length > 1 && performance.now() - lastFrameTime >= 100) {
    lastFrameTime = performance.now();
    drawFrame((currentFrame + 1) % payload.frames.length);
  }
  renderer.render(scene, camera);
}

function makeAgent(color, radius) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 0.28, 24),
    new THREE.MeshBasicMaterial({ color })
  );
  const heading = new THREE.Mesh(
    new THREE.BoxGeometry(radius * 1.8, 0.08, 0.08),
    new THREE.MeshBasicMaterial({ color: 0xf8fafc })
  );
  heading.position.x = radius;
  group.add(body, heading);
  return group;
}

function setAgent(agent, pose) {
  agent.visible = Boolean(pose);
  if (!pose) return;
  agent.position.set(pose.position[0], 0.18, pose.position[1]);
  agent.rotation.y = -pose.heading;
}

function makeLine(line, color) {
  return makePolyline([[line[0], line[2]], [line[1], line[3]]], color);
}

function makePolyline(points, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, y]) => new THREE.Vector3(x, 0.05, y))
  );
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
}

function replaceChildren(group, children) {
  group.traverse((object) => {
    if (object === group) return;
    disposeObject(object);
  });
  group.clear();
  children.filter(Boolean).forEach((child) => group.add(child));
}

function disposeObject(object) {
  if (object.geometry) object.geometry.dispose();
  const materials = Array.isArray(object.material) ? object.material : [object.material];
  materials.filter(Boolean).forEach((material) => {
    if (material.map) material.map.dispose();
    material.dispose();
  });
}

function fitCamera(map) {
  const origin = map.origin || [0, 0];
  const width = root.clientWidth || window.innerWidth;
  const height = root.clientHeight || window.innerHeight;
  const aspect = width / Math.max(height, 1);
  const centerX = origin[0] + map.width / 2;
  const centerY = origin[1] + map.height / 2;
  const margin = 1.16;
  let halfWidth = (map.width * margin) / 2;
  let halfHeight = (map.height * margin) / 2;

  if (halfWidth / halfHeight > aspect) {
    halfHeight = halfWidth / aspect;
  } else {
    halfWidth = halfHeight * aspect;
  }

  camera.left = centerX - halfWidth;
  camera.right = centerX + halfWidth;
  camera.top = halfHeight;
  camera.bottom = -halfHeight;
  camera.position.set(centerX, Math.max(map.width, map.height) * 2.1, centerY);
  camera.lookAt(centerX, 0, centerY);
  camera.updateProjectionMatrix();
}

function resize() {
  const width = root.clientWidth || window.innerWidth;
  const height = root.clientHeight || window.innerHeight;
  renderer.setSize(width, height);
  if (payload) fitCamera(payload.map);
}
