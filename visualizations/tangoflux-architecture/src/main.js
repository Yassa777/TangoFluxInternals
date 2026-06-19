import "./styles.css";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { architecture, modes, siteNotes, workflowNodes } from "./data.js";

const canvas = document.querySelector("#scene");
const labelRoot = document.querySelector("#labels");
const panelEyebrow = document.querySelector("#panel-eyebrow");
const panelTitle = document.querySelector("#panel-title");
const panelCopy = document.querySelector("#panel-copy");
const siteCard = document.querySelector("#site-card");
const evidenceTable = document.querySelector("#evidence-table");
const infoPanel = document.querySelector(".info-panel");
const labelToggle = document.querySelector("#label-toggle");
const panelToggle = document.querySelector("#panel-toggle");
const modeButtons = [...document.querySelectorAll(".mode-button")];

const palette = {
  prompt: new THREE.Color("#69d2e7"),
  audio: new THREE.Color("#f2b84b"),
  single: new THREE.Color("#7ce38b"),
  hot: new THREE.Color("#ff6b5f"),
  dim: new THREE.Color("#45515c"),
  rail: new THREE.Color("#9aa7b2"),
  background: new THREE.Color("#080a0d")
};

let activeMode = "model";
let selectedObject = null;
let hoveredObject = null;
let labelsVisible = true;
let panelVisible = true;

const scene = new THREE.Scene();
scene.background = palette.background;
scene.fog = new THREE.Fog("#080a0d", 10, 30);

const camera = new THREE.PerspectiveCamera(43, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0.4, 5.4, 15.6);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: "high-performance"
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 7;
controls.maxDistance = 20;
controls.maxPolarAngle = Math.PI * 0.68;
controls.target.set(0.15, -0.15, 0);

scene.add(new THREE.HemisphereLight("#bfd8ff", "#211a14", 1.8));
const key = new THREE.DirectionalLight("#ffffff", 2.4);
key.position.set(-3, 7, 7);
key.castShadow = true;
scene.add(key);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(22, 11),
  new THREE.MeshStandardMaterial({
    color: "#15191d",
    roughness: 0.8,
    metalness: 0.15
  })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -2.35;
floor.receiveShadow = true;
scene.add(floor);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const interactives = [];
const siteMeshes = new Map();
const animatedParticles = [];
const sceneLabels = [];

function makeMaterial(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color.clone().multiplyScalar(0.22),
    roughness: 0.42,
    metalness: 0.2,
    transparent: opacity < 1,
    opacity
  });
}

function makeAccentMaterial(color, opacity = 0.9) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity
  });
}

function label(text, position, className = "") {
  const el = document.createElement("div");
  el.className = `scene-label ${className}`;
  el.textContent = text;
  labelRoot.appendChild(el);
  const item = { element: el, position: position.clone() };
  sceneLabels.push(item);
  return item;
}

function addBlockDetail(mesh, { stack, color, size }) {
  const [sx, sy, sz] = size;
  const lift = color.clone().lerp(new THREE.Color("#ffffff"), 0.28);
  const shadow = color.clone().lerp(new THREE.Color("#050607"), 0.42);
  const frontZ = sz * 0.535;

  const capTop = new THREE.Mesh(
    new THREE.BoxGeometry(sx * 0.95, sy * 0.08, sz * 0.92),
    makeMaterial(lift, 0.92)
  );
  capTop.position.set(0, sy * 0.52, 0);
  mesh.add(capTop);

  const capBottom = new THREE.Mesh(
    new THREE.BoxGeometry(sx * 0.95, sy * 0.05, sz * 0.92),
    makeMaterial(shadow, 0.78)
  );
  capBottom.position.set(0, -sy * 0.52, 0);
  mesh.add(capBottom);

  const sublayers = stack === "dual"
    ? [
        { y: sy * 0.22, w: sx * 0.72, c: "#d9eef5" },
        { y: sy * 0.08, w: sx * 0.58, c: "#69d2e7" },
        { y: -sy * 0.07, w: sx * 0.5, c: "#f2b84b" },
        { y: -sy * 0.22, w: sx * 0.68, c: "#e7f5ee" }
      ]
    : [
        { y: sy * 0.24, w: sx * 0.76, c: "#d9eef5" },
        { y: sy * 0.08, w: sx * 0.64, c: "#7ce38b" },
        { y: -sy * 0.08, w: sx * 0.48, c: "#f2b84b" },
        { y: -sy * 0.24, w: sx * 0.7, c: "#e7f5ee" }
      ];
  for (const [index, layer] of sublayers.entries()) {
    const rail = new THREE.Mesh(
      new THREE.BoxGeometry(sx * 0.82, sy * 0.07, 0.012),
      makeAccentMaterial("#11181d", 0.72)
    );
    rail.position.set(0, layer.y, frontZ + index * 0.002);
    mesh.add(rail);

    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(layer.w, sy * 0.047, 0.018),
      makeAccentMaterial(layer.c, 0.76)
    );
    plate.position.set(-(sx * 0.82 - layer.w) / 2, layer.y, frontZ + 0.012 + index * 0.002);
    mesh.add(plate);
  }

  const leftPins = stack === "dual" ? ["#69d2e7", "#f2b84b"] : ["#69d2e7", "#f2b84b", "#7ce38b"];
  leftPins.forEach((pinColor, index) => {
    const pin = new THREE.Mesh(
      new THREE.CylinderGeometry(sx * 0.025, sx * 0.025, sz * 0.76, 8),
      makeAccentMaterial(pinColor, 0.72)
    );
    pin.rotation.x = Math.PI / 2;
    pin.position.set(-sx * 0.57, sy * (0.2 - index * 0.2), 0);
    mesh.add(pin);
  });

  const channelRibs = 5;
  for (let index = 0; index < channelRibs; index += 1) {
    const rib = new THREE.Mesh(
      new THREE.BoxGeometry(0.012, sy * 0.08, sz * 0.72),
      makeAccentMaterial("#d7e4ec", 0.2)
    );
    rib.position.set(-sx * 0.32 + index * sx * 0.16, sy * 0.55, 0);
    mesh.add(rib);
  }

  const suffixStrip = new THREE.Mesh(
    new THREE.BoxGeometry(sx * 0.1, sy * 0.74, 0.018),
    makeAccentMaterial(stack === "single" ? "#f2b84b" : "#69d2e7", stack === "single" ? 0.86 : 0.52)
  );
  suffixStrip.position.set(sx * 0.49, 0, -sz * 0.53);
  mesh.add(suffixStrip);

  const hookPlate = new THREE.Mesh(
    new THREE.BoxGeometry(sx * 0.18, sy * 0.16, 0.018),
    makeAccentMaterial("#ff6b5f", 0.82)
  );
  hookPlate.position.set(0, -sy * 0.55, frontZ);
  mesh.add(hookPlate);

  const latentCells = stack === "dual"
    ? ["#69d2e7", "#69d2e7", "#f2b84b", "#f2b84b"]
    : ["#69d2e7", "#69d2e7", "#7ce38b", "#f2b84b", "#f2b84b"];
  latentCells.forEach((cellColor, index) => {
    const cell = new THREE.Mesh(
      new THREE.BoxGeometry(sx * 0.09, sy * 0.028, sz * 0.1),
      makeAccentMaterial(cellColor, 0.86)
    );
    cell.position.set(-sx * 0.33 + index * sx * 0.16, sy * 0.72, -sz * 0.2);
    mesh.add(cell);
  });

  const componentTags = [
    { x: -sx * 0.28, c: "#d9eef5" },
    { x: -sx * 0.09, c: stack === "dual" ? "#69d2e7" : "#7ce38b" },
    { x: sx * 0.1, c: "#f2b84b" },
    { x: sx * 0.29, c: "#ff6b5f" }
  ];
  for (const tag of componentTags) {
    const chip = new THREE.Mesh(
      new THREE.BoxGeometry(sx * 0.11, sy * 0.08, 0.02),
      makeAccentMaterial(tag.c, 0.72)
    );
    chip.position.set(tag.x, sy * 0.48, frontZ + 0.02);
    mesh.add(chip);
  }

  const dotColors =
    stack === "single"
      ? ["#69d2e7", "#69d2e7", "#7ce38b", "#f2b84b", "#f2b84b"]
      : [color.getStyle(), color.getStyle(), color.getStyle(), color.getStyle()];
  dotColors.forEach((dotColor, index) => {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(0.018, sx * 0.045), 10, 10),
      makeAccentMaterial(dotColor, 0.95)
    );
    const x = -sx * 0.34 + index * sx * 0.17;
    dot.position.set(x, sy * 0.67, sz * 0.14);
    mesh.add(dot);
  });

  const port = new THREE.Mesh(
    new THREE.TorusGeometry(Math.max(0.05, sx * 0.13), Math.max(0.006, sx * 0.016), 8, 18),
    makeAccentMaterial("#f7f2d2", 0.82)
  );
  port.position.set(sx * 0.5, 0, sz * 0.55);
  port.rotation.y = Math.PI / 2;
  mesh.add(port);

  const outputMarker = new THREE.Mesh(
    new THREE.BoxGeometry(sx * 0.12, sy * 0.7, 0.014),
    makeAccentMaterial(stack === "dual" ? "#f2b84b" : "#7ce38b", 0.76)
  );
  outputMarker.position.set(-sx * 0.52, 0, sz * 0.55);
  mesh.add(outputMarker);

  const signalBars = [];
  for (let index = 0; index < 4; index += 1) {
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(sx * 0.08, sy * 0.22, 0.018),
      makeAccentMaterial("#ff6b5f", 0.0)
    );
    bar.position.set(sx * (0.18 + index * 0.11), -sy * 0.26, sz * 0.57);
    bar.visible = false;
    mesh.add(bar);
    signalBars.push(bar);
  }
  mesh.userData.signalBars = signalBars;
}

function makeBlock({ name, labelText, stack, block, position, color, size = [0.48, 0.36, 0.48] }) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), makeMaterial(color));
  mesh.position.copy(position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = {
    kind: "site",
    site: name,
    stack,
    block,
    baseColor: color.clone(),
    baseScale: new THREE.Vector3(1, 1, 1)
  };
  scene.add(mesh);
  interactives.push(mesh);
  siteMeshes.set(name, [...(siteMeshes.get(name) || []), mesh]);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: "#d7e4ec", transparent: true, opacity: 0.28 })
  );
  mesh.add(edges);

  if (labelText) {
    label(labelText, position.clone().add(new THREE.Vector3(0, 0.38, 0)), stack);
  }
  addBlockDetail(mesh, { stack, color, size });
  return mesh;
}

function makeNode(text, subtext, position, color) {
  const group = new THREE.Group();
  group.position.copy(position);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.16, 28), makeMaterial(color));
  mesh.rotation.x = Math.PI / 2;
  mesh.castShadow = true;
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.39, 0.014, 8, 36),
    makeAccentMaterial(color.clone().lerp(new THREE.Color("#ffffff"), 0.24), 0.86)
  );
  ring.rotation.x = Math.PI / 2;
  group.add(ring);
  const inner = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.16, 0.18, 20),
    makeMaterial(color.clone().lerp(new THREE.Color("#050607"), 0.32), 0.95)
  );
  inner.rotation.x = Math.PI / 2;
  group.add(inner);
  for (let index = 0; index < 3; index += 1) {
    const spoke = new THREE.Mesh(
      new THREE.BoxGeometry(0.32, 0.018, 0.018),
      makeAccentMaterial("#e8f2f4", 0.45)
    );
    spoke.rotation.z = (Math.PI * 2 * index) / 3;
    spoke.position.z = 0.09;
    group.add(spoke);
  }
  mesh.userData = {
    kind: "workflow",
    title: text,
    subtext,
    baseColor: color.clone(),
    baseScale: new THREE.Vector3(1, 1, 1)
  };
  group.add(mesh);
  scene.add(group);
  interactives.push(mesh);
  label(text, position.clone().add(new THREE.Vector3(0, 0.36, 0)), "workflow-label");
  return mesh;
}

function curveTube(points, color, radius = 0.025, opacity = 0.8) {
  const curve = new THREE.CatmullRomCurve3(points);
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 80, radius, 8, false),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
  );
  scene.add(tube);
  return curve;
}

function makeArrow(from, to, color) {
  const direction = to.clone().sub(from).normalize();
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.08, 0.22, 18),
    new THREE.MeshBasicMaterial({ color })
  );
  cone.position.copy(to);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  scene.add(cone);
}

function makeParticle(curve, color, offset, radius = 0.07) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 16),
    new THREE.MeshBasicMaterial({ color })
  );
  scene.add(mesh);
  animatedParticles.push({ mesh, curve, offset });
}

function makeWaveform() {
  const points = [];
  for (let i = 0; i < 120; i += 1) {
    const t = i / 119;
    points.push(
      new THREE.Vector3(
        6.4 + t * 1.55,
        Math.sin(t * Math.PI * 14) * (0.18 + 0.16 * Math.sin(t * Math.PI * 3)),
        -0.05
      )
    );
  }
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: "#f7f2d2", linewidth: 2 })
  );
  line.position.y = 0.05;
  scene.add(line);
}

function buildModel() {
  label("Prompt tokens", new THREE.Vector3(-7.2, 1.58, 0), "prompt");
  label("Audio latents", new THREE.Vector3(-7.2, -0.52, 0), "audio");
  label("[B, T_text, 1024]", new THREE.Vector3(-6.35, 1.62, 0.3), "shape-label prompt-shape");
  label("[B=2, T_audio=645, 1024]", new THREE.Vector3(-6.18, -0.48, 0.3), "shape-label audio-shape");

  const promptInput = makeNode("Text", "prompt conditioning", new THREE.Vector3(-7.25, 1.05, 0), palette.prompt);
  const audioInput = makeNode("Latents", "noisy audio tokens", new THREE.Vector3(-7.25, -1.05, 0), palette.audio);
  promptInput.userData.kind = "input";
  audioInput.userData.kind = "input";

  const textPoints = [new THREE.Vector3(-6.9, 1.05, 0)];
  const audioPoints = [new THREE.Vector3(-6.9, -1.05, 0)];
  const dualStartX = -5.9;
  for (let i = 0; i < architecture.dualBlocks; i += 1) {
    const x = dualStartX + i * 0.75;
    const site = `transformer.transformer_blocks.${i}`;
    makeBlock({
      name: site,
      labelText: i === 0 || i === 3 || i === 5 ? `dual ${i}` : "",
      stack: "dual",
      block: i,
      position: new THREE.Vector3(x, 1.05, 0),
      color: palette.prompt,
      size: [0.42, 0.28, 0.42]
    });
    makeBlock({
      name: site,
      labelText: "",
      stack: "dual",
      block: i,
      position: new THREE.Vector3(x, -1.05, 0),
      color: palette.audio,
      size: [0.42, 0.28, 0.42]
    });
    textPoints.push(new THREE.Vector3(x, 1.05, 0));
    audioPoints.push(new THREE.Vector3(x, -1.05, 0));

    const cross = curveTube(
      [
        new THREE.Vector3(x, 0.8, -0.18),
        new THREE.Vector3(x + 0.05, 0, -0.38),
        new THREE.Vector3(x, -0.8, -0.18)
      ],
      "#607786",
      0.011,
      0.35
    );
    if (i % 2 === 0) makeParticle(cross, "#b7c4cc", i / 7, 0.035);
  }

  const merge = makeNode("Merge", "text + audio stream", new THREE.Vector3(-1.0, 0, 0), palette.single);
  merge.userData.kind = "merge";
  label("concat [text || audio]", new THREE.Vector3(-1.04, 0.66, 0.25), "shape-label merge-shape");
  label("[B=2, T_text + 645, 1024]", new THREE.Vector3(0.55, 0.62, 0.28), "shape-label single-shape");
  textPoints.push(new THREE.Vector3(-1.2, 0.22, 0));
  audioPoints.push(new THREE.Vector3(-1.2, -0.22, 0));

  const textCurve = curveTube(textPoints, palette.prompt, 0.018, 0.66);
  const audioCurve = curveTube(audioPoints, palette.audio, 0.018, 0.7);
  makeArrow(textPoints.at(-2), textPoints.at(-1), palette.prompt);
  makeArrow(audioPoints.at(-2), audioPoints.at(-1), palette.audio);
  makeParticle(textCurve, palette.prompt, 0);
  makeParticle(textCurve, palette.prompt, 0.42);
  makeParticle(audioCurve, palette.audio, 0.2);
  makeParticle(audioCurve, palette.audio, 0.72);

  const singlePoints = [new THREE.Vector3(-0.7, 0, 0)];
  for (let i = 0; i < architecture.singleBlocks; i += 1) {
    const x = -0.35 + i * 0.32;
    const y = Math.sin(i * 0.7) * 0.18;
    const z = i % 2 === 0 ? 0.12 : -0.12;
    const site = `transformer.single_transformer_blocks.${i}`;
    makeBlock({
      name: site,
      labelText: [0, 3, 6, 8, 14, 17].includes(i) ? `single ${i}` : "",
      stack: "single",
      block: i,
      position: new THREE.Vector3(x, y, z),
      color: palette.single,
      size: [0.34, 0.34, 0.34]
    });
    singlePoints.push(new THREE.Vector3(x, y, z));
  }
  singlePoints.push(new THREE.Vector3(6.05, 0, 0));
  const singleCurve = curveTube(singlePoints, palette.single, 0.022, 0.72);
  makeArrow(singlePoints.at(-2), singlePoints.at(-1), palette.single);
  makeParticle(singleCurve, palette.single, 0.08);
  makeParticle(singleCurve, palette.single, 0.38);
  makeParticle(singleCurve, palette.single, 0.72);

  const decoder = makeNode("Decode", "latent audio to waveform", new THREE.Vector3(6.2, 0, 0), new THREE.Color("#e8dc7a"));
  decoder.userData.kind = "decoder";
  makeWaveform();
  label("pool -> [B=2, 1024]", new THREE.Vector3(4.15, -0.58, 0.24), "shape-label probe-shape");
  label("Waveform metrics", new THREE.Vector3(7.18, 0.62, 0), "waveform");
}

function buildWorkflow() {
  const startX = -6.7;
  const gap = 1.85;
  const y = -2.0;
  workflowNodes.forEach(([title, subtext], index) => {
    const x = startX + index * gap;
    const color = index % 3 === 0 ? new THREE.Color("#ff9f70") : index % 3 === 1 ? new THREE.Color("#8ab4ff") : new THREE.Color("#9be285");
    makeNode(title, subtext, new THREE.Vector3(x, y, 0.2), color);
    if (index > 0) {
      curveTube(
        [new THREE.Vector3(x - gap + 0.36, y, 0.2), new THREE.Vector3(x - 0.36, y, 0.2)],
        "#5f6d73",
        0.01,
        0.7
      );
    }
  });
}

function resetSiteAppearance() {
  for (const meshes of siteMeshes.values()) {
    for (const mesh of meshes) {
      const { baseColor } = mesh.userData;
      mesh.material.color.copy(baseColor);
      mesh.material.emissive.copy(baseColor).multiplyScalar(0.18);
      mesh.material.opacity = 1;
      mesh.scale.set(1, 1, 1);
      for (const [index, bar] of (mesh.userData.signalBars || []).entries()) {
        bar.visible = false;
        bar.scale.set(1, 1, 1);
        bar.position.y = -mesh.geometry.parameters.height * 0.26;
        bar.material.opacity = 0;
        bar.material.color.set(index === 0 ? "#ff6b5f" : index === 1 ? "#f2d26b" : "#69d2e7");
      }
    }
  }
}

function applyMode(modeName) {
  activeMode = modeName;
  const mode = modes[modeName];
  modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === modeName));
  panelEyebrow.textContent = mode.eyebrow;
  panelTitle.textContent = mode.title;
  panelCopy.textContent = mode.copy;
  evidenceTable.innerHTML = table(mode.rows);

  resetSiteAppearance();
  const highlights = mode.highlights || {};
  for (const [site, strength] of Object.entries(highlights)) {
    for (const mesh of siteMeshes.get(site) || []) {
      const mix = palette.hot.clone().lerp(new THREE.Color("#f9d36d"), 1 - strength * 0.5);
      mesh.material.color.copy(mix);
      mesh.material.emissive.copy(mix).multiplyScalar(0.65 + strength * 0.45);
      const scale = 1 + strength * 0.35;
      mesh.scale.set(scale, scale, scale);
      for (const [index, bar] of (mesh.userData.signalBars || []).entries()) {
        const h = 0.35 + strength * (0.45 + index * 0.18);
        bar.visible = true;
        bar.scale.y = h;
        bar.position.y = -mesh.geometry.parameters.height * 0.25 + h * 0.035;
        bar.material.opacity = 0.62 + strength * 0.28;
      }
    }
  }
  if (!selectedObject) updateSiteCard(null);
}

function table(rows) {
  return `<table>${rows
    .map(([a, b]) => `<tr><th>${escapeHtml(a)}</th><td>${escapeHtml(b)}</td></tr>`)
    .join("")}</table>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function siteDisplay(site) {
  if (!site) return "None";
  return site.replace("transformer.", "").replace("transformer_blocks", "dual").replace("single_transformer_blocks", "single");
}

function defaultSiteNote(data) {
  const isDual = data.stack === "dual";
  return {
    title: isDual ? `Dual block ${data.block}` : `Single block ${data.block}`,
    stack: data.stack,
    output: isDual ? "output_index=1 audio stream" : "merged stream audio suffix",
    rawShape: isDual ? "[B=2, T_audio=645, D=1024]" : "[B=2, T_text + 645, D=1024]",
    probeShape: isDual ? "[B=2, D=1024] after audio-token pooling" : "[B=2, D=1024] from trailing 645 audio tokens",
    components: isDual
      ? "dual-stream norm, self/cross attention, MLP, residual add, audio-output hook"
      : "merged-stream norm, attention, MLP, residual add, audio-suffix hook",
    note: "Hooked site used by capture, patching, or steering utilities."
  };
}

function updateSiteCard(object) {
  if (!object) {
    const shapeRows = architecture.tensorShapes
      .map(([name, shape]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(shape)}</dd></div>`)
      .join("");
    siteCard.innerHTML = `
      <p class="muted">Select a block, stream node, or workflow node.</p>
      <dl>
        <div><dt>Model</dt><dd>${architecture.modelName}</dd></div>
        <div><dt>Hook sites</dt><dd>${architecture.dualBlocks + architecture.singleBlocks}</dd></div>
        <div><dt>Hidden size</dt><dd>${architecture.hiddenSize}</dd></div>
        ${shapeRows}
      </dl>`;
    return;
  }

  const data = object.userData;
  if (data.kind === "site") {
    const note = { ...defaultSiteNote(data), ...(siteNotes[data.site] || {}) };
    const heat = modes[activeMode].highlights[data.site];
    siteCard.innerHTML = `
      <h4>${escapeHtml(note.title)}</h4>
      <dl>
        <div><dt>Site</dt><dd>${escapeHtml(siteDisplay(data.site))}</dd></div>
        <div><dt>Stack</dt><dd>${escapeHtml(note.stack)}</dd></div>
        <div><dt>Capture</dt><dd>${escapeHtml(note.output)}</dd></div>
        <div><dt>Raw tensor</dt><dd>${escapeHtml(note.rawShape)}</dd></div>
        <div><dt>Probe vector</dt><dd>${escapeHtml(note.probeShape)}</dd></div>
        <div><dt>Block internals</dt><dd>${escapeHtml(note.components)}</dd></div>
        <div><dt>Mode weight</dt><dd>${heat ? heat.toFixed(2) : "not highlighted"}</dd></div>
      </dl>
      <p>${escapeHtml(note.note)}</p>`;
    return;
  }

  siteCard.innerHTML = `
    <h4>${escapeHtml(data.title || data.kind)}</h4>
    <dl>
      <div><dt>Role</dt><dd>${escapeHtml(data.subtext || data.kind)}</dd></div>
      <div><dt>Mode</dt><dd>${escapeHtml(modes[activeMode].title)}</dd></div>
    </dl>`;
}

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const [hit] = raycaster.intersectObjects(interactives, false);
  hoveredObject = hit?.object || null;
  document.body.classList.toggle("is-pointing", Boolean(hoveredObject));
}

function onPointerDown() {
  if (!hoveredObject) return;
  selectedObject = hoveredObject;
  updateSiteCard(selectedObject);
}

function updateHoverPulse(time) {
  for (const mesh of interactives) {
    if (mesh === selectedObject) {
      const pulse = 1.08 + Math.sin(time * 0.006) * 0.03;
      mesh.scale.set(pulse, pulse, pulse);
    } else if (mesh === hoveredObject) {
      mesh.scale.set(1.12, 1.12, 1.12);
    } else if (!Object.prototype.hasOwnProperty.call(modes[activeMode].highlights, mesh.userData.site)) {
      mesh.scale.lerp(new THREE.Vector3(1, 1, 1), 0.12);
    }
  }
}

function animate(time) {
  for (const particle of animatedParticles) {
    const t = (time * 0.00008 + particle.offset) % 1;
    particle.mesh.position.copy(particle.curve.getPointAt(t));
  }
  updateHoverPulse(time);
  controls.update();
  updateLabels();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRoot.style.width = `${window.innerWidth}px`;
  labelRoot.style.height = `${window.innerHeight}px`;
}

function updateLabels() {
  if (!labelsVisible) {
    for (const item of sceneLabels) item.element.style.display = "none";
    return;
  }
  const width = window.innerWidth;
  const height = window.innerHeight;
  for (const item of sceneLabels) {
    const projected = item.position.clone().project(camera);
    const visible = projected.z > -1 && projected.z < 1;
    item.element.style.display = visible ? "block" : "none";
    if (!visible) continue;
    const x = (projected.x * 0.5 + 0.5) * width;
    const y = (-projected.y * 0.5 + 0.5) * height;
    item.element.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
  }
}

function setLabelVisibility(visible) {
  labelsVisible = visible;
  labelRoot.classList.toggle("is-hidden", !visible);
  updateLabels();
}

function setPanelVisibility(visible) {
  panelVisible = visible;
  infoPanel.classList.toggle("is-hidden", !visible);
  infoPanel.setAttribute("aria-hidden", String(!visible));
}

buildModel();
buildWorkflow();
applyMode("model");
updateSiteCard(null);

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedObject = null;
    applyMode(button.dataset.mode);
    updateSiteCard(null);
  });
});

labelToggle.addEventListener("change", () => setLabelVisibility(labelToggle.checked));
panelToggle.addEventListener("change", () => setPanelVisibility(panelToggle.checked));
window.addEventListener("resize", resize);
window.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerdown", onPointerDown);
requestAnimationFrame(animate);
