import * as ort from 'onnxruntime-web/wasm';
import modelUrl from '../public/models/waste_detector.onnx?url';

const IMAGE_SIZE = 416;
const CLASS_NAMES = ['Biodegradable', 'Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico'];
const CLASS_COLORS = ['#94613d', '#2772c7', '#747f89', '#d8a91e', '#2772c7', '#6e7b73'];
const disposal = [
  { label: 'Biodegradable', bin: 'marrón', stream: 'Orgánicos', color: '#94613d', wash: '#f5eee8', edge: '#ead8c8', detail: '#fff8f1' },
  { label: 'Cartón', bin: 'azul', stream: 'Papel y cartón', color: '#2772c7', wash: '#edf4fc', edge: '#d9e6f5', detail: '#fff' },
  { label: 'Vidrio', bin: 'plomo', stream: 'Vidrio', color: '#747f89', wash: '#f0f2f4', edge: '#dfe3e6', detail: '#fff' },
  { label: 'Metal', bin: 'amarillo', stream: 'Metales', color: '#e2b52e', wash: '#fbf6e4', edge: '#eee3b7', detail: '#fffdf2' },
  { label: 'Papel', bin: 'azul', stream: 'Papel y cartón', color: '#2772c7', wash: '#edf4fc', edge: '#d9e6f5', detail: '#fff' },
  { label: 'Plástico', bin: 'blanco', stream: 'Plástico', color: '#f8faf8', outline: '#aab7ae', wash: '#f3f6f3', edge: '#dfe6e1', detail: '#aab7ae' },
  { label: 'Peligrosos', bin: 'rojo', stream: 'Peligrosos', color: '#c93e3e', wash: '#fff0ef', edge: '#f1d5d3', detail: '#fff' },
];

ort.env.wasm.numThreads = 1;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayContext = overlay.getContext('2d');
const stage = document.getElementById('stage');
const idle = document.getElementById('idle');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const confidence = document.getElementById('confidence');
const confidenceValue = document.getElementById('confidenceValue');
const detectionsList = document.getElementById('detections');
const emptyState = document.getElementById('emptyState');
const count = document.getElementById('count');
const latency = document.getElementById('latency');
const notice = document.getElementById('notice');
const modelPill = document.getElementById('serverPill');
const modelText = document.getElementById('serverText');
const cameraPill = document.getElementById('cameraPill');
const cameraText = document.getElementById('cameraText');
const frameCanvas = document.createElement('canvas');
const frameContext = frameCanvas.getContext('2d', { willReadFrequently: true });
const inputCanvas = document.createElement('canvas');
inputCanvas.width = IMAGE_SIZE;
inputCanvas.height = IMAGE_SIZE;
const inputContext = inputCanvas.getContext('2d', { willReadFrequently: true });

let cameraStream = null;
let running = false;
let sessionPromise = null;

function setNotice(message, isError = false) {
  notice.textContent = message;
  notice.classList.toggle('error', isError);
}

function makeBinIcon(classId) {
  const bin = disposal[classId] || disposal[0];
  const wrapper = document.createElement('span');
  wrapper.className = 'bin-icon-wrap';
  wrapper.style.setProperty('--bin-color', bin.color);
  wrapper.style.setProperty('--bin-wash', bin.wash);
  wrapper.style.setProperty('--bin-edge', bin.edge);
  wrapper.style.setProperty('--bin-outline', bin.outline || bin.color);
  wrapper.style.setProperty('--bin-detail', bin.detail);
  wrapper.setAttribute('role', 'img');
  wrapper.setAttribute('aria-label', `Tacho ${bin.bin}`);
  wrapper.innerHTML = '<svg class="bin-icon" viewBox="0 0 40 48" aria-hidden="true"><path class="bin-body" d="M8 13h24l-1.8 27a3 3 0 0 1-3 2.8H12.8a3 3 0 0 1-3-2.8L8 13Z"/><path class="bin-lid" d="M5.5 9h29v4h-29z"/><path class="bin-handle" d="M15 6h10"/><path class="bin-detail" d="M15 19v18M20 19v18M25 19v18"/></svg>';
  return wrapper;
}

function renderBinGuide() {
  const guide = document.getElementById('binGuide');
  for (const [classId, bin] of disposal.entries()) {
    const item = document.createElement('div');
    item.className = 'guide-item';
    item.append(makeBinIcon(classId));
    const name = document.createElement('span');
    name.className = 'guide-name';
    const classLabel = document.createElement('strong');
    classLabel.textContent = bin.label;
    const destination = document.createElement('span');
    destination.textContent = `Tacho ${bin.bin}`;
    name.append(classLabel, destination);
    item.append(name);
    guide.append(item);
  }
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }
  return sessionPromise;
}

async function checkModel() {
  modelText.textContent = 'Cargando modelo…';
  setNotice('Descargando el modelo para ejecutarlo en este dispositivo…');
  try {
    await getSession();
    modelPill.classList.add('ready');
    modelPill.classList.remove('error');
    modelText.textContent = 'Modelo listo en el dispositivo';
    setNotice('Modelo listo. La cámara y sus imágenes se procesan en este dispositivo.');
  } catch (error) {
    modelPill.classList.add('error');
    modelText.textContent = 'No se pudo cargar el modelo';
    setNotice(`Error al cargar el modelo ONNX: ${error.message || error}`, true);
  }
}

function makeInput(source) {
  const width = source.width;
  const height = source.height;
  const scale = Math.min(IMAGE_SIZE / width, IMAGE_SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const horizontalPadding = IMAGE_SIZE - resizedWidth;
  const verticalPadding = IMAGE_SIZE - resizedHeight;
  const padLeft = Math.round(horizontalPadding / 2 - 0.1);
  const padTop = Math.round(verticalPadding / 2 - 0.1);

  inputContext.fillStyle = 'rgb(114, 114, 114)';
  inputContext.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  inputContext.drawImage(source, 0, 0, width, height, padLeft, padTop, resizedWidth, resizedHeight);

  const pixels = inputContext.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const planeSize = IMAGE_SIZE * IMAGE_SIZE;
  const tensorData = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const offset = pixel * 4;
    tensorData[pixel] = pixels[offset] / 255;
    tensorData[planeSize + pixel] = pixels[offset + 1] / 255;
    tensorData[2 * planeSize + pixel] = pixels[offset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor('float32', tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]),
    scale,
    padLeft,
    padTop,
    width,
    height,
  };
}

function boxIoU(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function decodeOutput(output, prepared, threshold) {
  const dims = output.dims;
  let featureCount = dims[dims.length - 2];
  let candidateCount = dims[dims.length - 1];
  let transposed = false;
  if (featureCount < candidateCount) {
    transposed = true;
  } else {
    [featureCount, candidateCount] = [candidateCount, featureCount];
  }

  if (featureCount !== 4 + CLASS_NAMES.length) {
    throw new Error(`La salida del modelo tiene ${featureCount} valores por candidato; se esperaban ${4 + CLASS_NAMES.length}.`);
  }

  const values = output.data;
  const at = (feature, candidate) => transposed
    ? values[feature * candidateCount + candidate]
    : values[candidate * featureCount + feature];
  const candidatesByClass = Array.from({ length: CLASS_NAMES.length }, () => []);

  for (let candidate = 0; candidate < candidateCount; candidate += 1) {
    let classId = 0;
    let score = -Infinity;
    for (let classIndex = 0; classIndex < CLASS_NAMES.length; classIndex += 1) {
      const classScore = at(4 + classIndex, candidate);
      if (classScore > score) {
        score = classScore;
        classId = classIndex;
      }
    }
    if (score < threshold) continue;

    const centerX = at(0, candidate);
    const centerY = at(1, candidate);
    const width = at(2, candidate);
    const height = at(3, candidate);
    const x1 = Math.round((centerX - width / 2 - prepared.padLeft) / prepared.scale);
    const y1 = Math.round((centerY - height / 2 - prepared.padTop) / prepared.scale);
    const boxWidth = Math.max(1, Math.round(width / prepared.scale));
    const boxHeight = Math.max(1, Math.round(height / prepared.scale));
    const left = Math.max(0, Math.min(prepared.width - 1, x1));
    const top = Math.max(0, Math.min(prepared.height - 1, y1));
    const right = Math.max(0, Math.min(prepared.width - 1, x1 + boxWidth));
    const bottom = Math.max(0, Math.min(prepared.height - 1, y1 + boxHeight));
    if (right <= left || bottom <= top) continue;
    candidatesByClass[classId].push({ classId, score, box: [left, top, right, bottom] });
  }

  const selected = [];
  for (const classCandidates of candidatesByClass) {
    classCandidates.sort((a, b) => b.score - a.score);
    const kept = [];
    while (classCandidates.length && kept.length < 100) {
      const best = classCandidates.shift();
      kept.push(best);
      for (let index = classCandidates.length - 1; index >= 0; index -= 1) {
        if (boxIoU(best.box, classCandidates[index].box) > 0.45) classCandidates.splice(index, 1);
      }
    }
    selected.push(...kept);
  }

  return selected
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((item) => ({
      class_id: item.classId,
      label: CLASS_NAMES[item.classId],
      confidence: Number(item.score.toFixed(4)),
      box: item.box,
    }));
}

async function infer(source, threshold) {
  const started = performance.now();
  const session = await getSession();
  const prepared = makeInput(source);
  const result = await session.run({ [session.inputNames[0]]: prepared.tensor });
  const output = result[session.outputNames[0]];
  const detections = decodeOutput(output, prepared, threshold);
  return {
    width: prepared.width,
    height: prepared.height,
    detections,
    inference_ms: Math.round((performance.now() - started) * 10) / 10,
  };
}

function resetDetections() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  detectionsList.replaceChildren();
  detectionsList.hidden = true;
  emptyState.hidden = false;
  count.textContent = '0';
  latency.textContent = '— ms';
}

function paint(data) {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  const scaleX = overlay.width / data.width;
  const scaleY = overlay.height / data.height;
  overlayContext.lineWidth = Math.max(2, overlay.width / 420);
  overlayContext.font = `600 ${Math.max(14, overlay.width / 42)}px system-ui, sans-serif`;
  overlayContext.textBaseline = 'middle';

  for (const detection of data.detections) {
    const [x1, y1, x2, y2] = detection.box;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    const color = CLASS_COLORS[detection.class_id] || '#2772c7';
    const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    const textWidth = overlayContext.measureText(label).width;
    const labelHeight = Math.max(24, overlay.width / 28);
    const labelY = Math.max(0, y - labelHeight);
    overlayContext.strokeStyle = color;
    overlayContext.fillStyle = color;
    overlayContext.strokeRect(x, y, width, height);
    overlayContext.fillRect(x, labelY, textWidth + 14, labelHeight);
    overlayContext.fillStyle = '#10231a';
    overlayContext.fillText(label, x + 7, labelY + labelHeight / 2);
  }

  count.textContent = String(data.detections.length);
  latency.textContent = `${data.inference_ms} ms`;
  detectionsList.replaceChildren();
  if (data.detections.length === 0) {
    detectionsList.hidden = true;
    emptyState.hidden = false;
    emptyState.innerHTML = '<div><div class="empty-symbol">⌕</div>No se detectan residuos en este cuadro.</div>';
  } else {
    emptyState.hidden = true;
    detectionsList.hidden = false;
    for (const detection of data.detections) {
      const row = document.createElement('li');
      row.className = 'detection-row';
      const bin = disposal[detection.class_id] || disposal[0];
      const info = document.createElement('span');
      info.className = 'det-info';
      const name = document.createElement('span');
      name.className = 'det-name';
      name.textContent = detection.label;
      const destination = document.createElement('span');
      destination.className = 'det-destination';
      destination.textContent = `Tacho ${bin.bin} · ${bin.stream}`;
      info.append(name, destination);
      const score = document.createElement('span');
      score.className = 'det-conf';
      score.textContent = `${Math.round(detection.confidence * 100)}%`;
      row.append(makeBinIcon(detection.class_id), info, score);
      detectionsList.append(row);
    }
  }
  setNotice('Detección en vivo activa. El análisis sigue ejecutándose en este dispositivo.');
}

function prepareFrame() {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const frameWidth = Math.min(sourceWidth, 640);
  const frameHeight = Math.max(1, Math.round(sourceHeight * frameWidth / sourceWidth));
  frameCanvas.width = frameWidth;
  frameCanvas.height = frameHeight;
  frameContext.drawImage(video, 0, 0, frameWidth, frameHeight);
  return frameCanvas;
}

async function inferenceLoop() {
  while (running && cameraStream) {
    try {
      const frame = prepareFrame();
      const data = await infer(frame, Number(confidence.value) / 100);
      if (!running) break;
      paint(data);
    } catch (error) {
      if (running) setNotice(`No se pudo ejecutar la detección: ${error.message || error}`, true);
    }
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setNotice('La cámara requiere una página segura HTTPS y un navegador actualizado.', true);
    return;
  }
  startButton.disabled = true;
  setNotice('Solicitando permiso para usar la cámara…');
  try {
    await getSession();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = cameraStream;
    await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
    await video.play();
    video.hidden = false;
    idle.hidden = true;
    overlay.hidden = false;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    cameraPill.classList.add('live');
    cameraText.textContent = 'Cámara activa';
    stopButton.disabled = false;
    startButton.textContent = 'Cámara activa';
    running = true;
    setNotice('Cámara activa. Apunta un residuo dentro del encuadre.');
    inferenceLoop();
  } catch (error) {
    stopCamera();
    const messages = {
      NotAllowedError: 'No se concedió permiso para la cámara. Permite el acceso en el navegador y vuelve a intentarlo.',
      NotFoundError: 'No se encontró una cámara disponible en este dispositivo.',
      NotReadableError: 'La cámara está ocupada por otra aplicación. Cierra Zoom, Teams u otra app e inténtalo otra vez.',
    };
    setNotice(messages[error.name] || `No se pudo iniciar: ${error.message || error}`, true);
  } finally {
    startButton.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;
  video.hidden = true;
  overlay.hidden = true;
  idle.hidden = false;
  stage.style.aspectRatio = '4 / 3';
  cameraPill.classList.remove('live');
  cameraText.textContent = 'Cámara apagada';
  startButton.textContent = 'Iniciar cámara';
  stopButton.disabled = true;
  resetDetections();
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
confidence.addEventListener('input', () => { confidenceValue.textContent = `${confidence.value}%`; });
window.addEventListener('beforeunload', () => {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
});

renderBinGuide();
checkModel();
