const input = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const fileName = document.getElementById("fileName");
const error = document.getElementById("error");

let imageVersion = 0;

chooseBtn.addEventListener("click", () => input.click());

input.addEventListener("change", () => {
  imageVersion++;
  analyzeBtn.disabled = true;
  clearResult();
  error.hidden = true;
  const file = input.files?.[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("Please select an image file.");
    return;
  }

  if (preview.src.startsWith("blob:")) URL.revokeObjectURL(preview.src);
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  placeholder.hidden = true;
  fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener("click", async () => {
  const file = input.files?.[0];
  if (!file) return;

  clearResult();
  error.hidden = true;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";

  const version = imageVersion;
  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (version !== imageVersion) return;
    if (!res.ok) throw new Error(data.detail || "Analysis failed.");
    renderResult(data);
  } catch (e) {
    if (version === imageVersion) showError(e.message);
  } finally {
    analyzeBtn.disabled = !input.files?.[0];
    analyzeBtn.textContent = "Analyze Image";
  }
});

function renderResult(data) {
  document.getElementById("emptyResult").hidden = true;
  document.getElementById("result").hidden = false;

  document.getElementById("diagnosis").textContent = data.guidance.name;
  document.getElementById("confidence").textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById("risk").textContent = data.guidance.risk;
  document.getElementById("message").textContent = data.guidance.message;
  document.getElementById("doctorText").textContent = data.guidance.urgency;
  document.getElementById("stageStatus").textContent = data.stage.status;
  document.getElementById("stageText").textContent = data.stage.explanation;
  document.getElementById("disclaimer").textContent = data.medical_disclaimer;

  const box = document.getElementById("probabilities");
  box.innerHTML = "";
  data.top_predictions.forEach(p => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-label"><span>${p.label}</span><span>${p.probability.toFixed(1)}%</span></div>
      <div class="track"><div class="fill" style="width:${Math.max(1, p.probability)}%"></div></div>
    `;
    box.appendChild(row);
  });
}

function clearResult() {
  document.getElementById("result").hidden = true;
  document.getElementById("emptyResult").hidden = false;
}

function showError(msg) {
  clearResult();
  error.textContent = msg;
  error.hidden = false;
}
