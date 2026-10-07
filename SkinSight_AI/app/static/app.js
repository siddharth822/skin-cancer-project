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
    const res = await fetch("/api/predict", { method: "POST", body: form, headers: {"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content} });
    const data = await res.json();
    if (version !== imageVersion) return;
    if (!res.ok) throw new Error(data.detail || "Analysis failed.");
    renderResult(data);
    watchEmail(data.email_status_url, version);
  } catch (e) {
    if (version === imageVersion) showError(e.message);
  } finally {
    analyzeBtn.disabled = !input.files?.[0];
    analyzeBtn.textContent = "Analyze Image";
  }
});

function renderResult(data) {
  document.getElementById("downloadReport").href = data.report_url;
  document.getElementById("emailStatus").textContent = "Preparing email…";
  document.getElementById("emptyResult").hidden = true;
  document.getElementById("result").hidden = false;

  document.getElementById("diagnosis").textContent = data.guidance.name;
  document.getElementById("confidence").textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById("risk").textContent = data.guidance.risk;
  document.getElementById("message").textContent = data.guidance.message;
  document.getElementById("doctorText").textContent = data.guidance.urgency;
  document.getElementById("stageStatus").textContent = data.stage.status;
  document.getElementById("stageText").textContent = data.stage.explanation;
  const guide = document.getElementById("stageGuide");
  guide.replaceChildren();
  (data.stage.education || []).forEach(item => {
    const p = document.createElement("p");
    p.textContent = item;
    guide.appendChild(p);
  });
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


async function watchEmail(url, version) {
  const messages = {accepted: "Gmail accepted your report email. Check your inbox and spam folder.", failed: "Email could not be sent. Download your report here; check the sending Gmail configuration.", not_configured: "Email sending is not configured. Your PDF report is available to download.", verification_required: "Verify your email in Email settings to receive future reports automatically."};
  for (let attempt = 0; attempt < 25; attempt++) {
    if (version !== imageVersion) return;
    try {
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (version !== imageVersion) return;
      if (data.email_status !== "pending") {
        document.getElementById("emailStatus").textContent = messages[data.email_status] || "Download your report here.";
        return;
      }
    } catch (_) {
      if (version === imageVersion) document.getElementById("emailStatus").textContent = "Email status unavailable. Your report can still be downloaded.";
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  if (version === imageVersion) document.getElementById("emailStatus").textContent = "Email is still processing. Your report is available to download.";
}
