const input = document.getElementById('fileInput');
const chooseBtn = document.getElementById('chooseBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const preview = document.getElementById('preview');
const placeholder = document.getElementById('placeholder');
const fileName = document.getElementById('fileName');
const error = document.getElementById('error');
const dropzone = document.getElementById('dropzone');
const overlay = document.getElementById('loadingOverlay');
let imageVersion = 0;
let selectedFile = null;
let busy = false;

if (input) {
  const choose = () => { if (!busy) input.click(); };
  chooseBtn.addEventListener('click', choose);
  dropzone.addEventListener('click', choose);
  dropzone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
  });
  input.addEventListener('change', () => selectFile(input.files?.[0]));
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault(); if (!busy) dropzone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault(); dropzone.classList.remove('dragging');
  }));
  dropzone.addEventListener('drop', event => { if (!busy) selectFile(event.dataTransfer.files?.[0]); });
  analyzeBtn.addEventListener('click', analyze);
}

function selectFile(file) {
  if (!file || busy) return;
  imageVersion++;
  selectedFile = null;
  analyzeBtn.disabled = true;
  clearResult();
  error.hidden = true;
  if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
  preview.removeAttribute('src'); preview.hidden = true; placeholder.hidden = false; fileName.textContent = '';
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { showError('Choose a JPG, PNG or WEBP skin photo.'); return; }
  if (file.size > 10 * 1024 * 1024) { showError('Please choose an image no larger than 10 MB.'); return; }
  selectedFile = file;
  preview.src = URL.createObjectURL(file); preview.hidden = false; placeholder.hidden = true;
  fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  analyzeBtn.disabled = false;
}

async function analyze() {
  if (!selectedFile || busy) return;
  busy = true;
  const version = imageVersion;
  clearResult(); error.hidden = true;
  analyzeBtn.disabled = true; chooseBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…'; overlay.hidden = false;
  document.querySelector('.analysis-grid').setAttribute('aria-busy', 'true');
  const form = new FormData(); form.append('image', selectedFile);
  try {
    const response = await fetch('/api/predict', {method: 'POST', body: form, headers: {'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();
    if (version !== imageVersion) return;
    if (!response.ok) throw new Error(data.detail || 'Analysis failed. Please try again.');
    renderResult(data); watchEmail(data.email_status_url, version);
  } catch (failure) {
    if (version === imageVersion) showError(failure.message);
  } finally {
    busy = false; overlay.hidden = true;
    document.querySelector('.analysis-grid').removeAttribute('aria-busy');
    analyzeBtn.disabled = !selectedFile; chooseBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze image →';
  }
}

function renderResult(data) {
  document.getElementById('downloadReport').href = data.report_url;
  document.getElementById('emailStatus').textContent = 'Checking email status…';
  document.getElementById('emptyResult').hidden = true;
  document.getElementById('result').hidden = false;
  document.getElementById('diagnosis').textContent = data.guidance.name;
  document.getElementById('confidence').textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById('risk').textContent = data.guidance.risk;
  document.getElementById('message').textContent = data.guidance.message;
  document.getElementById('doctorText').textContent = data.guidance.urgency;
  document.getElementById('stageStatus').textContent = data.stage.status;
  document.getElementById('stageText').textContent = data.stage.explanation;
  const guide = document.getElementById('stageGuide'); guide.replaceChildren();
  (data.stage.education || []).forEach(item => {
    const p = document.createElement('p'); p.textContent = item; guide.appendChild(p);
  });
  document.getElementById('disclaimer').textContent = data.medical_disclaimer;
  const names = {ACK: 'Actinic keratosis', BCC: 'Basal cell carcinoma', MEL: 'Melanoma', NEV: 'Melanocytic nevus', SCC: 'Squamous cell carcinoma', SEK: 'Benign / seborrheic keratosis'};
  const box = document.getElementById('probabilities'); box.replaceChildren();
  data.top_predictions.forEach(item => {
    const row = document.createElement('div'); row.className = 'prob-row';
    const label = document.createElement('div'); label.className = 'prob-label';
    const name = document.createElement('span'); name.textContent = names[item.label] || item.label;
    const score = document.createElement('span'); score.textContent = `${item.probability.toFixed(1)}%`;
    label.append(name, score);
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${Math.max(0, Math.min(100, item.probability))}%`;
    track.appendChild(fill); row.append(label, track); box.appendChild(row);
  });
}

function clearResult() {
  document.getElementById('result').hidden = true;
  document.getElementById('emptyResult').hidden = false;
}
function showError(message) { clearResult(); error.textContent = message; error.hidden = false; }

async function watchEmail(url, version) {
  const messages = {auth_failed: 'Gmail rejected the sender login. Check the saved sender address and Google App Password.', recipient_refused: 'Check your recipient email address in Email settings.', sender_refused: 'Gmail rejected the sending address. Check sender configuration.', tls_failed: 'Secure Gmail connection failed. Check computer date/time and network.', accepted: 'Gmail accepted your report email. Check your inbox and spam folder.', failed: 'Email could not be sent. You can still download the PDF. Check the sending Gmail configuration.', not_configured: 'Email sending is not configured. Your PDF is ready to download.', email_required: 'Add a report email in Email settings to receive future reports.'};
  for (let attempt = 0; attempt < 25; attempt++) {
    if (version !== imageVersion) return;
    try {
      const response = await fetch(url, {cache: 'no-store'});
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (version !== imageVersion) return;
      if (data.email_status !== 'pending') {
        document.getElementById('emailStatus').textContent = messages[data.email_status] || 'Your report is ready to download.';
        return;
      }
    } catch (_) {
      if (version === imageVersion) document.getElementById('emailStatus').textContent = 'Email status unavailable. Your PDF can still be downloaded.';
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  if (version === imageVersion) document.getElementById('emailStatus').textContent = 'Email is still processing. Your PDF is ready to download.';
}
if (window.savedReport) { renderResult(window.savedReport); watchEmail(window.savedReport.email_status_url, imageVersion); }
