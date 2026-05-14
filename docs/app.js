'use strict';

// ── PDF.js worker ────────────────────────────────────────────────────────────
// Set worker only if pdfjsLib loaded — error surfaces later inside analyzeFile
if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

// ── Section dictionary (mirrors Herramienta.py) ──────────────────────────────
const SECTIONS_DICT = {
  education:            ['education', 'academic background', 'studies', 'university studies', 'formación académica'],
  work_experience:      ['experience', 'work experience', 'employment history', 'career history', 'professional experience', 'experiencia laboral'],
  skills:               ['skills', 'technical skills', 'competencies', 'habilidades', 'conocimientos'],
  certifications:       ['certifications', 'licenses', 'accreditations', 'certificaciones'],
  achievements:         ['achievements', 'accomplishments', 'milestones', 'logros'],
  professional_profile: ['profile', 'summary', 'about me', 'professional summary', 'objective', 'perfil profesional', 'resumen'],
  languages:            ['languages', 'linguistic skills', 'spoken languages', 'idiomas'],
  projects:             ['projects', 'case studies', 'portfolio', 'proyectos'],
  publications:         ['publications', 'research papers', 'articles', 'publicaciones'],
  training_courses:     ['training', 'courses', 'workshops', 'seminars', 'courses and seminars', 'other studies', 'cursos', 'formación complementaria'],
  volunteer_work:       ['volunteer work', 'volunteering', 'community service', 'voluntariado'],
};

const SIMILARITY_THRESHOLD = 75; // same as Python tool

// ── Fuzzy matching ────────────────────────────────────────────────────────────

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  // Use two rows to save memory
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array(n + 1);
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      curr[j] = a[i - 1] === b[j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j], curr[j - 1], prev[j - 1]);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

function ratio(s1, s2) {
  if (s1.length === 0 && s2.length === 0) return 100;
  const dist = levenshtein(s1, s2);
  return Math.round((1 - dist / Math.max(s1.length, s2.length)) * 100);
}

// Replicates RapidFuzz's partial_ratio: slide the shorter string over the longer.
function partialRatio(query, candidate) {
  if (!query || !candidate) return 0;
  // Fast path: substring containment
  if (candidate.includes(query) || query.includes(candidate)) return 100;

  const shorter = query.length <= candidate.length ? query : candidate;
  const longer  = query.length <= candidate.length ? candidate : query;
  if (shorter.length === longer.length) return ratio(shorter, longer);

  let best = 0;
  for (let i = 0; i <= longer.length - shorter.length; i++) {
    const sub = longer.slice(i, i + shorter.length);
    const r = ratio(shorter, sub);
    if (r > best) best = r;
    if (best === 100) break;
  }
  return best;
}

// ── Section detection ─────────────────────────────────────────────────────────

function detectSections(text) {
  const detected = Object.fromEntries(Object.keys(SECTIONS_DICT).map(k => [k, false]));
  const lines = text.split('\n');
  const processed = new Set();

  for (const line of lines) {
    const clean = line.replace(/\s+/g, ' ').trim().toLowerCase();
    if (clean.length < 3 || processed.has(clean)) continue;

    let matched = false;
    for (const [section, synonyms] of Object.entries(SECTIONS_DICT)) {
      if (detected[section]) continue; // already found
      for (const synonym of synonyms) {
        if (partialRatio(clean, synonym) >= SIMILARITY_THRESHOLD) {
          detected[section] = true;
          processed.add(clean);
          matched = true;
          break;
        }
      }
      if (matched) break;
    }
  }
  return detected;
}

// ── PDF extraction ────────────────────────────────────────────────────────────

async function extractFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  let fullText = '';
  const urls = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);

    // Text
    const content = await page.getTextContent();
    // Preserve line breaks using transform y-coordinate heuristic
    let lastY = null;
    for (const item of content.items) {
      if (lastY !== null && Math.abs(item.transform[5] - lastY) > 5) {
        fullText += '\n';
      }
      fullText += item.str;
      lastY = item.transform[5];
    }
    fullText += '\n';

    // Links from PDF annotations
    const annotations = await page.getAnnotations();
    for (const ann of annotations) {
      if (ann.subtype === 'Link') {
        const url = ann.url || ann.action?.url;
        if (url) urls.push(url);
      }
    }
  }

  return { text: fullText.trim(), urls };
}

// ── Feature computation ───────────────────────────────────────────────────────

const LINKEDIN_RE = /linkedin\.com\/(in|pub)\//i;
const GITHUB_RE   = /github\.com\//i;

function computeFeatures(text, urls) {
  const sections = detectSections(text);
  const sectionCount = Object.values(sections).filter(Boolean).length;

  const hasLinkedIn = urls.some(u => LINKEDIN_RE.test(u));
  const hasGitHub   = urls.some(u => GITHUB_RE.test(u));
  const hasWebsite  = urls.some(u => u.startsWith('http') && !LINKEDIN_RE.test(u) && !GITHUB_RE.test(u));

  return {
    texto_extraido_len:    text.length,
    secciones_completas:   sectionCount,
    'Website/Otro':        hasWebsite ? 1 : 0,
    Seccion_training_courses: sections.training_courses ? 1 : 0,
    // Extra context for display only (not used in prediction)
    _sections: sections,
    _links: { linkedin: hasLinkedIn, github: hasGitHub, website: hasWebsite },
  };
}

// ── K-Means prediction ────────────────────────────────────────────────────────

function predictCluster(features, model) {
  const { mean, scale } = model.scaler;
  const featureOrder = model.features;

  // StandardScaler: z = (x - mean) / scale
  const scaled = featureOrder.map((f, i) => (features[f] - mean[i]) / scale[i]);

  // Nearest centroid (Euclidean distance in scaled space)
  let minDist = Infinity;
  let cluster = 0;
  const distances = model.kmeans.centroids.map((centroid, i) => {
    const dist = Math.sqrt(centroid.reduce((sum, c, j) => sum + (scaled[j] - c) ** 2, 0));
    if (dist < minDist) { minDist = dist; cluster = i; }
    return dist;
  });

  // Confidence proxy: how much closer to the winner vs second nearest (0–100)
  const sorted = [...distances].sort((a, b) => a - b);
  const confidence = sorted.length > 1
    ? Math.min(100, Math.round(((sorted[1] - sorted[0]) / sorted[1]) * 100))
    : 100;

  return { cluster, distances, confidence };
}

// ── UI rendering ──────────────────────────────────────────────────────────────

function icon(type) {
  const icons = {
    check: '✓',
    dash:  '—',
    dot:   '◆',
    warn:  '!',
  };
  return icons[type] ?? '';
}

function renderResults(cluster, features, confidence, modelClusters) {
  const info = modelClusters[String(cluster)];
  const colorClass = info.color; // 'success' | 'warning' | 'danger'

  const detectedSections = Object.entries(features._sections)
    .filter(([, v]) => v)
    .map(([k]) => k.replace(/_/g, ' '));

  const textLenGood = features.texto_extraido_len >= 1000;

  return `
    <div class="result-card">

      <div class="result-header ${colorClass}">
        <div class="cluster-badge">
          <span class="cluster-label">${info.label}</span>
          <span class="cluster-tag">${info.tag}</span>
        </div>
        <div class="success-block">
          <span class="success-rate-label">Success Rate</span>
          <span class="success-rate-value" style="color: var(--${colorClass === 'success' ? 'success' : colorClass === 'warning' ? 'warning' : 'danger'})">${info.successRate}</span>
        </div>
      </div>

      <div class="result-body">

        <p class="result-description">${info.description}</p>

        <div class="features-grid">
          <div class="feature-card ${textLenGood ? 'feat-good' : 'feat-warn'}">
            <div class="feature-icon">${textLenGood ? icon('check') : icon('warn')}</div>
            <div class="feature-name">Text Length</div>
            <div class="feature-value">${features.texto_extraido_len.toLocaleString()} chars</div>
          </div>
          <div class="feature-card feat-neutral">
            <div class="feature-icon">${icon('dot')}</div>
            <div class="feature-name">Sections</div>
            <div class="feature-value">${features.secciones_completas} / 11</div>
          </div>
          <div class="feature-card ${features['Website/Otro'] ? 'feat-good' : 'feat-neutral'}">
            <div class="feature-icon">${features['Website/Otro'] ? icon('check') : icon('dash')}</div>
            <div class="feature-name">Website Link</div>
            <div class="feature-value">${features['Website/Otro'] ? 'Detected' : 'Not found'}</div>
          </div>
          <div class="feature-card ${features.Seccion_training_courses ? 'feat-good' : 'feat-neutral'}">
            <div class="feature-icon">${features.Seccion_training_courses ? icon('check') : icon('dash')}</div>
            <div class="feature-name">Training Section</div>
            <div class="feature-value">${features.Seccion_training_courses ? 'Present' : 'Not found'}</div>
          </div>
        </div>

        ${detectedSections.length > 0 ? `
        <div class="detected-sections">
          <h4>Sections identified (${detectedSections.length})</h4>
          <div class="section-tags">
            ${detectedSections.map(s => `<span class="section-tag">${s}</span>`).join('')}
          </div>
        </div>` : ''}

        <div class="links-info">
          <h4>Links detected</h4>
          <div class="links-badges">
            ${features._links.linkedin ? '<span class="link-badge linkedin">LinkedIn</span>' : ''}
            ${features._links.github   ? '<span class="link-badge github">GitHub</span>'    : ''}
            ${features._links.website  ? '<span class="link-badge website">Website</span>'  : ''}
            ${!features._links.linkedin && !features._links.github && !features._links.website
              ? '<span class="link-badge none">No hyperlinks found in PDF</span>' : ''}
          </div>
        </div>

        <div class="recommendations">
          <h3>Recommendations</h3>
          <ul>
            ${info.recommendations.map(r => `<li>${r}</li>`).join('')}
          </ul>
        </div>

        <button class="analyze-again-btn" id="analyzeAgainBtn">Analyze another CV</button>

      </div>
    </div>`;
}

function renderError(message) {
  return `
    <div class="error-card">
      <div class="error-icon">⚠</div>
      <div>
        <p class="error-title">Could not analyze this PDF</p>
        <p class="error-msg">${message}</p>
        <button class="error-retry" id="retryBtn">Try another file</button>
      </div>
    </div>`;
}

// ── State management ──────────────────────────────────────────────────────────

let model = null;

function showSection(id) {
  ['uploadSection', 'processingSection', 'resultsSection'].forEach(s => {
    const el = document.getElementById(s);
    if (el) el.hidden = (s !== id);
  });
}

function resetUI() {
  showSection('uploadSection');
  document.getElementById('fileInput').value = '';
}

async function analyzeFile(file) {
  if (!file) return;

  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please upload a PDF file.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert('File is too large. Please upload a PDF under 10 MB.');
    return;
  }

  showSection('processingSection');

  try {
    if (typeof pdfjsLib === 'undefined') {
      throw new Error('PDF.js failed to load. Please check your connection and refresh the page.');
    }

    // Load model on first run
    if (!model) {
      const res = await fetch('model.json');
      if (!res.ok) throw new Error('Could not load the analysis model.');
      model = await res.json();
    }

    const { text, urls } = await extractFromPDF(file);

    if (!text || text.trim().length < 10) {
      throw new Error('No readable text found in this PDF. The file may be scanned or image-based.');
    }

    const features = computeFeatures(text, urls);
    const { cluster, confidence } = predictCluster(features, model);

    const resultsEl = document.getElementById('resultsSection');
    resultsEl.innerHTML = renderResults(cluster, features, confidence, model.clusters);
    showSection('resultsSection');
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Wire up "analyze again" button
    document.getElementById('analyzeAgainBtn')?.addEventListener('click', resetUI);

  } catch (err) {
    console.error(err);
    const resultsEl = document.getElementById('resultsSection');
    resultsEl.innerHTML = renderError(err.message || 'An unexpected error occurred.');
    showSection('resultsSection');
    document.getElementById('retryBtn')?.addEventListener('click', resetUI);
  }
}

// ── Initialise ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Pre-fetch the model so the first analysis is instant
  fetch('model.json')
    .then(r => r.json())
    .then(m => { model = m; })
    .catch(() => {}); // non-fatal; will retry on first analysis

  const dropZone  = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  // Keyboard accessibility for the drop zone
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) analyzeFile(fileInput.files[0]);
  });

  // Drag & drop
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', e => {
    if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });

  // Allow drag over the whole page
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });
});
