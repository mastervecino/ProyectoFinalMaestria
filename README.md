# CV Analyzer — Candidate Profile Assessment

A machine-learning tool that analyzes a résumé (CV) in PDF format and classifies it into one of **3 candidate profiles** discovered through K-Means clustering of 600+ real CVs from the [Manatal](https://www.manatal.com/) recruitment platform.

Built as a Master's thesis in Business Analytics at **Universidad del Rosario**.

---

## Live Web App

> **No installation required.** Upload your CV directly in the browser — all processing happens client-side.

👉 **[Open CV Analyzer](https://mastervecino.github.io/cv_evaluation_system/)**

---

## How It Works

Four structural features are extracted from each CV:

| Feature | Description |
|---|---|
| `texto_extraido_len` | Total character length of extracted text |
| `secciones_completas` | Number of recognized CV sections (out of 11) |
| `Website/Otro` | Presence of a personal website / portfolio link |
| `Seccion_training_courses` | Presence of a training, courses, or certifications section |

These features are scaled and compared against 3 profiles via K-Means (K=3):

| Profile | Key Characteristics | Success Rate |
|---|---|---|
| **Strong Profile** | Website link + Training section | Highest |
| **Standard Profile** | Training section, no website | Medium |
| **Needs Improvement** | Short CV, missing key sections | Lowest |

---

## CLI Tool (Python)

For batch processing or integration into pipelines.

### Setup

```bash
cd Herramienta/

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Usage

```bash
python3 Herramienta.py CV/my_cv.pdf \
    --scaler kmeans_scaler_k3_4f.joblib \
    --kmeans  kmeans_model_k3_4f.joblib
```

**Machine-readable JSON output:**

```bash
python3 Herramienta.py CV/my_cv.pdf \
    --scaler kmeans_scaler_k3_4f.joblib \
    --kmeans  kmeans_model_k3_4f.joblib \
    --json
```

**Options:**

```
positional:
  pdf_path            Path to the CV PDF file.

optional:
  --scaler PATH       Path to the StandardScaler .joblib file.
  --kmeans  PATH      Path to the K-Means model .joblib file.
  --json              Output results as JSON to stdout.
  --log-level LEVEL   DEBUG | INFO | WARNING | ERROR (default: INFO)
```

### Sample output

```
────────────────────────────────────────────────────────
  CV ANALYSIS RESULTS
────────────────────────────────────────────────────────

  File : sample2.pdf

  FEATURES EXTRACTED
    Text length        : 3,241 characters
    Sections detected  : 7 / 11
    Personal website   : No
    Training section   : Yes
    LinkedIn link      : Yes
    GitHub link        : No

  SECTIONS FOUND  : education, work_experience, skills, languages, professional_profile, projects, training_courses

  CLUSTER ASSIGNED  : 2  —  Standard Profile — Training Section Present, No Website
  SUCCESS RATE      : MEDIUM

  RECOMMENDATION

    • Add a link to a personal website, GitHub profile, or project portfolio if you have one.
    • A portfolio link is especially impactful for technical and creative roles.

────────────────────────────────────────────────────────
```

---

## Repository Structure

```
ProyectoFinalMaestria/
├── docs/                          # GitHub Pages web app (static)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── model.json                 # Exported K-Means + scaler parameters
│
├── Herramienta/                   # Python CLI tool
│   ├── Herramienta.py
│   ├── requirements.txt
│   ├── kmeans_model_k3_4f.joblib
│   ├── kmeans_scaler_k3_4f.joblib
│   └── CV/                        # Sample CVs for testing
│
├── scripts/
│   └── export_model.py            # Re-exports model params to docs/model.json
│
├── Bases/                         # Datasets used in the research
├── Modelado/                      # Clustering & PCA analysis
├── Modelos creación CV/           # Feature extraction pipeline
├── Descriptivos/                  # Statistical analysis & model evaluation
└── Analisis de Datos.ipynb        # Data quality analysis notebook
```

---

## Re-training / Updating the Model

If you retrain the K-Means model, regenerate `docs/model.json` so the web app stays in sync:

```bash
python scripts/export_model.py \
    --scaler Herramienta/kmeans_scaler_k3_4f.joblib \
    --kmeans  Herramienta/kmeans_model_k3_4f.joblib \
    --output  docs/model.json
```

---

## Deploying to GitHub Pages

1. Push changes to the `main` branch.
2. Go to **Settings → Pages** in the repository.
3. Set **Source** to `Deploy from a branch`, branch `main`, folder `/docs`.
4. The site will be live at `https://<username>.github.io/<repo>/`.

---

## Dependencies (CLI)

| Package | Purpose |
|---|---|
| PyMuPDF | PDF text and link extraction |
| rapidfuzz | Fuzzy section-header matching |
| scikit-learn | K-Means model & StandardScaler |
| pandas / numpy | Data handling |
| joblib | Model serialization |

The web app has **no dependencies** — it runs entirely in the browser using [PDF.js](https://mozilla.github.io/pdf.js/).

---

## Author

**Manuel Vecino** — Master's in Business Analytics, Universidad del Rosario
https://www.linkedin.com/in/manuelfvecinom/
