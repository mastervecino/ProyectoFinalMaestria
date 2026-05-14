#!/usr/bin/env python3
# coding: utf-8
"""
CV Analyzer — extracts 4 key features from a PDF résumé and assigns it
to one of 3 K-Means clusters identified in the Business Analytics research.

Usage:
    python Herramienta.py CV/my_cv.pdf \
        --scaler kmeans_scaler_k3_4f.joblib \
        --kmeans  kmeans_model_k3_4f.joblib

    # Machine-readable output:
    python Herramienta.py CV/my_cv.pdf --scaler ... --kmeans ... --json
"""

import argparse
import json
import logging
import os
import re
import sys

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.cluster import KMeans          # noqa: F401 – needed by joblib.load
from sklearn.preprocessing import StandardScaler  # noqa: F401 – needed by joblib.load

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,  # keep stderr clean; stdout is reserved for results
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SECTIONS_DICT: dict[str, list[str]] = {
    "education":            ["education", "academic background", "studies", "university studies", "formación académica"],
    "work_experience":      ["experience", "work experience", "employment history", "career history", "professional experience", "experiencia laboral"],
    "skills":               ["skills", "technical skills", "competencies", "habilidades", "conocimientos"],
    "certifications":       ["certifications", "licenses", "accreditations", "certificaciones"],
    "achievements":         ["achievements", "accomplishments", "milestones", "logros"],
    "professional_profile": ["profile", "summary", "about me", "professional summary", "objective", "perfil profesional", "resumen"],
    "languages":            ["languages", "linguistic skills", "spoken languages", "idiomas"],
    "projects":             ["projects", "case studies", "portfolio", "proyectos"],
    "publications":         ["publications", "research papers", "articles", "publicaciones"],
    "training_courses":     ["training", "courses", "workshops", "seminars", "courses and seminars", "other studies", "cursos", "formación complementaria"],
    "volunteer_work":       ["volunteer work", "volunteering", "community service", "voluntariado"],
}

SIMILARITY_THRESHOLD = 75  # RapidFuzz partial_ratio cutoff

FEATURES_ORDER = [
    "texto_extraido_len",
    "secciones_completas",
    "Website/Otro",
    "Seccion_training_courses",
]

CLUSTER_INFO: dict[int, dict] = {
    0: {
        "name": "Strong Profile — With Website & Training Section",
        "success_rate": "highest",
        "suggestion": (
            "The structure of your CV is solid. Focus on content quality and tailoring "
            "keywords to each job posting. Keep your portfolio / personal website up to date."
        ),
    },
    1: {
        "name": "Needs Improvement — Short, Missing Key Sections",
        "success_rate": "lowest",
        "suggestion": (
            "Add a Training / Courses / Certifications section if you have relevant credentials. "
            "Consider adding links to projects or a portfolio. Expand work-experience descriptions "
            "with concrete achievements. Aim for at least 1–2 full pages of content."
        ),
    },
    2: {
        "name": "Standard Profile — Training Section Present, No Website",
        "success_rate": "medium",
        "suggestion": (
            "Add a link to a personal website, GitHub profile, or project portfolio — "
            "that is the main differentiator between this profile and the highest-success one. "
            "If your CV feels brief, expand existing sections with more detail."
        ),
    },
}

# ── Feature extraction ────────────────────────────────────────────────────────

def detect_sections(text: str) -> dict[str, bool]:
    detected = {section: False for section in SECTIONS_DICT}
    if not text:
        return detected

    seen: set[str] = set()
    for line in text.split("\n"):
        clean = re.sub(r"\s+", " ", line.strip()).lower()
        if len(clean) < 3 or clean in seen:
            continue
        for section, synonyms in SECTIONS_DICT.items():
            if detected[section]:
                continue
            match = process.extractOne(
                clean, synonyms,
                scorer=fuzz.partial_ratio,
                score_cutoff=SIMILARITY_THRESHOLD,
            )
            if match:
                detected[section] = True
                seen.add(clean)
                break
    return detected


def extract_links(doc: fitz.Document) -> dict[str, bool]:
    links = {"LinkedIn": False, "GitHub": False, "Website/Otro": False}
    linkedin_re = re.compile(r"linkedin\.com/(in|pub)/", re.I)
    github_re   = re.compile(r"github\.com/", re.I)
    try:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if not uri:
                    continue
                if linkedin_re.search(uri):
                    links["LinkedIn"] = True
                elif github_re.search(uri):
                    links["GitHub"] = True
                elif uri.lower().startswith("http"):
                    links["Website/Otro"] = True
    except Exception as exc:
        log.warning("Could not extract links: %s", exc)
    return links

# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_cv(
    pdf_path: str,
    scaler_path: str,
    kmeans_path: str,
) -> tuple[dict, int | None]:
    """
    Returns (features_dict, cluster_int).
    cluster_int is None if the analysis cannot be completed.
    """
    # Validate inputs
    for path, label in [(pdf_path, "PDF"), (scaler_path, "scaler"), (kmeans_path, "K-Means model")]:
        if not os.path.isfile(path):
            log.error("%s not found: %s", label, path)
            return {}, None

    # Load models
    try:
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        log.info("Models loaded — K=%d", kmeans.n_clusters)
    except Exception as exc:
        log.error("Failed to load models: %s", exc)
        return {}, None

    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        log.error("Cannot open PDF '%s': %s", pdf_path, exc)
        return {}, None

    features: dict = {}
    cluster: int | None = None

    try:
        text = "".join(page.get_text("text") + "\n" for page in doc).strip()
        if not text:
            log.error("No extractable text found in '%s'. The file may be scanned/image-based.", pdf_path)
            return {}, None

        sections = detect_sections(text)
        links    = extract_links(doc)

        features = {
            "texto_extraido_len":      len(text),
            "secciones_completas":     sum(sections.values()),
            "Website/Otro":            int(links["Website/Otro"]),
            "Seccion_training_courses": int(sections["training_courses"]),
            # Extra detail (not used in prediction)
            "LinkedIn":  int(links["LinkedIn"]),
            "GitHub":    int(links["GitHub"]),
            "sections_detected": [k for k, v in sections.items() if v],
        }
        log.info("Features: %s", {k: features[k] for k in FEATURES_ORDER})

        X = pd.DataFrame([features])[FEATURES_ORDER]
        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])
        log.info("Cluster assigned: %d", cluster)

    except Exception as exc:
        log.error("Error during analysis: %s", exc)
    finally:
        doc.close()

    return features, cluster

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a CV PDF and assign it to a K-Means candidate profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path",    help="Path to the CV PDF file.")
    parser.add_argument("--scaler",    required=True, help="Path to the StandardScaler .joblib file.")
    parser.add_argument("--kmeans",    required=True, help="Path to the K-Means model .joblib file.")
    parser.add_argument("--json",      action="store_true", help="Output results as JSON (stdout).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO).")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    features, cluster = analyze_cv(args.pdf_path, args.scaler, args.kmeans)

    if args.json:
        output = {
            "file":    os.path.basename(args.pdf_path),
            "cluster": cluster,
            "features": {k: features.get(k) for k in FEATURES_ORDER} if features else {},
            "links": {
                "LinkedIn":    features.get("LinkedIn"),
                "GitHub":      features.get("GitHub"),
                "Website/Otro": features.get("Website/Otro"),
            } if features else {},
            "sections_detected": features.get("sections_detected", []),
            "profile": CLUSTER_INFO.get(cluster) if cluster is not None else None,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(0 if cluster is not None else 1)

    # Human-readable output
    print("\n" + "─" * 56)
    print("  CV ANALYSIS RESULTS")
    print("─" * 56)

    if not features:
        print("  Analysis failed. See error messages above.")
        sys.exit(1)

    print(f"\n  File : {os.path.basename(args.pdf_path)}")
    print("\n  FEATURES EXTRACTED")
    print(f"    Text length        : {features['texto_extraido_len']:,} characters")
    print(f"    Sections detected  : {features['secciones_completas']} / 11")
    print(f"    Personal website   : {'Yes' if features['Website/Otro'] else 'No'}")
    print(f"    Training section   : {'Yes' if features['Seccion_training_courses'] else 'No'}")
    print(f"    LinkedIn link      : {'Yes' if features.get('LinkedIn') else 'No'}")
    print(f"    GitHub link        : {'Yes' if features.get('GitHub') else 'No'}")

    if features.get("sections_detected"):
        print(f"\n  SECTIONS FOUND  : {', '.join(features['sections_detected'])}")

    if cluster is None:
        print("\n  Cluster could not be assigned.")
        sys.exit(1)

    info = CLUSTER_INFO[cluster]
    print(f"\n  CLUSTER ASSIGNED  : {cluster}  —  {info['name']}")
    print(f"  SUCCESS RATE      : {info['success_rate'].upper()}")
    print(f"\n  RECOMMENDATION\n")
    for line in info["suggestion"].split(". "):
        if line.strip():
            print(f"    • {line.strip()}.")
    print("\n" + "─" * 56)


if __name__ == "__main__":
    main()
