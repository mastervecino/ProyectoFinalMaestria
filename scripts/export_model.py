#!/usr/bin/env python3
"""
Export trained K-Means model and StandardScaler to docs/model.json.

Run from the repo root after re-training (e.g. Modelado/Clustering y PCA.py):

    python scripts/export_model.py \
        --scaler Herramienta/kmeans_scaler_k3_4f.joblib \
        --kmeans  Herramienta/kmeans_model_k3_4f.joblib \
        --output  docs/model.json
"""

import argparse
import json
import os
import sys

import joblib

CLUSTER_METADATA = {
    "0": {
        "label": "Strong Profile",
        "tag": "Website + Training Section",
        "color": "success",
        "successRate": "Highest",
        "description": (
            "Your CV demonstrates a comprehensive structure with strong personalization elements. "
            "It includes a training or courses section and links to external projects or a personal "
            "portfolio — the two differentiating factors associated with the highest candidate success rate."
        ),
        "recommendations": [
            "The structure of your CV is solid — focus now on content quality and relevance.",
            "Make sure your personal website or portfolio is up to date and showcases your best work.",
            "Tailor keywords and language to each specific job posting for maximum impact.",
            "Consider quantifying achievements with metrics (e.g., 'increased sales by 30%').",
        ],
    },
    "1": {
        "label": "Needs Improvement",
        "tag": "Short — Missing Key Sections",
        "color": "danger",
        "successRate": "Lowest",
        "description": (
            "Your CV is currently shorter than average and appears to be missing important sections. "
            "This profile is associated with the lowest success rate in recruitment processes. "
            "Adding structured content and key sections can significantly improve your standing."
        ),
        "recommendations": [
            "Add a Training, Courses, or Certifications section — even short online courses are worth including.",
            "Add links to personal projects, a GitHub profile, or a portfolio website if available.",
            "Expand work experience descriptions with specific responsibilities and measurable achievements.",
            "Ensure all key sections are present: Profile Summary, Work Experience, Education, Skills, Languages.",
            "A longer, more detailed CV signals thoroughness — aim for at least 1–2 full pages.",
        ],
    },
    "2": {
        "label": "Standard Profile",
        "tag": "Training Section — No Website",
        "color": "warning",
        "successRate": "Medium",
        "description": (
            "Your CV has a solid standard structure and includes a training or courses section. "
            "The main opportunity for improvement is adding links to external projects or a personal "
            "portfolio, which is the key differentiator between this profile and the highest-success one."
        ),
        "recommendations": [
            "Add a link to a personal website, GitHub profile, or project portfolio if you have one.",
            "A portfolio link is especially impactful for technical and creative roles.",
            "If your CV feels brief, expand existing sections with more detail and context.",
            "Consider adding a concise professional summary at the top if not already present.",
        ],
    },
}

FEATURE_NAMES = [
    "texto_extraido_len",
    "secciones_completas",
    "Website/Otro",
    "Seccion_training_courses",
]


def export(scaler_path: str, kmeans_path: str, output_path: str) -> None:
    for path, label in [(scaler_path, "scaler"), (kmeans_path, "K-Means model")]:
        if not os.path.isfile(path):
            sys.exit(f"Error: {label} file not found: {path}")

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)

    payload = {
        "scaler": {
            "mean":  scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "kmeans": {
            "n_clusters": int(kmeans.n_clusters),
            "centroids":  kmeans.cluster_centers_.tolist(),
        },
        "features": FEATURE_NAMES,
        "clusters": CLUSTER_METADATA,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Exported model to {output_path}")
    print(f"  Scaler mean  : {scaler.mean_.tolist()}")
    print(f"  Scaler scale : {scaler.scale_.tolist()}")
    print(f"  K-Means K    : {kmeans.n_clusters}")
    print(f"  Centroids    : {kmeans.cluster_centers_.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scaler",  default="Herramienta/kmeans_scaler_k3_4f.joblib")
    parser.add_argument("--kmeans",  default="Herramienta/kmeans_model_k3_4f.joblib")
    parser.add_argument("--output",  default="docs/model.json")
    args = parser.parse_args()
    export(args.scaler, args.kmeans, args.output)


if __name__ == "__main__":
    main()
