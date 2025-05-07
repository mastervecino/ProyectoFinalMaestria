#!/usr/bin/env python
# coding: utf-8

# # Script para Extraer Features Clave y Asignar Cluster K-Means

import os
import fitz  # PyMuPDF
import re
from rapidfuzz import process, fuzz
import numpy as np
import pandas as pd
import joblib # Para cargar modelos guardados
from sklearn.preprocessing import StandardScaler # Para cargar el scaler
from sklearn.cluster import KMeans # Para cargar el modelo KMeans
import argparse # Para manejar argumentos de línea de comandos
import logging

# Configuración del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes y Definiciones ---
# (Traídas del script anterior, ajustadas si es necesario)

# Diccionario de Secciones (Asegúrate que 'training_courses' esté bien definido)
SECCIONES_CV_DICT = {
    "education": ["education", "academic background", "studies", "university studies", "formación académica"],
    "work_experience": ["experience", "work experience", "employment history", "career history", "experiencia laboral", "professional experience"],
    "skills": ["skills", "technical skills", "competencies", "habilidades", "conocimientos"],
    "certifications": ["certifications", "licenses", "accreditations", "certificaciones"],
    "achievements": ["achievements", "accomplishments", "milestones", "logros"],
    "professional_profile": ["profile", "summary", "about me", "professional summary", "objective", "perfil profesional", "resumen"],
    "languages": ["languages", "linguistic skills", "spoken languages", "idiomas"],
    "projects": ["projects", "case studies", "portfolio", "proyectos"],
    "publications": ["publications", "research papers", "articles", "books", "publicaciones"],
    "training_courses": ["training", "courses", "workshops", "seminars", "courses and seminars", "Other Studies", "cursos", "formación complementaria"], # Clave importante
    "volunteer_work": ["volunteer work", "volunteering", "community service", "social impact", "non-profit", "voluntariado"],
}
UMBRAL_SIMILITUD_SECCION = 75 # Mismo umbral que antes

# Nombres exactos de las 4 features usadas para entrenar K-Means K=3
# Deben coincidir con las columnas del DataFrame usado para ajustar el scaler/kmeans
FEATURES_FOR_CLUSTERING = [
    'texto_extraido_len',
    'secciones_completas',
    'Website/Otro',
    'Seccion_training_courses'
]

# --- Funciones de Extracción Necesarias ---

def detectar_secciones_texto(texto: str, secciones_dict: dict, umbral: int) -> dict:
    """Detecta la presencia de secciones clave y retorna un dict booleano."""
    secciones_detectadas = {seccion: False for seccion in secciones_dict}
    if not texto: return secciones_detectadas
    lineas = texto.split("\n")
    lineas_procesadas_header = set()
    for linea in lineas:
        linea_limpia = re.sub(r"\s+", " ", linea.strip()).lower()
        if len(linea_limpia) < 3 or linea_limpia in lineas_procesadas_header: continue
        # Simplificado: buscar en todas las líneas (no solo cortas) para robustez
        for seccion, sinonimos in secciones_dict.items():
            resultado = process.extractOne(linea_limpia, sinonimos, scorer=fuzz.partial_ratio, score_cutoff=umbral)
            if resultado:
                secciones_detectadas[seccion] = True
                lineas_procesadas_header.add(linea_limpia)
                break # Asignar solo a la primera sección que coincida en una línea
    return secciones_detectadas

def extract_links_fitz(doc: fitz.Document) -> dict:
    """Extrae links y clasifica LinkedIn, GitHub y otros."""
    links_encontrados = {"LinkedIn": False, "GitHub": False, "Website/Otro": False}
    if not doc: return links_encontrados
    linkedin_pattern = r"linkedin\.com\/(?:in|pub)\/" # Más específico
    github_pattern = r"github\.com\/"
    try:
        for page in doc:
            page_links = page.get_links()
            for link in page_links:
                uri = link.get("uri")
                if uri:
                    uri_lower = uri.lower()
                    is_linkedin = bool(re.search(linkedin_pattern, uri_lower))
                    is_github = bool(re.search(github_pattern, uri_lower))
                    if is_linkedin:
                        links_encontrados["LinkedIn"] = True
                    if is_github:
                         links_encontrados["GitHub"] = True
                    # Si es un link web Y NO es LinkedIn ni GitHub
                    if uri_lower.startswith("http") and not is_linkedin and not is_github:
                         links_encontrados["Website/Otro"] = True
            # Podríamos parar antes si ya encontramos todos, pero es rápido
    except Exception as e:
        logging.warning(f"Error extrayendo links: {e}")
    return links_encontrados

# --- Función Principal ---

def analizar_cv_y_asignar_cluster(pdf_path: str, scaler_path: str, kmeans_path: str) -> tuple[dict, int | None]:
    """
    Extrae las 4 features clave de un CV y le asigna un cluster K-Means pre-entrenado.

    Args:
        pdf_path: Ruta al archivo PDF del CV.
        scaler_path: Ruta al archivo .joblib del StandardScaler entrenado.
        kmeans_path: Ruta al archivo .joblib del modelo KMeans entrenado (K=3).

    Returns:
        Una tupla: (diccionario_features, cluster_asignado)
        El cluster_asignado es None si ocurre un error.
    """
    extracted_features = {}
    cluster_asignado = None

    # 1. Cargar modelos
    try:
        scaler = joblib.load(scaler_path)
        kmeans_model = joblib.load(kmeans_path)
        logging.info("Scaler y modelo K-Means cargados exitosamente.")
        # Verificar que el modelo KMeans tenga 3 clusters
        if kmeans_model.n_clusters != 3:
             logging.warning(f"El modelo K-Means cargado tiene {kmeans_model.n_clusters} clusters, se esperaban 3.")
             # Podríamos continuar o detenernos, por ahora continuamos.

    except FileNotFoundError:
        logging.error(f"Error: No se encontró el scaler en '{scaler_path}' o el kmeans en '{kmeans_path}'.")
        return extracted_features, None
    except Exception as e_load:
        logging.error(f"Error cargando los modelos guardados: {e_load}")
        return extracted_features, None

    # 2. Procesar PDF
    doc = None
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Procesando PDF: {os.path.basename(pdf_path)}")

        # Extraer texto completo
        texto_completo = ""
        for page in doc:
            texto_completo += page.get_text("text") + "\n"
        texto_completo = texto_completo.strip()
        extracted_features['texto_extraido_len'] = len(texto_completo)

        # Extraer secciones y links
        secciones_detectadas = detectar_secciones_texto(texto_completo, SECCIONES_CV_DICT, UMBRAL_SIMILITUD_SECCION)
        links_detectados = extract_links_fitz(doc)

        # Calcular features necesarias
        extracted_features['secciones_completas'] = sum(secciones_detectadas.values())
        extracted_features['Seccion_training_courses'] = int(secciones_detectadas.get('training_courses', False)) # Convertir a 0/1
        extracted_features['Website/Otro'] = int(links_detectados.get('Website/Otro', False)) # Convertir a 0/1

        logging.info(f"Features extraídas: {extracted_features}")

        # 3. Preparar datos para predicción
        # Crear un DataFrame con UNA fila y las 4 features en el ORDEN CORRECTO
        # Es CRUCIAL que el orden aquí coincida con el orden usado para entrenar el scaler/kmeans
        try:
            current_cv_data = pd.DataFrame([extracted_features])[FEATURES_FOR_CLUSTERING]
        except KeyError as e_key:
             logging.error(f"Error: Falta una feature necesaria en los datos extraídos: {e_key}")
             logging.error(f"Asegúrate que FEATURES_FOR_CLUSTERING ({FEATURES_FOR_CLUSTERING}) coincida con las claves de 'extracted_features'.")
             return extracted_features, None

        # 4. Escalar features
        X_scaled = scaler.transform(current_cv_data)

        # 5. Predecir cluster
        cluster_asignado = kmeans_model.predict(X_scaled)[0] # predict devuelve un array, tomamos el primer elemento
        logging.info(f"Cluster asignado: {cluster_asignado}")

    except Exception as e_process:
        logging.error(f"Error procesando el PDF '{pdf_path}': {e_process}")
        # Devolver features extraídas hasta el momento, pero cluster None
        return extracted_features, None
    finally:
        if doc:
            doc.close()

    return extracted_features, cluster_asignado


# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Extrae features clave de un CV y asigna un cluster K-Means.')
    parser.add_argument('pdf_path', type=str, help='Ruta al archivo PDF del CV a analizar.')
    parser.add_argument('--scaler', type=str, required=True, help='Ruta al archivo .joblib del StandardScaler entrenado.')
    parser.add_argument('--kmeans', type=str, required=True, help='Ruta al archivo .joblib del modelo KMeans entrenado (K=3).')

    args = parser.parse_args()

    # Ejecutar análisis y asignación
    features, cluster = analizar_cv_y_asignar_cluster(args.pdf_path, args.scaler, args.kmeans)

    print("\n--- Resultados del Análisis ---")
    if features:
        print("Features Extraídas:")
        for key, value in features.items():
            # Solo mostrar las 4 usadas para clustering para claridad
            if key in FEATURES_FOR_CLUSTERING:
                 print(f"  - {key}: {value}")
    else:
        print("No se pudieron extraer features.")

    if cluster is not None:
        print(f"\nCluster Asignado (K=3): {cluster}")
        # Aquí podrías añadir una interpretación basada en el cluster
        if cluster == 0:
            print("Cluster 0: Con Website y Sección Training.\n"
                  "\n"
                  "Sugerencia: Revisa el contenido de la CV del candidato, su formato en general debería estar bien")
        elif cluster == 1:
            print("Cluster 1:  Sin Training (y Mayormente sin Website) - Corto\n"
                  "\n"
                  "Sugerencia: Ayúdale al candidato a brindarle más contenido a su CV, si es posible agregar una sección de entrenamiento,\n cursos o certificaciones en caso de no tenerla.\n"
                  "Sugiere añadir links a websites externos de proyectos anteriores o inclusive un portafolio si lo tiene")
        elif cluster == 2:
             print("Cluster 2: Con Sección Training, Sin Website (Grupo Estándar)\n"
                   "\n"
                   "Sugerencia: Mencionale al candidato si es posible agregar links a proyectos pasados o un portafolio si li tiene.\n"
                   "Su formato debería ser bastante estándar, por lo que puedes sugerir también, incrementar la longitud del CV si se ve necesario")
        # Añadir Cluster 4 si K fuera 5, etc.
    else:
        print("\nNo se pudo asignar un cluster debido a errores.")