#!/usr/bin/env python
# coding: utf-8

# # Script para extracción de información de las CVs

# ##### Obtener librerías necesarias

# In[137]:


import os
import fitz  # PyMuPDF for PDFs
import pytesseract
from PIL import Image
import pandas as pd
import re
import spacy
import pdfplumber
from rapidfuzz import process, fuzz
from dateutil import parser
from collections import Counter
import io
import numpy as np
import cv2
import PyPDF2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import language_tool_python
from spellchecker import SpellChecker


# ##### Importar SpaCy para NLP
# Spacy es una librería de python que permite por medio de modelos de lenguaje pre-importados realizar análisis de texto, identificando palabras, nombres, lugares, objetos, verbos, adjetivos y la relación entre los mismos.
# 
# En este caso, importamos el modelo pre-entrenado en inglés, lo que requiere que todas las CVs a procesar estén en este idioma.

# In[20]:


nlp = spacy.load("en_core_web_sm")


# #### Obtenemos las CVs a estudiar

# In[21]:


hv_dir_exitosas = "hojas_de_vida_copy/Paso"
hv_dir_noexitosas = "hojas_de_vida_copy/No paso"


# #### Extraer el texto

# In[22]:


def extraer_texto_pdf(pdf_path):
    """Extrae texto de un PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        texto = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return texto


# ## Estructura y Organización

# 
# #### Extensión del CV: Contar páginas

# In[23]:


def contar_paginas(pdf_path):
    with fitz.open(pdf_path) as pdf:
        return len(pdf)


# ### Uso de secciones bien definidas - Extraer secciones

# In[48]:


def detectar_secciones(texto):
    """Detecta si el CV tiene secciones bien definidas."""
    secciones_dict = {
        "education": ["education", "academic background", "studies", "university studies"],
        "work_experience": ["experience", "work experience", "employment history", "career history"],
        "skills": ["skills", "technical skills", "competencies"],
        "certifications": ["certifications", "licenses", "accreditations"],
        "achievements": ["achievements", "accomplishments", "milestones"],
        "professional_profile": ["profile", "summary", "about me", "professional summary", "objective"],
        "languages": ["languages", "linguistic skills", "spoken languages"],
        "projects": ["projects", "case studies", "portfolio"],
        "publications": ["publications", "research papers", "articles", "books"],
        "training_courses": ["training", "courses", "workshops", "seminars", "courses and seminars", "Other Studies"],
        "volunteer_work": ["volunteer work", "volunteering", "community service", "social impact", "non-profit"],
    }

    lineas = texto.split("\n")
    secciones_detectadas = {seccion: False for seccion in secciones_dict}  # Inicializamos todas en False

    for i, linea in enumerate(lineas):
        linea_limpia = re.sub(r"\s+", " ", linea.strip())  # 🔹 Limpia espacios innecesarios

        for seccion, sinonimos in secciones_dict.items():
            resultado = process.extractOne(linea_limpia, sinonimos, scorer=fuzz.partial_ratio)
            if resultado:
                mejor_coincidencia, score, *_ = resultado

                if score >= 75:  # 🔹 Bajamos umbral a 75 para mayor flexibilidad
                    secciones_detectadas[seccion] = True  # Marcamos como encontrada
                    break  # Pasamos a la siguiente línea para evitar duplicados

    return secciones_detectadas


# ### Orden Cronológico o funcional de la información

# In[56]:


# Expresiones regulares mejoradas para fechas
patrones_fechas = [
    r"\b\d{4}\b",  # Años sueltos (ej. 2020)
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",  # Mes Año (ej. Mar 2020)
    r"\b(?:Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)\s\d{4}\b",  # Español
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # dd/mm/yyyy
    r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # dd-mm-yyyy
    r"\b\d{4}[-–]\d{4}\b",  # Rango de años (ej. 2018-2020)
    r"\b\d{4}[-–]\s*(?:Presente|Actual|Present)\b",  # Ejemplo: 2019 - Presente
]

def extraer_fechas(texto):
    """Extrae fechas de un texto usando regex y NLP."""
    fechas_encontradas = set()

    # Usar expresiones regulares para detectar fechas
    for patron in patrones_fechas:
        coincidencias = re.findall(patron, texto, re.IGNORECASE)
        fechas_encontradas.update(coincidencias)

    # Usar spaCy para extraer entidades de fecha
    doc = nlp(texto)
    for ent in doc.ents:
        if ent.label_ in ["DATE"]:
            fechas_encontradas.add(ent.text)

    # Convertir a objetos datetime
    fechas_convertidas = []
    for fecha in fechas_encontradas:
        try:
            if "presente" in fecha.lower() or "actual" in fecha.lower() or "present" in fecha.lower():
                fecha_convertida = "Presente"
            else:
                fecha_convertida = parser.parse(fecha, fuzzy=True)
            fechas_convertidas.append(fecha_convertida)
        except:
            pass

    # Ordenar fechas (coloca "Presente" como fecha más reciente)
    fechas_convertidas = sorted(
        fechas_convertidas, key=lambda x: x if isinstance(x, str) else x, reverse=True
    )

    return fechas_convertidas

def determinar_orden_cronologico(texto):
    """Determina si el CV sigue un orden cronológico inverso o no."""
    fechas = extraer_fechas(texto)

    if not fechas or len(fechas) < 2:
        return "No se detectaron suficientes fechas"

    # Verificamos si las fechas están en orden descendente (cronológico)
    fechas_numericas = [f for f in fechas if isinstance(f, str) or hasattr(f, "year")]

    if fechas_numericas == sorted(fechas_numericas, reverse=True):
        return "Orden Cronológico"
    else:
        return "Orden Funcional"


# ### Uso de viñetas (bullet-points) o parrafos para escribir la información

# In[63]:


def detectar_formato_texto(texto):
    """
    Determina si el CV usa principalmente viñetas, párrafos o es mixto.
    """
    lineas = texto.split("\n")
    total_lineas = len([l for l in lineas if l.strip()])
    lineas_vinetas = 0
    lineas_parrafos = 0

    # Expresión regular para detectar viñetas
    regex_vinetas = r"^(•|-|\*|▪|○|\d+\.)\s+"

    for linea in lineas:
        linea_limpia = linea.strip()

        if not linea_limpia:
            continue  # Ignorar líneas vacías

        palabras = linea_limpia.split()
        num_palabras = len(palabras)
        num_caracteres = len(linea_limpia)

        # Si la línea coincide con una viñeta, sumamos al contador de viñetas
        if re.match(regex_vinetas, linea_limpia) or num_palabras < 8:
            lineas_vinetas += 1
        elif num_caracteres > 30 and num_palabras > 12:  # Mayor umbral para considerar párrafo
            lineas_parrafos += 1

    # Decidir clasificación según la proporción
    if lineas_vinetas / total_lineas > 0.5:
        return "Viñetas"
    elif lineas_parrafos / total_lineas > 0.5:
        return "Párrafos"
    else:
        return "Mixto"


# ## Diseño y Estilo de la CV

# ### Tipografía y tamaño de fuente (legibilidad, uso de negritas o cursiva)

# In[77]:


import pdfplumber
from collections import Counter

def analizar_tipografia(pdf_path):
    """
    Extrae información sobre la tipografía, tamaño de fuente y uso de negritas/cursivas en un PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        fuentes = []
        negritas = 0
        cursivas = 0

        for page in pdf.pages:
            for block in page.extract_words(extra_attrs=["fontname", "size"]):
                fuente = block["fontname"]
                tamano = round(block["size"], 1)  # Redondeamos el tamaño de fuente

                fuentes.append((fuente, tamano))

                # Detectar negrita y cursiva (puede variar según el PDF)
                if "Bold" in fuente or "Black" in fuente:
                    negritas += 1
                if "Italic" in fuente or "Oblique" in fuente:
                    cursivas += 1

        # Análisis de fuentes y tamaños más comunes
        fuente_mas_comun, tamano_mas_comun = Counter(fuentes).most_common(1)[0][0]

        # Evaluar legibilidad
        legibilidad = "Buena" if tamano_mas_comun >= 10 else "Deficiente"

        return {
            "Fuente principal": fuente_mas_comun,
            "Tamaño de fuente más usado": tamano_mas_comun,
            "Legibilidad": legibilidad,
            "Uso de negritas": negritas,
            "Uso de cursivas": cursivas
        }


# ### Uso de colores y gráficos

# In[106]:


def analizar_colores_graficos(pdf_path):
    """
    Analiza un PDF en busca de:
    - Uso de colores en texto y gráficos (excluyendo azul de enlaces y fondo blanco).
    - Porcentaje de color real en la CV.
    - Cantidad de imágenes o gráficos.
    """
    cantidad_imagenes = 0
    total_paginas = 0
    color_detectado = 0
    color_total = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            total_paginas += 1
            cantidad_imagenes += len(page.images)  # Contar imágenes

            # Convertir página a imagen para análisis de color
            img = np.array(page.to_image(resolution=150).annotated)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Crear máscara para detectar colores (excluir blancos y grises)
            lower = np.array([0, 50, 50])  # No contar colores con baja saturación (grisáceos)
            upper = np.array([179, 255, 255])
            mask = cv2.inRange(img_hsv, lower, upper)
            porcentaje_color = (np.sum(mask > 0) / mask.size) * 100  # Porcentaje de color real

            # Si más del 90% de la página es color, asumimos que es un fondo y no cuenta
            if porcentaje_color < 90:
                color_detectado += porcentaje_color

            color_total += porcentaje_color

    # Promediar el porcentaje de color en todas las páginas
    porcentaje_color_promedio = round(color_detectado / total_paginas, 2) if total_paginas > 0 else 0

    return {
        "Uso de colores (excluyendo fondo blanco y links)": "Sí" if porcentaje_color_promedio > 0 else "No",
        "Porcentaje de color real en la CV": f"{porcentaje_color_promedio}%",
        "Cantidad de imágenes o gráficos": cantidad_imagenes
    }


# ### Distribución del espacio en la hoja - Densidad de la información vs espacios en blanco

# In[114]:


def calcular_densidad_informacion(pdf_path):
    """
    Calcula el porcentaje total de ocupación de texto en el documento completo.
    Retorna un único valor porcentual representando la densidad de información.
    """
    with pdfplumber.open(pdf_path) as pdf:
        porcentajes = []

        for page in pdf.pages:
            bbox = page.bbox  # Dimensiones de la página (ancho, alto)
            ancho, alto = bbox[2], bbox[3]
            area_total = ancho * alto

            palabras = page.extract_words()
            if palabras:
                # Calcular área ocupada por texto
                area_texto = sum((word['x1'] - word['x0']) * (word['bottom'] - word['top']) for word in palabras)
                porcentaje_texto = (area_texto / area_total) * 100
                porcentajes.append(porcentaje_texto)

        # Calcular densidad total como promedio de todas las páginas
        densidad_total = sum(porcentajes) / len(porcentajes) if porcentajes else 0

    return round(densidad_total, 2)


# ### Consistencia en el formato - Alineación, tamaños de títulos, márgenes

# In[118]:


def analizar_formato_cv(pdf_path):
    """
    Evalúa la consistencia del formato en términos de alineación, tamaños de títulos y márgenes.
    """
    with pdfplumber.open(pdf_path) as pdf:
        alineaciones = []
        tamaños_fuente = []
        margenes = []

        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["x0", "x1", "top", "fontname", "size"])
            if not words:
                continue

            # Detectar alineaciones (izquierda, derecha, centrado, justificado)
            for word in words:
                if word["x0"] < page.width * 0.15:
                    alineaciones.append("Izquierda")
                elif word["x1"] > page.width * 0.85:
                    alineaciones.append("Derecha")
                elif abs(word["x0"] - (page.width - word["x1"])) < 10:
                    alineaciones.append("Centrado")
                else:
                    alineaciones.append("Justificado")

                tamaños_fuente.append(round(word["size"], 1))

            # Calcular márgenes (mínima distancia del texto al borde)
            margen_izq = min(word["x0"] for word in words)
            margen_der = page.width - max(word["x1"] for word in words)
            margen_top = min(word["top"] for word in words)
            margen_bottom = page.height - max(word["top"] for word in words)
            margenes.append((margen_izq, margen_der, margen_top, margen_bottom))

        # Determinar consistencia
        alineacion_mas_comun = Counter(alineaciones).most_common(1)[0][0]
        tamaño_mas_comun = Counter(tamaños_fuente).most_common(1)[0][0]
        variabilidad_tamaños = len(set(tamaños_fuente)) > 3  # Si hay más de 3 tamaños diferentes, se considera inconsistente
        variabilidad_margenes = len(set(margenes)) > 2  # Si hay más de 2 configuraciones de márgenes, se considera inconsistente

        return {
            "Alineación principal": alineacion_mas_comun,
            "Tamaño de fuente más usado": tamaño_mas_comun,
            "Consistencia de tamaños": "Inconsistente" if variabilidad_tamaños else "Consistente",
            "Consistencia de márgenes": "Inconsistente" if variabilidad_margenes else "Consistente"
        }


# ## Claridad y Lenguaje

# ### Cantidad de palabras y densidad textual

# In[121]:


def calcular_palabras_y_densidad(pdf_path):
    """
    Calcula la cantidad de palabras y la densidad textual en un CV en PDF.
    """
    total_palabras = 0
    total_area_texto = 0
    total_area_pagina = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texto = page.extract_text()
            if texto:
                total_palabras += len(texto.split())  # Contar palabras
            total_area_pagina += page.width * page.height  # Área total de la página

            for block in page.extract_words(extra_attrs=["x0", "x1", "top", "bottom"]):
                ancho = block["x1"] - block["x0"]
                alto = block["bottom"] - block["top"]
                total_area_texto += ancho * alto  # Sumar áreas de bloques de texto

    # Calcular densidad textual como proporción del área de texto en la página
    densidad_textual = (total_area_texto / total_area_pagina) * 100 if total_area_pagina > 0 else 0

    return {
        "Cantidad de palabras": total_palabras,
    }


# ### Presencia de errores ortográficos o gramaticales

# In[136]:


# def detectar_errores_ortograficos_gramaticales(pdf_path, idioma="en"):
#     """
#     Detecta errores ortográficos y gramaticales en un CV en PDF.
#     """
#
#     # Inicializar correctores
#     spell = SpellChecker(language=idioma)
#     tool = language_tool_python.LanguageToolPublicAPI(idioma)
#
#     texto_completo = ""
#
#     # Extraer texto del PDF
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             texto = page.extract_text()
#             if texto:
#                 texto = texto.replace("\n", " ")  # Reemplazar saltos de línea por espacios
#                 texto = re.sub(r"\s+", " ", texto)  # Normalizar espacios múltiples a uno solo
#                 texto = re.sub(r"([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])([.,])([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])", r"\1\2 \3", texto)
#                 texto_completo += texto + " "
#
#     # 🔹 Normalizar texto (eliminar puntuación y convertir a minúsculas)
#     texto_completo = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ ]", "", texto_completo.lower())
#
#     # 🔹 Separar palabras y eliminar las demasiado cortas
#     palabras = [palabra for palabra in texto_completo.split() if len(palabra) > 2]
#
#     # 🔹 Revisar errores ortográficos (palabras desconocidas)
#     errores_ortograficos = {palabra: spell.correction(palabra) for palabra in spell.unknown(palabras)}
#
#     # 🔹 Dividir el texto en fragmentos más pequeños para LanguageTool
#     oraciones = texto_completo.split(". ")  # Dividir en oraciones
#     errores_gramaticales = sum(len(tool.check(ora)) for ora in oraciones)  # Contar errores totales
#
#     return {
#         "Errores ortográficos detectados": len(errores_ortograficos),
#         "Ejemplo de errores ortográficos": dict(list(errores_ortograficos.items())[:10]),  # Solo muestra 10
#         "Cantidad de errores gramaticales": errores_gramaticales,
#     }
#
# # 📌 Ejemplo de uso
# pdf_file = "hojas_de_vida_copy/Paso/CVJCOC.pdf"
# resultado_errores = detectar_errores_ortograficos_gramaticales(pdf_file)
# print(f"Errores detectados: {resultado_errores}")


"""Por ahora no se avanzará con este ya que es muy exigente con el sistema y muy poco precisa"""


# ### Creamos un diccionario con terminos técnicos y palabras clave
# Esto para identificar la cantidad de términos y keywords que un candidato puede usar en su CV

# In[148]:


TECH_TERMS = {
        # Lenguajes de programación
        "Java", "Spring Boot", "Hibernate", "JPA", "Java EE",
        "JavaScript", "TypeScript", "React", "Angular", "Vue",
        "Node.js", "Express.js", "NestJS", "HTML", "CSS", "SASS",
        "Python", "Django", "Flask", "FastAPI", "C#", ".NET", "ASP.NET",
        "Ruby", "Rails", "PHP", "Laravel", "Go", "Rust", "Kotlin", "Swift",

        # Bases de datos
        "SQL", "PostgreSQL", "MySQL", "MariaDB", "MongoDB", "NoSQL",
        "Redis", "Elasticsearch", "Firebase", "DynamoDB", "GraphQL",

        # DevOps y Cloud
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform",
        "Jenkins", "CI/CD", "GitHub Actions", "Ansible", "Linux",

        # Arquitectura y Metodologías
        "Microservices", "API", "REST", "SOAP", "GraphQL",
        "Agile", "Scrum", "Kanban", "DDD", "TDD", "SOLID", "Clean Architecture",

        # Seguridad y Testing
        "OWASP", "JWT", "OAuth", "SAML", "Penetration Testing",
        "Selenium", "JUnit", "Mockito", "Jest", "Cypress",

        # Otros conceptos importantes
        "Multithreading", "Concurrency", "Asynchronous Programming",
        "WebSockets", "Event-Driven Architecture", "Kafka", "RabbitMQ",
        "gRPC", "WebAssembly"
}


# 

# ### Uso de lenguaje técnico o genérico

# In[149]:


def analyze_technical_language(pdf_path):
    """Extrae texto de un PDF y calcula el porcentaje de lenguaje técnico vs genérico."""

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"Error al leer el PDF: {e}"

    words = re.findall(r'\b\w+\b', text)
    if not words:
        return "No se pudo extraer texto del PDF"

    total_words = len(words)
    tech_words = sum(1 for word in words if word in TECH_TERMS)

    tech_ratio = (tech_words / total_words) * 100 if total_words > 0 else 0

    return f"Lenguaje Técnico: {tech_ratio:.2f}% | Lenguaje Genérico: {100 - tech_ratio:.2f}%"


# ## Personalización y adaptabilidad

# ### Inclusión de enlaces a portafolio o redes sociales

# In[163]:


def extract_links_from_cv(pdf_path):
    patterns = {
        "LinkedIn": r"https?:\/\/(?:www\.)?linkedin\.com\/[a-zA-Z0-9\-_/]+",
        "GitHub": r"https?:\/\/(?:www\.)?github\.com\/[a-zA-Z0-9\-_/]+",
        "Website": r"https?:\/\/(?:www\.)?[a-zA-Z0-9\-.]+\.[a-zA-Z]{2,6}(?:\/[a-zA-Z0-9\-_/]*)?",
    }

    # Extraer texto del PDF
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text", flags=1).replace("\n", " ") for page in doc])

    # Buscar LinkedIn y GitHub
    linkedin_found = bool(re.search(patterns["LinkedIn"], text))
    github_found = bool(re.search(patterns["GitHub"], text))

    # Buscar cualquier otro enlace
    all_extracted_links = set(re.findall(patterns["Website"], text))

    # Filtrar para que no incluya LinkedIn ni GitHub
    personal_website_found = any(link for link in all_extracted_links if not (
        re.match(patterns["LinkedIn"], link) or re.match(patterns["GitHub"], link)
    ))

    return {
        "LinkedIn": linkedin_found,
        "GitHub": github_found,
        "Personal Website": personal_website_found
    }


# ## Información visual y Complementaria

# ### Incorporación de foto o ausencia de la misma

# In[167]:


def tiene_foto(pdf_path):
    doc = fitz.open(pdf_path)

    for page in doc:
        if page.get_images(full=True):
            return True  # Si encuentra al menos una imagen, retorna True

    return False


# ### Presencia de íconos o elementos gráficos

# In[192]:


def detectar_elementos_graficos(pdf_path, umbral_area=2000, max_elementos=3):
    """
    Detecta si un CV en PDF tiene elementos gráficos significativos (líneas gruesas, rectángulos, dibujos vectoriales).

    Parámetros:
        - pdf_path: Ruta del archivo PDF.
        - umbral_area: Área mínima para considerar un elemento gráfico como relevante.
        - max_elementos: Número mínimo de elementos grandes para considerar que hay gráficos.

    Retorna:
        - Un diccionario con 'Tiene_elementos_graficos' como True o False.
    """
    elementos_grandes = 0
    doc = fitz.open(pdf_path)

    for page in doc:
        drawings = page.get_drawings()
        for draw in drawings:
            bbox = draw["rect"]  # Bounding box (x0, y0, x1, y1)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height  # Calculamos el área del elemento gráfico

            if area > umbral_area:  # Consideramos solo elementos suficientemente grandes
                elementos_grandes += 1
                if elementos_grandes >= max_elementos:
                    return {"Tiene_elementos_graficos": True}

    return {"Tiene_elementos_graficos": False}


# ## Extracción de la información

# In[ ]:




