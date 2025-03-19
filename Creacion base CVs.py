#!/usr/bin/env python
# coding: utf-8

# # Creación de base de datos para identificación de HVs
# 
# El presente archivo se creará la base de datos extrayendo información de HVs usando procesamiento de lenguaje natural. Esto con el objetivo de más adelante tener la posibilidad de generar con base en 300 HVs tanto rechazadas como avanzadas, un modelo que nos permita identificar las características predominantes de una HV que hace que avance o no durante el proceso.
# 
# 

# #### Importar librerias

# In[229]:


import os
import fitz  # PyMuPDF for PDFs
import pytesseract
from PIL import Image
import pandas as pd
import re
import spacy
import io
import numpy as np


# ### Importar SpaCy para NLP
# Spacy es una librería de python que permite por medio de modelos de lenguaje pre-importados realizar análisis de texto, identificando palabras, nombres, lugares, objetos, verbos, adjetivos y la relación entre los mismos.
# 
# En este caso, importamos el modelo pre-entrenado en inglés, lo que requiere que todas las CVs a procesar estén en este idioma.
# 

# In[230]:


nlp = spacy.load("en_core_web_sm")


# #### Cargar las carpetas con las HVs

# In[231]:


hv_dir_exitosas_java = "hojas_de_vida/java/Paso"
hv_dir_noexitosas_java = "hojas_de_vida/java/No Paso"
hv_dir_exitosas_front = "hojas_de_vida/frontend/Paso"
hv_dir_noexitosas_front = "hojas_de_vida/frontend/No Paso"


# ### Deifinir palabras clave
# En este caso, se definirar palabras clave que podrán tener las HVs teniendo en cuenta que para este modelo en particular se está utlizando solo HVs para un requerimiento de **desarrolladores Java**.
# 

# In[232]:


palabras_clave_java = ["Java", "Spring", "spring boot", "AWS", "Azure", "GCP", "Google Cloud Platform", "microservices", "Maven", "Gradle", "Java Server Pages", "JSP", "JEE", "Java Enterprise Edition", "Java8", "Java11", "Java17", "Java21", "JVM", "Java virtual machine"]

palabras_clave_front_end = ["Javascript", "Typescript", "React", "Angular", "Vue", "react.js", "vue.js", "HTML", "CSS", "Redux", "Hooks", "Micro frontends"]


# ### Definir las secciones y los patrones en las que estas van a aparecer
# Además de definir la cantidad de palabras clave, es importante contar con las secciones con las que cada documento puede contar y entender si cuenta o no con este.

# In[233]:


secciones = {
    "education": r"education|academic background|studies|study|university studies|professional education",
    "work_experience": r"experience|work|work experience|employment history|professional experience|background|professional background",
    "skills": r"skills|technical skills|competencies",
    "certifications": r"certifications|licenses|accreditations",
    "achievements": r"achievements|achieved",
    "professional_profile": r"profile|summary|about me|professional summary|objective|summary",
    "languages": r"languages|linguistic skills|spoken languages",
    "projects": r"projects|case studies|portfolio",
    "publications": r"publications|research papers|articles|books",
    "training_courses": r"training|courses|workshops|online learning",
    "volunteer_work": r"volunteer|volunteering|social impact|community service",
}


# ### Detectar el tipo de HV para posterior procesamiento de palabras clave
# 

# In[234]:


def detect_cv_type(cv_path):
    if "java" in cv_path.lower():
        return "java"
    elif "frontend" in cv_path.lower():
        return "frontend"
    else:
        return "unknown"  # Default case if it's unclear


# ### Extraer el texto de los PDFs
# 
# A continuación se usará la librería FITZ, la cual ayuda a extraer el texto de un PDF, ver si tiene imágenes, contar sus páginas y detectar colores en los mismos.
# 
# La declaramos como función para llamarla más adelante en el procesamiento de todas las características que buscamos extraer.

# In[235]:


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF, handling both text-based and image-based (scanned) content."""
    text = []
    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            # Extract text from the page (use "blocks" for better structure)
            page_text = "\n".join(block[4] for block in page.get_text("blocks"))
            text.append(page_text)

            # Handle OCR for image-based text (scanned PDFs)
            for img_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                img_bytes = base_image["image"]

                # Convert image bytes to a PIL image
                img_pil = Image.open(io.BytesIO(img_bytes))

                # ✅ Fix: Convert RGBA or P-mode images (with transparency) to RGB
                if img_pil.mode in ("RGBA", "P"):
                    img_pil = img_pil.convert("RGB")

                # Convert image to text using OCR
                ocr_text = pytesseract.image_to_string(img_pil)
                text.append(ocr_text)

    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")

    return "\n".join(text).strip()


# ### Contar palabras en general

# In[236]:


def contar_palabras(text):
    return len(text.split()) if text else 0


# #### Contar palabras clave

# In[237]:


def contar_palabras_clave(text, cv_type):
    """
    Counts occurrences of keywords in the CV text using regex for better accuracy.
    - Handles variations like hyphens, spaces, and case differences.
    - Uses word boundaries `\b` but allows for slight variations in spacing.
    """
    text_lower = text.lower()

    # Choose the correct keyword list based on CV type
    keyword_list = palabras_clave_java if cv_type == "java" else palabras_clave_front_end

    count = 0
    for keyword in keyword_list:
        keyword_pattern = re.escape(keyword).replace(" ", r"\s*[-_]?\s*")  # ✅ Fix: Create regex separately
        keyword_regex = rf"\b{keyword_pattern}\b"  # ✅ Now safely formatted

        matches = re.findall(keyword_regex, text_lower, re.IGNORECASE)
        count += len(matches)  # Sum occurrences

    return count


# #### Extraer las secciones

# Para extraer las secciones, usamos expresiones regulares. Con la biblioteca Re, busca el patron definido en la variable secciones más arriba, que ayuda a identificar si el texto obtenido del PDF tiene o no esta sección.

# In[238]:


def extraer_secciones(text):
    """
    Extracts sections from the CV text using regex patterns.
    - Matches variations of section headers (Education, Work Experience, etc.).
    - Ignores dashes, colons, and extra spaces for better detection.
    - Counts words correctly by detecting next section OR using end of text.
    """
    sections = {key: {"exists": False, "word_count": 0} for key in secciones.keys()}

    for section, pattern in secciones.items():
        # ✅ Improved regex pattern: handles variations, dashes, colons, spacing
        section_regex = rf"\b{pattern}\s*[:\-]?\s*\b"

        matches = list(re.finditer(section_regex, text, re.IGNORECASE))
        if matches:
            sections[section]["exists"] = True
            start_pos = matches[0].start()

            # Find the next section (or use the end of text)
            next_starts = [
                m.start() for s, p in secciones.items()
                if (m := re.search(p, text[start_pos + 1:], re.IGNORECASE))
            ]
            end_pos = min(next_starts, default=len(text))  # ✅ Use end of text as fallback

            # ✅ Ensure section extraction doesn't fail
            section_text = text[start_pos:end_pos].strip()
            sections[section]["word_count"] = contar_palabras(section_text) if section_text else 0

    return sections


# ### Verificar factores como foto y colores
# De vuelta se usa la librería fitz para poder leer el PDF

# #### Verificar si tiene o no foto

# In[239]:


def tiene_foto_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if len(page.get_images(full=True)) > 0:
                return True
    except Exception as e:
        print(f"Error revisando foto en PDF {pdf_path}: {e}")
    return False


# #### Verificar si tiene colores adicionales el PDF

# In[240]:


def tiene_color_pdf(pdf_path):
    """Detects if a PDF contains non-black/gray colors in drawings or images."""
    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            # Check vector elements (lines, shapes)
            for draw in page.get_drawings():
                if "color" in draw and draw["color"] is not None:  # ✅ Check if "color" exists
                    r, g, b = draw["color"]
                    if (r, g, b) != (0, 0, 0) and r != g != b:  # Exclude black & grayscale
                        return True

            # Check images in the PDF
            for img_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                img_bytes = base_image["image"]

                # Convert image bytes to a PIL image
                img_pil = Image.open(io.BytesIO(img_bytes))

                # ✅ Fix: Convert RGBA images (with transparency) to RGB
                if img_pil.mode == "RGBA":
                    img_pil = img_pil.convert("RGB")

                # Convert image to a NumPy array
                img_np = np.array(img_pil)

                # Check if the image contains any non-grayscale pixels
                if len(img_np.shape) == 3:  # Ensure it's a color image
                    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                    if np.any((r != g) | (g != b)):  # If any pixel has unequal R, G, B, it's colored
                        return True

    except Exception as e:
        print(f"❌ Error detecting color in PDF {pdf_path}: {e}")

    return False  # If no color found, return False


# #### Contar páginas

# In[241]:


def contar_paginas(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return len(doc)
    except Exception as e:
        print(f"Error counting pages in PDF {pdf_path}: {e}")
        return 1


# ### Procesamiento del CV
# A continuación la función de procesamiento, nos ayudará a procesar un solo CV de acuerdo a los parámetros establecidos anteriormente, ejecutando cada una de las funciones ya establecidas

# In[248]:


def process_cv(cv_path):
    tipo_cv = detect_cv_type(cv_path)

    text = extract_text_from_pdf(cv_path)
    if not text:
        print(f"⚠️ No text extracted from {cv_path}")

    has_photo = tiene_foto_pdf(cv_path)
    has_colors = tiene_color_pdf(cv_path)
    num_pages = contar_paginas(cv_path)

    total_word_count = contar_palabras(text)
    keyword_count = contar_palabras_clave(text, tipo_cv)
    sections = extraer_secciones(text)

    return {
        "CV_Name": os.path.basename(cv_path),
        "Total_Word_Count": total_word_count,
        "Has_Photo": int(has_photo),
        "Has_Colors": int(has_colors),
        "Pages": num_pages,
        "Keyword_Count": keyword_count,
        "Education_Exists": int(sections["education"]["exists"]),
        "Education_Word_Count": sections["education"]["word_count"],
        "Work_Experience_Exists": int(sections["work_experience"]["exists"]),
        "Work_Experience_Word_Count": sections["work_experience"]["word_count"],
        "Skills_Exists": int(sections["skills"]["exists"]),
        "Skills_Word_Count": sections["skills"]["word_count"],
        "Certifications_Exists": int(sections["certifications"]["exists"]),
        "Certifications_Word_Count": sections["certifications"]["word_count"],
        "Achievements_Exists": int(sections["achievements"]["exists"]),
        "Achievements_Word_Count": sections["achievements"]["word_count"],
        "Professional_Profile_Exists": int(sections["professional_profile"]["exists"]),
        "Professional_Profile_Word_Count": sections["professional_profile"]["word_count"],
        "Projects_Exists": int(sections["projects"]["exists"]),
        "projects_Word_Count": sections["projects"]["word_count"],
        "volunteer_work_Exists": int(sections["volunteer_work"]["exists"]),
        "volunteer_work_Word_Count": sections["volunteer_work"]["word_count"]
    }


# ### Procesamiento de CVs en la carpeta
# La siguiente función nos ayuda a de acuerdo con lo establecido anteriormente, procesar todas las CVs en las carpetas seleccionadas y devolverlas en una lista

# In[249]:


def process_folder(folder_path, label):
    cv_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            cv_path = os.path.join(folder_path, filename)
            print(f"Processing: {cv_path}")
            cv_info = process_cv(cv_path)
            cv_info["Passed"] = label
            cv_data.append(cv_info)
    return cv_data


# ## Creación de la base de datos

# Se crean las variables donde se almacenan las CVs exitosas procesadas, agregando la información de 1 si es exitosa y 0 si no es exitosa.

# In[250]:


# 📌 Process All CVs & Create Dataset
data_total = (process_folder(hv_dir_exitosas_java, 1) +
              process_folder(hv_dir_noexitosas_java, 0) +
              process_folder(hv_dir_exitosas_front, 1) +
              process_folder(hv_dir_noexitosas_front, 0))


# Se guarda esta información en un dataframe

# In[245]:


baseCVs = pd.DataFrame(data_total)

#borramos el CV name ya que no es necesaria y buscamos información anónima
baseCVs = baseCVs.drop('CV_Name', axis=1)

#Aleatorizamos el orden para que no queden juntos los 1 y los 0 todos juntos y las muestras sean más representativas
baseCVs = baseCVs.sample(frac=1, random_state=42).reset_index(drop=True)


# In[246]:


baseCVs


# ### Exportar base en un archivo CSV para posterior lectura

# In[247]:


baseCVs.to_csv("baseCVs.csv", index=False)

