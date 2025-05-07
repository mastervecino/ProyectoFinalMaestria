#!/usr/bin/env python
# coding: utf-8

# # Script para creación de CV

# ## Importaciones y Configuración Inicial

# In[1]:


# # Script para extracción de información de las CV (Versión Refactorizada)

# ##### Obtener librerías necesarias
import os
import fitz  # PyMuPDF (principal para interactuar con PDF)
import pandas as pd
import re
import spacy
from rapidfuzz import process, fuzz # Para comparación de strings difusa (secciones)
from dateutil import parser        # Para interpretar fechas
from collections import Counter    # Para contar elementos (fuentes, palabras, etc.)
import numpy as np                 # Requerido por OpenCV si se usa
import cv2                         # Para análisis de imagen (colores) - Podríamos intentar minimizar su uso
import logging                     # Para registrar progreso y errores
# import language_tool_python      # Comentado, como en el original
# from spellchecker import SpellChecker # Comentado, como en el original

# Configuración del Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes y Configuraciones (ej. umbrales, listas de palabras)
# Es buena práctica definirlas aquí
UMBRAL_SIMILITUD_SECCION = 75
MIN_AREA_ELEMENTO_GRAFICO = 2000 # Pixeles cuadrados (aprox)
MIN_ELEMENTOS_GRAFICOS_GRANDES = 3
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
    "training_courses": ["training", "courses", "workshops", "seminars", "courses and seminars", "Other Studies", "cursos", "formación complementaria"],
    "volunteer_work": ["volunteer work", "volunteering", "community service", "social impact", "non-profit", "voluntariado"],
}
# Expresiones regulares para fechas (combinadas y mejoradas)
# Prioriza formatos más específicos primero
# Cerca del inicio del script, reemplazar la lista PATRONES_FECHAS_REGEX por esto:

PATRONES_FECHAS_FORMATOS = [
    # Formatos con Mes (texto) y Año: Jan 2020, January 2020, Enero 2020, etc.
    # Separar nombre completo e abreviado puede dar más detalle
    {'name': 'Mon YYYY (EN)', 'pattern': r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}\b"},
    {'name': 'Month YYYY (EN)', 'pattern': r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"},
    {'name': 'Mes YYYY (ES Abr)', 'pattern': r"\b(?:Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)\.?\s+\d{4}\b"},
    {'name': 'Mes YYYY (ES Comp)', 'pattern': r"\b(?:Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)\s+\d{4}\b"},

    # Formatos numéricos: MM/YYYY, MM-YYYY
    {'name': 'MM/YYYY', 'pattern': r"\b(0?[1-9]|1[0-2])/\d{4}\b"}, # Asegurar mes válido
    {'name': 'MM-YYYY', 'pattern': r"\b(0?[1-9]|1[0-2])-\d{4}\b"}, # Asegurar mes válido

    # Formatos con día: DD/MM/YYYY, DD-MM-YYYY (o MM/DD/YYYY - ambiguo sin contexto)
    # Ser más específico si es posible, o usar un nombre genérico
    {'name': 'DD/MM/YYYY', 'pattern': r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/\d{4}\b"},
    {'name': 'DD-MM-YYYY', 'pattern': r"\b(0?[1-9]|[12]\d|3[01])-(0?[1-9]|1[0-2])-\d{4}\b"},
    # Podríamos añadir variaciones con año de 2 dígitos si son comunes: \d{1,2}[/-]\d{1,2}[/-]\d{2}\b

    # Rangos de años: 2018-2020, 2018 – 2020
    {'name': 'Rango YYYY-YYYY', 'pattern': r"\b(19[89]\d|20[0-3]\d)\s*[-–]\s*(19[89]\d|20[0-3]\d)\b"},

    # Año hasta presente: 2019 - Present, 2020 - Actualidad
    {'name': 'Rango YYYY-Presente', 'pattern': r"\b(19[89]\d|20[0-3]\d)\s*[-–]\s*(?:Present|Actual|Actualidad|Today|Now)\b"},

    # Años sueltos (menos prioritario, poner al final)
    # Usar (?<!\d) y (?!\d) para asegurar que no es parte de un número más grande
    {'name': 'YYYY', 'pattern': r"(?<!\d|\.|/|-)(19[89]\d|20[0-3]\d)(?!:|\d|\.|/|-)\b"} # Año aislado 1980-2039
]


# ## Carga del Modelo SpaCy

# In[2]:


# ##### Importar SpaCy para NLP
nlp = spacy.load("en_core_web_sm")
logging.info("Modelo SpaCy 'en_core_web_sm' cargado exitosamente.")


# ## Definición de Rutas de Entrada

# In[3]:


# #### Definir directorios de CV a estudiar
# NOTA: Es recomendable pasar estas rutas como argumentos usando argparse
# en un script .py final para mayor flexibilidad.
# Ejemplo: python tu_script.py --exitosas ../ruta/paso --no-exitosas ../ruta/no_paso

# Directorios de entrada (cambiar según sea necesario)
# Asegúrate de que las rutas sean correctas desde donde ejecutas el script
base_dir = "../hojas_de_vida_copy/" # Ajusta si es necesario
hv_dir_exitosas = os.path.join(base_dir, "Paso")
hv_dir_noexitosas = os.path.join(base_dir, "No paso")

# Crear lista de todos los archivos PDF a procesar
pdf_files_to_process = []
if os.path.isdir(hv_dir_exitosas):
    pdf_files_to_process.extend([(os.path.join(hv_dir_exitosas, f), "Exitoso")
                                 for f in os.listdir(hv_dir_exitosas) if f.lower().endswith('.pdf')])
    logging.info(f"Encontrados {len(pdf_files_to_process)} CV en '{hv_dir_exitosas}'.")
else:
    logging.warning(f"Directorio de CV exitosos no encontrado: '{hv_dir_exitosas}'")

initial_count = len(pdf_files_to_process)
if os.path.isdir(hv_dir_noexitosas):
    pdf_files_to_process.extend([(os.path.join(hv_dir_noexitosas, f), "No Exitoso")
                                  for f in os.listdir(hv_dir_noexitosas) if f.lower().endswith('.pdf')])
    logging.info(f"Encontrados {len(pdf_files_to_process) - initial_count} CV en '{hv_dir_noexitosas}'.")
else:
    logging.warning(f"Directorio de CV no exitosos no encontrado: '{hv_dir_noexitosas}'")

logging.info(f"Total de archivos PDF a procesar: {len(pdf_files_to_process)}")


# ### Contar páginas

# In[4]:


# ## Estructura y Organización

def contar_paginas_fitz(doc: fitz.Document) -> int:
    """Cuenta el número de páginas en un documento fitz."""
    if doc:
        return doc.page_count
    return 0


# ### Detectar secciones

# In[5]:


def detectar_secciones_texto(texto: str, secciones_dict: dict, umbral: int) -> dict:
    """
    Detecta la presencia de secciones clave en el texto de un CV usando fuzzy matching.

    Args:
        texto: El texto completo del CV como un string.
        secciones_dict: Diccionario con nombres de sección y sus sinónimos.
        umbral: Puntuación mínima de similitud (0-100) para considerar una coincidencia.

    Returns:
        Un diccionario indicando qué secciones fueron detectadas (True/False).
    """
    secciones_detectadas = {seccion: False for seccion in secciones_dict}
    if not texto:
        return secciones_detectadas

    # Considerar las primeras N líneas o una porción inicial para eficiencia si el CV es muy largo?
    # Por ahora, analiza todas las líneas.
    lineas = texto.split("\n")

    # Usar un conjunto para evitar procesar la misma línea limpia múltiples veces si es idéntica
    lineas_unicas_procesadas = set()

    for linea in lineas:
        linea_limpia = re.sub(r"\s+", " ", linea.strip()).lower() # Limpia y normaliza a minúsculas

        # Ignorar líneas muy cortas o ya procesadas
        if len(linea_limpia) < 3 or linea_limpia in lineas_unicas_procesadas:
            continue
        lineas_unicas_procesadas.add(linea_limpia)

        # Comprobar si la línea parece un encabezado (pocas palabras, quizás mayúsculas)
        # Esto es una heurística opcional para enfocar la búsqueda
        # if len(linea_limpia.split()) < 6: # Considerar solo líneas cortas como posibles encabezados

        for seccion, sinonimos in secciones_dict.items():
            # Usar extractOne con partial_ratio puede ser bueno para encontrar frases dentro de líneas más largas
            resultado = process.extractOne(linea_limpia, sinonimos, scorer=fuzz.partial_ratio, score_cutoff=umbral)

            if resultado:
                # Si encuentra una coincidencia por encima del umbral para *cualquier* sinónimo de esta sección
                secciones_detectadas[seccion] = True
                # Optimización: Si ya encontramos esta sección, podemos pasar a la siguiente sección del diccionario.
                # Sin embargo, una línea podría coincidir con sinónimos de diferentes secciones (menos probable).
                # Por seguridad, seguimos buscando en la misma línea por si acaso, pero podríamos romper el bucle interno:
                # break # Descomentar si queremos asignar solo la primera sección que coincida en una línea

    # Convertir booleanos a "Sí"/"No" para el resultado final si se prefiere
    # return {k: "Sí" if v else "No" for k, v in secciones_detectadas.items()}
    return secciones_detectadas # Devolver booleanos es más útil para procesamiento posterior


# ### Longitud de cada sección

# In[6]:


# NUEVA FUNCIÓN: Calcular líneas aproximadas por sección

def calcular_lineas_por_seccion(texto: str, secciones_dict: dict, umbral: int) -> dict:
    """
    Estima el número de líneas de texto dentro de cada sección detectada.

    Args:
        texto: El texto completo del CV.
        secciones_dict: Diccionario con nombres de sección y sus sinónimos.
        umbral: Puntuación mínima de similitud para detectar un encabezado.

    Returns:
        Un diccionario con el número de líneas estimado por sección detectada.
        Ej: {"Lineas_work_experience": 25, "Lineas_education": 15}
    """
    lineas_por_seccion = {f"Lineas_{k}": 0 for k in secciones_dict.keys()}
    if not texto:
        return lineas_por_seccion

    lineas = texto.split("\n")
    n_lineas_total = len(lineas)
    secciones_encontradas = [] # Lista para guardar (indice_linea, nombre_seccion)

    # 1. Detectar encabezados y sus índices de línea
    lineas_procesadas_header = set() # Para evitar doble conteo si línea se repite
    for i, linea in enumerate(lineas):
        linea_limpia = re.sub(r"\s+", " ", linea.strip()).lower()

        if len(linea_limpia) < 3 or linea_limpia in lineas_procesadas_header:
            continue

        # Considerar solo líneas cortas como posibles encabezados (heurística)
        # Ajustar este umbral si es necesario
        if len(linea_limpia.split()) < 7:
            for seccion, sinonimos in secciones_dict.items():
                resultado = process.extractOne(linea_limpia, sinonimos, scorer=fuzz.partial_ratio, score_cutoff=umbral)
                if resultado:
                    # Se encontró un encabezado
                    secciones_encontradas.append((i, seccion))
                    lineas_procesadas_header.add(linea_limpia)
                    # Romper para no asignar la misma línea a múltiples secciones si coincide con varias
                    break

    if not secciones_encontradas:
        logging.debug("No se detectaron encabezados de sección para calcular líneas.")
        return lineas_por_seccion # Devolver ceros

    # 2. Ordenar secciones por índice de línea
    secciones_encontradas.sort(key=lambda x: x[0])

    # 3. Calcular líneas entre encabezados
    for j in range(len(secciones_encontradas)):
        idx_inicio_seccion, nombre_seccion = secciones_encontradas[j]

        # Determinar el índice final de la sección actual
        if j + 1 < len(secciones_encontradas):
            # El final es el inicio de la siguiente sección
            idx_fin_seccion = secciones_encontradas[j+1][0]
        else:
            # Es la última sección encontrada, va hasta el final del documento
            idx_fin_seccion = n_lineas_total

        # Contar líneas no vacías entre el encabezado (excluyéndolo) y el final
        # idx_inicio_seccion + 1 para empezar después del encabezado
        lineas_contenido = [lineas[k].strip() for k in range(idx_inicio_seccion + 1, idx_fin_seccion) if lineas[k].strip()]
        conteo_lineas = len(lineas_contenido)

        lineas_por_seccion[f"Lineas_{nombre_seccion}"] = conteo_lineas

    logging.debug(f"Líneas por sección calculadas: {lineas_por_seccion}")
    return lineas_por_seccion


# ### Fechas

# In[7]:


# REEMPLAZAR la función extraer_fechas_texto con esta versión:

def extraer_fechas_texto(texto: str, patrones_fechas_defs: list, nlp_model) -> tuple[list, list]:
    """
    Extrae fechas de un texto usando regex y opcionalmente SpaCy.
    *** Versión que retorna fechas parseadas Y lista de formatos detectados ***

    Args:
        texto: El texto completo del CV.
        patrones_fechas_defs: Lista de diccionarios {'name': ..., 'pattern': ...}.
        nlp_model: El modelo SpaCy cargado (puede ser None).

    Returns:
        Una tupla: (lista_fechas_parseadas, lista_formatos_detectados)
        - lista_fechas_parseadas: Objetos datetime o "Presente", ordenada de reciente a antigua.
        - lista_formatos_detectados: Lista con los nombres de formato de cada fecha encontrada.
    """
    fechas_encontradas_str = set()
    formatos_detectados_dict = {} # Usar dict para evitar duplicados por la misma string exacta
                                 # formato: {fecha_string_original: formato_nombre}
    if not texto:
        return [], []

    # 1. Usar expresiones regulares y registrar formato
    # Iterar en el orden definido (más específicos primero)
    texto_procesado_regex = texto # Para marcar texto ya cubierto por regex más específico (opcional)
    for fmt_def in patrones_fechas_defs:
        patron_regex = fmt_def['pattern']
        formato_nombre = fmt_def['name']
        try:
            # Usar finditer para obtener la posición y el texto exacto
            matches_found = [] # Guardar matches para procesar después y evitar solapamientos
            for match in re.finditer(patron_regex, texto_procesado_regex, re.IGNORECASE):
                fecha_str = match.group(0).strip()
                # Evitar registrar si ya se encontró esta string con un formato (prioriza el primero)
                if fecha_str not in formatos_detectados_dict:
                     formatos_detectados_dict[fecha_str] = formato_nombre
                     fechas_encontradas_str.add(fecha_str)
                     # Opcional: Marcar esta parte del texto como procesada para evitar solapamientos
                     # start, end = match.span()
                     # texto_procesado_regex = texto_procesado_regex[:start] + (' ' * (end-start)) + texto_procesado_regex[end:]

        except re.error as e:
            logging.warning(f"Error en regex para formato '{formato_nombre}': {e}")
            continue

    # 2. Usar spaCy si está disponible y clasificar formato si es posible
    if nlp_model:
        try:
            max_len = 100000
            # Limitar longitud del texto a SpaCy si es muy grande para evitar consumo extremo
            texto_para_spacy = texto[:max_len*5] # Procesar hasta 500k caracteres con SpaCy
            if len(texto) > len(texto_para_spacy):
                 logging.warning(f"Texto truncado para análisis de fechas con SpaCy (longitud original: {len(texto)})")

            doc_fragments = [nlp_model(texto_para_spacy[i:i+max_len]) for i in range(0, len(texto_para_spacy), max_len)]
            for doc in doc_fragments:
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        fecha_str_spacy = ent.text.strip()
                        # Solo añadir si no fue detectada por regex (regex suele ser más preciso para formato)
                        if fecha_str_spacy not in formatos_detectados_dict:
                            fechas_encontradas_str.add(fecha_str_spacy)
                            # Intentar clasificar formato de SpaCy usando nuestros patrones
                            formato_spacy = "SpaCy DATE (Sin clasificar)" # Default
                            for fmt_def in patrones_fechas_defs:
                                # Usar fullmatch para ver si la entidad completa coincide con un formato
                                try:
                                     if re.fullmatch(fmt_def['pattern'], fecha_str_spacy, re.IGNORECASE):
                                          formato_spacy = fmt_def['name'] + " (SpaCy)"
                                          break # Quedarse con la primera coincidencia
                                except re.error:
                                     continue # Ignorar patrones regex inválidos aquí también
                            formatos_detectados_dict[fecha_str_spacy] = formato_spacy

        except Exception as e:
            logging.warning(f"Error durante el procesamiento de fechas con SpaCy: {e}")


    # 3. Convertir a objetos datetime y manejar "Presente" (usando fechas_encontradas_str)
    # (Reutilizar la lógica de parseo de la versión anterior con filtro,
    # pero adaptada para trabajar sobre fechas_encontradas_str directamente)
    fechas_convertidas = []
    current_year = pd.Timestamp.now().year

    # Usar la lista de strings únicos encontrados
    for fecha_str in fechas_encontradas_str:
        fecha_str_lower = fecha_str.lower()
        try:
             # Intentar parseo directo primero (mejorado para manejar algunos casos comunes)
             # Manejo especial para 'Presente' y similares
            if "present" in fecha_str_lower or "actual" in fecha_str_lower or "today" in fecha_str_lower or "now" in fecha_str_lower:
                 fechas_convertidas.append("Presente")
                 continue # Pasar a la siguiente cadena

            parsed_date = parser.parse(fecha_str, fuzzy=False)
            fechas_convertidas.append(parsed_date)

        except(ValueError, OverflowError, TypeError, parser.ParserError) as e:
             # Si falla, intentar regex de rangos/año (evitar parsear duraciones aquí)
            try:
                 if re.fullmatch(r"\b(19[89]\d|20[0-3]\d)\s*[-–]\s*(19[89]\d|20[0-3]\d)\b", fecha_str, re.IGNORECASE):
                      year_end = re.findall(r'\d{4}', fecha_str)[-1]
                      fechas_convertidas.append(pd.Timestamp(f'{year_end}-12-31'))
                 elif re.fullmatch(r"\b(19[89]\d|20[0-3]\d)\s*[-–]\s*(?:Present|Actual|Actualidad|Today|Now)\b", fecha_str, re.IGNORECASE):
                      fechas_convertidas.append("Presente")
                 else:
                      match_year = re.fullmatch(r"\b(19[89]\d|20[0-3]\d)\b", fecha_str) # Solo año
                      if match_year:
                           fechas_convertidas.append(pd.Timestamp(f'{match_year.group(0)}-12-31'))
                      else:
                           # Ignorar si no se pudo parsear ni coincide con rangos/año simple
                           logging.debug(f"No se pudo parsear ni clasificar como fecha válida: '{fecha_str}' - Error: {e}")
            except (ValueError, OverflowError, TypeError, re.error):
                 logging.debug(f"Error secundario al intentar extraer año/rango de '{fecha_str}'")

    # 4. Ordenar fechas parseadas
    def sort_key(fecha):
        if fecha == "Presente": return (0, pd.Timestamp.max)
        elif isinstance(fecha, pd.Timestamp): return (1, fecha)
        else: return (2, pd.Timestamp.min)

    fechas_ordenadas = sorted([f for f in fechas_convertidas if f == "Presente" or isinstance(f, pd.Timestamp)],
                              key=sort_key, reverse=True)

    # 5. Crear lista final de nombres de formato (basado en las strings originales que se parsearon o detectaron)
    lista_final_formatos = [formatos_detectados_dict.get(f_str, "Desconocido") for f_str in fechas_encontradas_str if f_str in formatos_detectados_dict]


    return fechas_ordenadas, lista_final_formatos


# In[8]:


def determinar_orden_cronologico_fechas(fechas: list) -> str:
    """
    Determina si una lista de fechas (ya extraídas y ordenadas) sugiere un orden cronológico inverso.

    Args:
        fechas: Lista de fechas (pd.Timestamp y/o "Presente") ya ordenada de más reciente a más antigua.

    Returns:
        "Orden Cronológico Inverso", "Orden No Cronológico" o "Pocas Fechas".
    """
    if not fechas or len(fechas) < 2:
        return "Pocas Fechas"

    # La función extraer_fechas_texto ya devuelve las fechas ordenadas de más reciente a más antigua.
    # Si la lista original de fechas extraídas (antes de ordenar) sigue este orden, es cronológico inverso.
    # Sin embargo, la implementación actual ordena directamente.
    # Una heurística simple es verificar si la lista *está* ordenada de forma descendente.
    # (Lo cual será siempre cierto si la ordenación funcionó, así que esta lógica necesita revisión).

    # REVISIÓN: La idea original era ver si las fechas *tal como aparecen en el texto* están en orden.
    # Esto es mucho más complejo. Una aproximación simple pero imperfecta:
    # Si la función de extracción devolvió una lista que *ya estaba* mayormente ordenada descendente,
    # podríamos asumirlo. Pero tal como está, la función ordena al final.

    # Compromiso: Asumiremos que si hay fechas y se pudieron ordenar, el candidato *intentó* un orden.
    # La calidad de ese orden es difícil de medir sin saber la posición original en el texto.
    # Por ahora, simplemente confirmamos que hay fechas suficientes.
    # Podríamos intentar comparar las posiciones de las fechas en el texto original, pero es complejo.

    # Simplificación: Si hay fechas, asumimos que el CV tiene *algún* orden temporal.
    # Diferenciar entre cronológico inverso perfecto y otros órdenes es difícil sin contexto.
    # Devolvemos "Orden Detectado" si hay fechas, o mantenemos la lógica original si se prefiere.

    # Lógica Original (adaptada): Comprueba si la lista está ordenada desc.
    # Necesitaríamos comparar con la lista antes de ordenar, lo cual no tenemos aquí.
    # Vamos a devolver un resultado más simple por ahora.
    if len(fechas) >= 2:
         # Podemos verificar si la primera fecha es "Presente" o la más reciente
         # y si la última es la más antigua. Es una verificación débil.
        return "Orden Temporal Detectado" # O mantener "Orden Cronológico" vs "Orden Funcional"
    else:
        return "Pocas Fechas"


# ### Formato de fecha más común

# In[9]:


def analizar_formato_fecha_comun(format_list: list) -> str:
    """
    Determina el formato de fecha más común de una lista de formatos detectados.

    Args:
        format_list: Lista de strings con los nombres de los formatos detectados.

    Returns:
        El nombre del formato más común, o "N/A" si la lista está vacía,
        o "Mixto" si no hay un formato claramente dominante (opcional).
    """
    if not format_list:
        return "N/A"

    count = Counter(format_list)
    # Encontrar el más común
    most_common_items = count.most_common(1)

    if not most_common_items:
         return "N/A" # No debería pasar si format_list no está vacía

    formato_comun, freq_comun = most_common_items[0]

    # Opcional: Añadir lógica para considerar si es realmente "común"
    # Por ejemplo, si representa más del 40% del total de fechas encontradas
    # total_fechas = len(format_list)
    # if freq_comun / total_fechas < 0.4:
    #     return "Mixto"

    return formato_comun


# ### Formato de texto

# In[10]:


# ### Uso de viñetas (bullet-points) o parrafos para escribir la información

def detectar_formato_texto_lineas(texto: str) -> str:
    """
    Determina si el CV usa principalmente viñetas, párrafos o es mixto, analizando líneas.

    Args:
        texto: El texto completo del CV.

    Returns:
        "Viñetas", "Párrafos" o "Mixto".
    """
    if not texto:
        return "Indeterminado"

    lineas = texto.split("\n")
    lineas_validas = [l.strip() for l in lineas if l.strip()] # Ignorar líneas vacías
    total_lineas_validas = len(lineas_validas)

    if total_lineas_validas == 0:
        return "Vacío"

    lineas_vinetas = 0
    lineas_parrafos = 0

    # Expresión regular para detectar viñetas comunes (incluye números como 1.)
    regex_vinetas = r"^\s*(?:•|\*|-|–|·|▪|▫|○|●|>|\u2022|\u2023|\u25E6|\u25AA|\u25AB|\d+\.|\([a-zA-Z\d]+\)|[a-zA-Z]\))\s+"
    # Heurísticas para párrafos (más de N palabras, más de M caracteres)
    MIN_PALABRAS_PARRAFO = 10
    MIN_CARACTERES_PARRAFO = 60

    for linea in lineas_validas:
        num_palabras = len(linea.split())

        # Considerar viñeta si empieza con el patrón O si es una línea muy corta (título/elemento corto)
        if re.match(regex_vinetas, linea) or (num_palabras > 0 and num_palabras < 5 and len(linea) < 30) :
             # La segunda condición (línea corta) es una heurística para capturar elementos de lista sin viñeta explícita
             # Podría ser demasiado agresiva y contar títulos. Ajustar si es necesario.
            lineas_vinetas += 1
        elif num_palabras >= MIN_PALABRAS_PARRAFO and len(linea) >= MIN_CARACTERES_PARRAFO:
            lineas_parrafos += 1
        # Las líneas que no cumplen ninguna condición (intermedias) no se cuentan claramente
        # pero contribuyen al total.

    # Decidir clasificación según la proporción de líneas claramente identificadas
    if total_lineas_validas == 0: # Evitar división por cero
        return "Indeterminado"

    proporcion_vinetas = lineas_vinetas / total_lineas_validas
    proporcion_parrafos = lineas_parrafos / total_lineas_validas

    # Ajustar umbrales según se necesite
    if proporcion_vinetas > 0.45: # Si casi la mitad o más son viñetas/elementos cortos
        return "Viñetas"
    elif proporcion_parrafos > 0.45: # Si casi la mitad o más son párrafos claros
        return "Párrafos"
    else:
        # Si ninguna categoría domina claramente
        return "Mixto"


# ### Tipografia

# In[11]:


# ## Diseño y Estilo de la CV

# ### Tipografía y tamaño de fuente (legibilidad, uso de negritas o cursiva)

def analizar_tipografia_fitz(doc: fitz.Document) -> dict:
    """
    Extrae información sobre la tipografía, tamaño de fuente y uso de negritas/cursivas
    usando PyMuPDF (fitz). Asegura retorno numérico (0.0) para porcentajes en todos los casos.

    Args:
        doc: El objeto fitz.Document del PDF.

    Returns:
        Un diccionario con información de tipografía.
    """
    # --- Define valores numéricos por defecto/error para los porcentajes ---
    default_values = {
        "Fuente principal": "N/A",
        "Tamaño de fuente más usado": 0,
        "Tamaño cuerpo probable": 0,
        "Legibilidad general": "Indeterminada",
        "Uso de negritas (estimado %)": 0.0, # <-- Valor numérico por defecto
        "Uso de cursivas (estimado %)": 0.0, # <-- Valor numérico por defecto
        "Variedad de fuentes": 0,
        "Variedad de tamaños": 0
    }

    if not doc:
        # Si no hay documento, actualiza el estado y devuelve los defaults
        error_values = default_values.copy()
        error_values["Legibilidad general"] = "No Aplica (sin doc)"
        logging.debug(f"Analizar tipografía: No se proporcionó documento.")
        return error_values # Retorna defaults (con 0.0 para %)

    # Variables para acumular datos
    total_spans = 0
    negritas_count = 0
    cursivas_count = 0
    all_fonts = [] # Lista para permitir conteo fácil con Counter
    all_sizes = [] # Lista para permitir conteo fácil con Counter

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # Extraer información detallada del texto
            dict_options = {"flags": fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE}
            blocks = page.get_text("dict", **dict_options)["blocks"]
            for block in blocks:
                if block["type"] == 0: # Bloque de texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            total_spans += 1
                            font_name = span["font"]
                            size = round(span["size"])
                            flags = span["flags"]

                            # Heurística para negrita/cursiva
                            es_negrita = "bold" in font_name.lower() or (flags & 16)
                            es_cursiva = "italic" in font_name.lower() or "oblique" in font_name.lower() or (flags & 4)

                            # Acumular para análisis posterior
                            all_fonts.append(font_name)
                            all_sizes.append(size)
                            if es_negrita: negritas_count += 1
                            if es_cursiva: cursivas_count += 1

    except Exception as e:
        # En caso de error durante el bucle, loguear y devolver valores por defecto/error
        logging.warning(f"Error al analizar tipografía con fitz para {doc.name if doc else 'N/A'}: {e}")
        error_values = default_values.copy()
        error_values.update({
             "Fuente principal": "Error",
             "Legibilidad general": "Error",
             "Variedad de fuentes": len(set(all_fonts)), # Contar variedad de lo acumulado hasta el error
             "Variedad de tamaños": len(set(all_sizes))
        })
        # Los porcentajes ya están en 0.0 en error_values
        return error_values

    # Si no se encontraron spans de texto en todo el documento
    if total_spans == 0:
        logging.debug(f"Analizar tipografía: No se encontraron spans de texto en {doc.name}.")
        empty_values = default_values.copy()
        empty_values.update({
            "Fuente principal": "Ninguna",
            "Legibilidad general": "Vacío"
        })
        # Los porcentajes ya están en 0.0 en empty_values
        return empty_values

    # --- Análisis normal (si se encontraron spans) ---
    font_counts = Counter(all_fonts)
    size_counts = Counter(all_sizes)

    fuente_principal = font_counts.most_common(1)[0][0] if font_counts else "N/A"
    tamano_mas_comun = size_counts.most_common(1)[0][0] if size_counts else 0

    # Heurística para tamaño cuerpo probable (ajustada)
    tamano_cuerpo_probable = 0
    if size_counts:
        sorted_sizes = size_counts.most_common()
        # Tomar el más común, pero si es > 16 (título probable) y hay más tamaños, tomar el segundo
        if sorted_sizes[0][0] > 16 and len(sorted_sizes) > 1:
            tamano_cuerpo_probable = sorted_sizes[1][0]
        else:
            tamano_cuerpo_probable = sorted_sizes[0][0]
    # Si tamano_cuerpo_probable sigue siendo 0 o muy pequeño, intentar otra heurística? Por ahora lo dejamos así.


    legibilidad = "Buena" if tamano_cuerpo_probable >= 10 and tamano_cuerpo_probable <= 14 else "Potencialmente Deficiente"
    if tamano_cuerpo_probable == 0: legibilidad = "Indeterminada"


    # --- Calcular porcentajes (total_spans > 0 está garantizado aquí) ---
    perc_negritas = round((negritas_count / total_spans) * 100, 1)
    perc_cursivas = round((cursivas_count / total_spans) * 100, 1)

    # Crear diccionario final partiendo de los defaults y actualizando
    final_results = default_values.copy()
    final_results.update({
        "Fuente principal": fuente_principal,
        "Tamaño de fuente más usado": tamano_mas_comun,
        "Tamaño cuerpo probable": tamano_cuerpo_probable,
        "Legibilidad general": legibilidad,
        "Uso de negritas (estimado %)": perc_negritas, # <-- Valor numérico calculado
        "Uso de cursivas (estimado %)": perc_cursivas, # <-- Valor numérico calculado
        "Variedad de fuentes": len(font_counts),
        "Variedad de tamaños": len(size_counts)
    })

    return final_results


# ### Colores y Graficos

# In[12]:


# ### Uso de colores y gráficos (con fitz) - CORREGIDA

def analizar_colores_graficos_fitz(doc: fitz.Document) -> dict:
    """
    Analiza un PDF en busca de imágenes y uso significativo de color en texto o dibujos,
    usando PyMuPDF (fitz). Excluye negro, blanco y grises puros.
    Maneja colores RGB (int), Grayscale (tuple len 1) y CMYK (tuple len 4).

    Args:
        doc: El objeto fitz.Document del PDF.

    Returns:
        Un diccionario con la cantidad de imágenes y si se detectó color.
    """
    cantidad_imagenes = 0
    color_texto_detectado = False
    color_dibujo_detectado = False
    if not doc:
        return {"Cantidad de imágenes": 0, "Uso de colores (texto)": "No", "Uso de colores (dibujos)": "No"}

    # --- Inicio Sub-Función es_gris CORREGIDA ---
    def es_gris(color_val, umbral_saturacion=0.05, umbral_cmy_zero=0.01):
         """
         Determina si un valor de color de fitz representa negro, blanco o gris.
         Maneja None, int (RGB), tuple (Grayscale o CMYK).
         Retorna True si es gris/negro/blanco, False si es color.
         """
         if color_val is None:
              return True # Sin color definido -> tratar como gris/no color

         # Caso 1: Entero (RGB)
         if isinstance(color_val, int):
             # Asegurarse de que no sea negativo si viene de algún sitio extraño
             if color_val < 0: return True
             # Decodificar color entero
             r = (color_val >> 16) & 0xFF
             g = (color_val >> 8) & 0xFF
             b = color_val & 0xFF
             # Convertir a flotante [0, 1]
             rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
             # Calcular max, min y delta (aproximación a saturación en RGB)
             cmax = max(rf, gf, bf)
             cmin = min(rf, gf, bf)
             delta = cmax - cmin
             # Si delta es muy bajo, es gris/blanco/negro
             return delta < umbral_saturacion

         # Caso 2: Tupla (Grayscale o CMYK)
         elif isinstance(color_val, (tuple, list)): # Aceptar lista también por si acaso
             # Subcaso 2a: Grayscale (1 elemento)
             if len(color_val) == 1:
                 # Cualquier valor de escala de grises se considera "no color" para este análisis
                 return True
             # Subcaso 2b: CMYK (4 elementos)
             elif len(color_val) == 4:
                 try:
                     c, m, y, k = color_val
                     # Si C, M, Y son prácticamente cero, es un gris/negro definido por K
                     if c < umbral_cmy_zero and m < umbral_cmy_zero and y < umbral_cmy_zero:
                         return True
                     else: # Si hay Cyan, Magenta o Amarillo significativo, es color
                         return False
                 except TypeError: # Si los elementos de la tupla no son números
                      logging.debug(f"Elemento no numérico en tupla de color CMYK: {color_val}")
                      return False # Asumir color por precaución
             # Subcaso 2c: RGB como tupla (3 elementos flotantes 0-1) - Algunos PDFs lo usan
             elif len(color_val) == 3:
                  try:
                       rf, gf, bf = color_val
                       cmax = max(rf, gf, bf)
                       cmin = min(rf, gf, bf)
                       delta = cmax - cmin
                       return delta < umbral_saturacion
                  except TypeError: # Si los elementos no son números
                      logging.debug(f"Elemento no numérico en tupla de color RGB: {color_val}")
                      return False # Asumir color
             else:
                 # Tupla de longitud inesperada, tratar como no gris por seguridad o loggear advertencia
                 logging.debug(f"Color tuple de longitud inesperada encontrado: {len(color_val)} -> {color_val}")
                 return False # Asumir que es color si no se reconoce

         # Caso 3: Otro tipo inesperado
         else:
             logging.warning(f"Tipo de color inesperado encontrado: {type(color_val)}, valor: {color_val}")
             return False # Asumir que es color si el tipo es desconocido
    # --- Fin Sub-Función es_gris CORREGIDA ---


    # --- Resto de la función analizar_colores_graficos_fitz (sin cambios) ---
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # Contar imágenes en la página
            cantidad_imagenes += len(page.get_images(full=True))

            # Revisar color del texto (si aún no se detectó)
            if not color_texto_detectado:
                # Usar get_text("dict") es más detallado para colores de span
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
                for block in blocks:
                    if block["type"] == 0: # Texto
                        for line in block["lines"]:
                            for span in line["spans"]:
                                color_val = span.get("color") # Puede ser int o None
                                if not es_gris(color_val): # Usa la función corregida
                                    color_texto_detectado = True
                                    break
                            if color_texto_detectado: break
                    if color_texto_detectado: break

            # Revisar color de los dibujos (si aún no se detectó)
            if not color_dibujo_detectado:
                 drawings = page.get_drawings()
                 for draw in drawings:
                      # 'color' es trazo, 'fill' es relleno. Pueden ser int, tuple, None.
                      color_stroke = draw.get("color")
                      color_fill = draw.get("fill")
                      if (not es_gris(color_stroke)) or (not es_gris(color_fill)): # Usa la función corregida
                          color_dibujo_detectado = True
                          break

            # Optimización: Si ya detectamos ambos tipos de color en páginas tempranas, podríamos parar
            # if color_texto_detectado and color_dibujo_detectado: break

    except Exception as e:
         logging.warning(f"Error al analizar colores/gráficos con fitz: {e}")
         # No sobrescribir la cantidad de imágenes si ya se contaron algunas
         return {"Cantidad de imágenes": cantidad_imagenes, "Uso de colores (texto)": "Error", "Uso de colores (dibujos)": "Error"}


    return {
        "Cantidad de imágenes": cantidad_imagenes,
        "Uso de colores (texto)": "Sí" if color_texto_detectado else "No",
        "Uso de colores (dibujos)": "Sí" if color_dibujo_detectado else "No"
    }


# ### Densidad de la información

# In[13]:


# ### Distribución del espacio en la hoja - Densidad de la información vs espacios en blanco

# !! VERSIÓN MODIFICADA PARA DEBUGGING: Evita get_text("blocks") !!
def calcular_densidad_informacion_fitz(doc: fitz.Document, texto_completo: str) -> float:
    """
    Calcula el porcentaje promedio de ocupación de texto en las páginas del documento.
    *** Versión Modificada para Debug: Usa len(texto) como proxy, evita get_text('blocks') ***

    Args:
        doc: El objeto fitz.Document del PDF.
        texto_completo: El texto ya extraído del documento.

    Returns:
        Un valor flotante representando la densidad aproximada (0-100), o -1.0 si hay error.
    """
    total_chars = len(texto_completo)
    total_area_paginas = 0
    if not doc:
        return 0.0

    try:
        if doc.page_count == 0:
            return 0.0 # No hay páginas, densidad cero

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_rect = page.rect
            area_pagina = page_rect.width * page_rect.height
            # Evitar división por cero si la página no tiene área válida
            if area_pagina > 0:
                 total_area_paginas += area_pagina
            else:
                 logging.debug(f"Página {page_num} sin área válida en {doc.name}")


        if total_area_paginas == 0:
            logging.warning(f"Área total de página cero para {doc.name}, no se puede calcular densidad.")
            return 0.0

        # Heurística MUY APROXIMADA: Asumir un área promedio por caracter.
        # Este factor es arbitrario y solo para evitar el crash.
        # Puedes ajustarlo si quieres que el % resultante sea más "razonable".
        APPROX_AREA_PER_CHAR = 15 # Ejemplo: 15 unidades de área por caracter (ajustar)
        densidad_aprox = (total_chars * APPROX_AREA_PER_CHAR / total_area_paginas) * 100

        # Limitar al 100%
        densidad_aprox = min(densidad_aprox, 100.0)

        return round(densidad_aprox, 2)

    except Exception as e:
        logging.warning(f"Error al calcular densidad (aproximada) de información con fitz: {e}")
        return -1.0 # Indicar error


# ### Consistencia del Formato

# In[14]:


# # ### Consistencia en el formato - Alineación, tamaños de títulos, márgenes

def analizar_consistencia_formato_fitz(doc: fitz.Document) -> dict:
    """
    Evalúa la consistencia del formato en términos de tamaños de fuente y márgenes (aproximado).

    Args:
        doc: El objeto fitz.Document del PDF.

    Returns:
        Un diccionario evaluando la consistencia.
    """
    if not doc:
        return {
            "Consistencia tamaños fuente": "Indeterminada",
            "Consistencia márgenes (aprox)": "Indeterminada",
            "Tamaños fuente detectados": [],
        }

    all_sizes = []
    margin_sets = [] # Lista de tuplas (izq, der, top, bottom) por página

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_rect = page.rect
            if page_rect.is_empty: continue

            page_sizes = set()
            min_x0, max_x1, min_y0, max_y1 = page_rect.width, 0, page_rect.height, 0
            has_content = False

            # Extraer tamaños y límites del texto
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
            for block in blocks:
                if block["type"] == 0: # Texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_sizes.add(round(span["size"]))
                            bbox = fitz.Rect(span["bbox"]) # Bbox del span
                            if not bbox.is_empty:
                                 has_content = True
                                 min_x0 = min(min_x0, bbox.x0)
                                 max_x1 = max(max_x1, bbox.x1)
                                 min_y0 = min(min_y0, bbox.y0)
                                 max_y1 = max(max_y1, bbox.y1)

            all_sizes.extend(list(page_sizes))

            if has_content:
                 margen_izq = min_x0
                 margen_der = page_rect.width - max_x1
                 margen_top = min_y0
                 # 'bottom' en fitz es y1 (el borde inferior del texto), page.height es el borde inferior de la página
                 margen_bottom = page_rect.height - max_y1
                 # Redondear márgenes para comparar consistencia
                 margin_sets.append(tuple(round(m/5)*5 for m in (margen_izq, margen_der, margen_top, margen_bottom))) # Agrupar por múltiplos de 5
            # Si no hay contenido, no añadimos márgenes para esta página

    except Exception as e:
         logging.warning(f"Error al analizar consistencia de formato con fitz: {e}")
         return {
            "Consistencia tamaños fuente": "Error",
            "Consistencia márgenes (aprox)": "Error",
            "Tamaños fuente detectados": list(set(all_sizes)),
        }


    # Evaluar consistencia
    num_distinct_sizes = len(set(all_sizes))
    num_distinct_margin_sets = len(set(margin_sets))

    # Umbrales (ajustables)
    umbral_variedad_tamanos = 5 # Más de 5 tamaños diferentes puede ser inconsistente
    umbral_variedad_margenes = 3 # Más de 3 configuraciones de márgenes distintas puede ser inconsistente

    consistencia_tamanos = "Consistente" if num_distinct_sizes <= umbral_variedad_tamanos else "Inconsistente"
    consistencia_margenes = "Consistente" if num_distinct_margin_sets <= umbral_variedad_margenes else "Inconsistente"

    if not all_sizes: consistencia_tamanos = "N/A"
    if not margin_sets: consistencia_margenes = "N/A"


    return {
        "Consistencia tamaños fuente": consistencia_tamanos,
        "Consistencia márgenes (aprox)": consistencia_margenes,
        "Tamaños fuente detectados": sorted(list(set(all_sizes))),
    }


# ### Conteo de Palabras

# In[15]:


# ## Claridad y Lenguaje

# ### Cantidad de palabras
def contar_palabras_texto(texto: str) -> int:
    """
    Cuenta la cantidad de palabras en un texto.

    Args:
        texto: El texto completo del CV.

    Returns:
        El número total de palabras.
    """
    if not texto:
        return 0
    # Simple split por espacio en blanco. Puede no ser perfecto para todos los casos.
    palabras = texto.split()
    return len(palabras)


# In[16]:


# ### Presencia de errores ortográficos o gramaticales (Función Original Comentada)
# #NOTA: Esta función sigue siendo experimental y puede tener problemas de rendimiento/precisión.
# #Requiere instalar language_tool_python y pyspellchecker
# import language_tool_python
# from spellchecker import SpellChecker
#
# def detectar_errores_ortograficos_gramaticales_texto(texto: str, idioma="en"):
#     """
#     Detecta errores ortográficos y gramaticales en un texto. (EXPERIMENTAL)
#     """
#     resultados = {
#         "Errores ortográficos detectados": "No ejecutado",
#         "Ejemplo de errores ortográficos": {},
#         "Cantidad de errores gramaticales": "No ejecutado",
#     }
#     if not texto:
#         resultados["Errores ortográficos detectados"] = 0
#         resultados["Cantidad de errores gramaticales"] = 0
#         return resultados
#
#     try:
#         # Inicializar correctores
#         spell = SpellChecker(language=idioma)
#         # Usar servidor local puede ser más rápido/privado si se configura
#         tool = language_tool_python.LanguageToolPublicAPI(idioma)
#
#         # Limpieza básica del texto (podría necesitar más refinamiento)
#         texto_limpio = texto.replace("\n", " ")
#         texto_limpio = re.sub(r"\s+", " ", texto_limpio)
#
#         # Ortografía (requiere procesamiento adicional para ser preciso)
#         # Normalizar a minúsculas y quitar puntuación para spellchecker
#         texto_para_spell = re.sub(r"[^a-z\s]", "", texto_limpio.lower())
#         palabras = [palabra for palabra in texto_para_spell.split() if len(palabra) > 2]
#         errores_ortograficos_dict = {}
#         if palabras:
#             unknown_words = spell.unknown(palabras)
#             errores_ortograficos_dict = {word: spell.correction(word) for word in unknown_words}
#         resultados["Errores ortográficos detectados"] = len(errores_ortograficos_dict)
#         resultados["Ejemplo de errores ortográficos"] = dict(list(errores_ortograficos_dict.items())[:10])
#
#         # Gramática (procesar por fragmentos para evitar timeouts/errores)
#         # Dividir por frases o párrafos podría ser mejor que por longitud fija
#         max_len_grammar = 1000 # LanguageTool puede tener límites
#         errores_gramaticales_count = 0
#         for i in range(0, len(texto_limpio), max_len_grammar):
#             fragmento = texto_limpio[i:i+max_len_grammar]
#             try:
#                  errores_gramaticales_count += len(tool.check(fragmento))
#             except Exception as e_lt:
#                  logging.debug(f"Error en fragmento de LanguageTool: {e_lt}")
#                  # Continuar con el siguiente fragmento
#
#         resultados["Cantidad de errores gramaticales"] = errores_gramaticales_count
#         tool.close() # Cerrar conexión si es necesario
#
#     except Exception as e:
#         logging.error(f"Error general en detección de errores orto/gram: {e}")
#         resultados["Errores ortográficos detectados"] = "Error"
#         resultados["Cantidad de errores gramaticales"] = "Error"
#
#     return resultados


# ### Terminos técnicos

# In[17]:


# ### Uso de lenguaje técnico o genérico

def analizar_lenguaje_tecnico_texto(texto: str, terminos_tecnicos: set) -> dict:
    """
    Calcula el porcentaje de términos técnicos encontrados en el texto.

    Args:
        texto: El texto completo del CV.
        terminos_tecnicos: Un set con los términos técnicos a buscar.

    Returns:
        Un diccionario con el porcentaje de lenguaje técnico y genérico.
    """
    resultados = {"Porcentaje Lenguaje Técnico": 0.0, "Porcentaje Lenguaje Genérico": 100.0}
    if not texto or not terminos_tecnicos:
        return resultados

    try:
        # Normalizar a minúsculas y extraer palabras (solo alfanuméricas)
        palabras = re.findall(r'\b[a-zA-Z0-9#+.]+\b', texto.lower()) # Incluir #, +, . para C#, .NET etc.
        total_palabras = len(palabras)

        if total_palabras == 0:
            return resultados

        # Contar términos técnicos encontrados
        # Normalizar también los términos técnicos a minúsculas para comparación
        terminos_tecnicos_lower = {term.lower() for term in terminos_tecnicos}
        tech_words_count = sum(1 for palabra in palabras if palabra in terminos_tecnicos_lower)

        # Calcular porcentaje
        porcentaje_tecnico = (tech_words_count / total_palabras) * 100

        resultados["Porcentaje Lenguaje Técnico"] = round(porcentaje_tecnico, 2)
        resultados["Porcentaje Lenguaje Genérico"] = round(100 - porcentaje_tecnico, 2)

    except Exception as e:
        logging.warning(f"Error al analizar lenguaje técnico: {e}")
        resultados["Porcentaje Lenguaje Técnico"] = "Error"
        resultados["Porcentaje Lenguaje Genérico"] = "Error"

    return resultados


# ### Enlaces

# In[18]:


# ## Personalización y adaptabilidad

# ### Inclusión de enlaces a portafolio o redes sociales (con fitz)

def extract_links_fitz(doc: fitz.Document) -> dict:
    """
    Extrae enlaces de un PDF usando fitz, clasificando LinkedIn, GitHub y otros.

    Args:
        doc: El objeto fitz.Document del PDF.

    Returns:
        Un diccionario indicando si se encontraron enlaces de cada tipo.
    """
    links_encontrados = {"LinkedIn": False, "GitHub": False, "Website/Otro": False}
    if not doc:
        return links_encontrados

    linkedin_pattern = r"linkedin\.com\/in\/|linkedin\.com\/pub\/"
    github_pattern = r"github\.com\/"

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_links = page.get_links() # Devuelve una lista de diccionarios de enlaces

            for link in page_links:
                uri = link.get("uri")
                if uri:
                    uri_lower = uri.lower()
                    if re.search(linkedin_pattern, uri_lower):
                        links_encontrados["LinkedIn"] = True
                    elif re.search(github_pattern, uri_lower):
                        links_encontrados["GitHub"] = True
                    # Considerar otros enlaces HTTP/HTTPS como website/otro
                    elif uri_lower.startswith("http://") or uri_lower.startswith("https://"):
                         # Evitar contar los ya clasificados si la URL es la misma (poco probable)
                         if not (re.search(linkedin_pattern, uri_lower) or re.search(github_pattern, uri_lower)):
                              links_encontrados["Website/Otro"] = True

                # Si ya encontramos todos los tipos, podemos parar
                if all(links_encontrados.values()):
                    break
            if all(links_encontrados.values()):
                break

    except Exception as e:
        logging.warning(f"Error al extraer enlaces con fitz: {e}")
        return {"LinkedIn": "Error", "GitHub": "Error", "Website/Otro": "Error"}

    # Convertir a Sí/No si se prefiere
    # return {k: "Sí" if v else "No" for k, v in links_encontrados.items()}
    return links_encontrados


# ### Detectar foto

# In[19]:


# ## Información visual y Complementaria

# ### Incorporación de foto o ausencia de la misma (con fitz y heurística)

def tiene_foto_fitz(doc: fitz.Document, min_size_px=100, max_pos_y_ratio=0.25) -> str:
    """
    Detecta si hay una imagen que *podría* ser una foto de perfil.
    Busca imágenes de tamaño razonable en la parte superior del documento.

    Args:
        doc: El objeto fitz.Document del PDF.
        min_size_px: Tamaño mínimo (ancho o alto) en píxeles para considerar una imagen.
        max_pos_y_ratio: Posición vertical máxima (relativa al alto de página, 0.0=arriba, 1.0=abajo)
                         donde buscar la foto (ej. 0.25 = en el 25% superior).

    Returns:
        "Posible Foto Detectada", "No se detectaron imágenes candidatas", "Error" o "No Aplica (sin doc)".
    """
    if not doc:
        return "No Aplica (sin doc)"

    try:
        for page_num in range(min(doc.page_count, 2)): # Revisar solo las primeras 2 páginas
            page = doc.load_page(page_num)
            page_rect = page.rect
            if page_rect.is_empty: continue

            img_list = page.get_images(full=True)
            if not img_list: continue

            for img_index, img_info in enumerate(img_list):
                xref = img_info[0]
                # Obtener rectángulos donde se usa la imagen
                rects = page.get_image_rects(xref)
                if not rects: continue

                # Analizar la primera instancia de la imagen en la página
                img_rect = rects[0] # fitz.Rect
                img_width = img_rect.width
                img_height = img_rect.height
                img_y0 = img_rect.y0 # Posición superior de la imagen

                # Heurística: ¿Es suficientemente grande Y está en la parte superior?
                if (img_width >= min_size_px or img_height >= min_size_px) and \
                   (img_y0 / page_rect.height <= max_pos_y_ratio):
                    # Podríamos añadir más chequeos: relación de aspecto, si es la única imagen grande arriba, etc.
                    return "Posible Foto Detectada"

        return "No se detectaron imágenes candidatas"

    except Exception as e:
        logging.warning(f"Error al detectar foto con fitz: {e}")
        return "Error en detección de foto"


# ### Detectar elementos gráficos

# In[20]:


# ### Presencia de íconos o elementos gráficos (con fitz)

def detectar_elementos_graficos_fitz(doc: fitz.Document, umbral_area: int, max_elementos: int) -> bool:
    """
    Detecta si un CV tiene elementos gráficos vectoriales significativos.

    Args:
        doc: El objeto fitz.Document del PDF.
        umbral_area: Área mínima (width*height) para considerar relevante.
        max_elementos: Número mínimo de elementos grandes para retornar True.

    Returns:
        True si se detectan suficientes elementos gráficos grandes, False en caso contrario o error.
    """
    elementos_grandes_count = 0
    if not doc:
        return False

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            drawings = page.get_drawings() # Obtiene paths de dibujo vectorial

            for draw in drawings:
                rect = draw["rect"] # fitz.Rect object
                if rect.is_empty or rect.is_infinite: continue

                area = rect.width * rect.height
                if area > umbral_area:
                    elementos_grandes_count += 1
                    if elementos_grandes_count >= max_elementos:
                        return True # Suficientes elementos encontrados

        return False # No se alcanzó el umbral de elementos grandes

    except Exception as e:
        logging.warning(f"Error al detectar elementos gráficos con fitz: {e}")
        return False # Asumir Falso en caso de error


# ### Procesar CV

# In[21]:


# ## Función Principal de Procesamiento de un CV
def process_cv(pdf_path: str, spacy_nlp_model) -> dict:
    logging.info(f"Procesando archivo: {os.path.basename(pdf_path)}")
    results = {"archivo": os.path.basename(pdf_path), "status_procesamiento": "Iniciado"}
    doc = None # Inicializar doc a None

    try:
        # 1. Abrir PDF con Fitz
        try:
            doc = fitz.open(pdf_path)
        except Exception as e_open:
            logging.error(f"No se pudo abrir el archivo PDF: {pdf_path}. Error: {e_open}")
            results["status_procesamiento"] = f"Error al abrir PDF: {e_open}"
            return results # Salir si no se puede abrir

        # 2. Extraer Texto Completo (una sola vez)
        texto_completo = ""
        try:
            for page in doc:
                texto_completo += page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES) + "\n" # flags opcional
            texto_completo = texto_completo.strip()
            if not texto_completo:
                 logging.warning(f"No se pudo extraer texto de: {pdf_path}")
                 results["status_procesamiento"] = "Advertencia: No se extrajo texto"
                 # Continuar igualmente para análisis basados en fitz.Document
        except Exception as e_text:
            logging.error(f"Error extrayendo texto de {pdf_path}: {e_text}")
            results["status_procesamiento"] = f"Error extrayendo texto: {e_text}"
            # Continuar igualmente si es posible

        results["texto_extraido_len"] = len(texto_completo) # Guardar longitud para referencia

        # --- Llamada a las funciones de análisis ---

        # Estructura y Organización
        results["Numero de Paginas"] = contar_paginas_fitz(doc)
        secciones_detectadas = detectar_secciones_texto(texto_completo, SECCIONES_CV_DICT, UMBRAL_SIMILITUD_SECCION)
        results.update({f"Seccion_{k}": v for k,v in secciones_detectadas.items()}) # Añadir prefijo

        lineas_seccion_results = calcular_lineas_por_seccion(texto_completo, SECCIONES_CV_DICT, UMBRAL_SIMILITUD_SECCION)
        results.update(lineas_seccion_results) # Fusionar diccionario de resultados de líneas


        fechas_extraidas, formatos_detectados = extraer_fechas_texto(texto_completo, PATRONES_FECHAS_FORMATOS, nlp)
        results["Orden Temporal"] = determinar_orden_cronologico_fechas(fechas_extraidas)
        results["Fechas Detectadas (Count)"] = len(fechas_extraidas)
        results["Formato Fecha Más Común"] = analizar_formato_fecha_comun(formatos_detectados)
        # results["Fechas Recientes (ej)"] = [str(f) for f in fechas_extraidas[:3]] # Opcional: guardar ejemplos

        results["Formato Texto (Lineas)"] = detectar_formato_texto_lineas(texto_completo)

        # Diseño y Estilo
        typo_results = analizar_tipografia_fitz(doc)
        results.update(typo_results) # Fusionar diccionario de resultados

        color_graph_results = analizar_colores_graficos_fitz(doc)
        results.update(color_graph_results)

        results["Densidad Informacion (%)"] = calcular_densidad_informacion_fitz(doc, texto_completo)

        consistency_results = analizar_consistencia_formato_fitz(doc)
        results.update(consistency_results)

        # Claridad y Lenguaje
        results["Cantidad de Palabras"] = contar_palabras_texto(texto_completo)
        # results.update(detectar_errores_ortograficos_gramaticales_texto(texto_completo)) # Si se activa

        tech_lang_results = analizar_lenguaje_tecnico_texto(texto_completo, TECH_TERMS)
        results.update(tech_lang_results)

        # Personalización y Adaptabilidad
        link_results = extract_links_fitz(doc)
        results.update(link_results)

        # Información Visual y Complementaria
        results["Deteccion Foto Perfil"] = tiene_foto_fitz(doc)
        results["Tiene Elementos Graficos"] = detectar_elementos_graficos_fitz(doc, MIN_AREA_ELEMENTO_GRAFICO, MIN_ELEMENTOS_GRAFICOS_GRANDES)

        results["status_procesamiento"] = "Completado" if results["status_procesamiento"] == "Iniciado" else results["status_procesamiento"] # Mantener advertencias/errores

    except Exception as e_main:
        logging.exception(f"Error fatal procesando {pdf_path}: {e_main}") # Log con traceback
        results["status_procesamiento"] = f"Error Fatal: {e_main}"
        # Asegurarse de que las claves principales existan aunque sea con valor de error
        keys_expected = ["Numero de Paginas", "Orden Temporal", "Formato Texto (Lineas)",
                         "Fuente principal", "Densidad Informacion (%)", "Cantidad de Palabras",
                         "Porcentaje Lenguaje Técnico", "LinkedIn", "Deteccion Foto Perfil",
                         "Tiene Elementos Graficos"]
        for k in keys_expected:
            if k not in results:
                results[k] = "Error"

    finally:
        # Asegurarse de cerrar el documento PDF
        if doc:
            try:
                doc.close()
            except Exception as e_close:
                logging.error(f"Error al cerrar el archivo PDF {pdf_path}: {e_close}")

    return results


# ### Ejecutar

# In[22]:


# ## Ejecución Principal del Script

if __name__ == "__main__":
    logging.info("Iniciando el script de análisis de CV.")

    if not pdf_files_to_process:
        logging.warning("No se encontraron archivos PDF para procesar. Finalizando.")
    else:
        all_results = []
        errores_procesamiento = 0

        for pdf_file, category in pdf_files_to_process:
            # Pasar el modelo nlp cargado a la función
            cv_data = process_cv(pdf_file, nlp)
            # *** MODIFICACIÓN 1: Asegurarse de guardar la categoría original ***
            # (Esto ya estaba, solo confirmar que 'category' viene de la tupla)
            cv_data["Categoria Original"] = category
            all_results.append(cv_data)
            if "Error" in cv_data.get("status_procesamiento", ""):
                errores_procesamiento += 1

        logging.info(f"Procesamiento completado. Total CV: {len(all_results)}, Errores: {errores_procesamiento}")

        # Convertir resultados a DataFrame de Pandas
        if all_results:
            df_results = pd.DataFrame(all_results)

            # *** MODIFICACIÓN 2: Crear la columna 'Passed' a partir de 'Categoria Original' ***
            if 'Categoria Original' in df_results.columns:
                # Definir el mapeo: "Paso" (Exitoso) -> 1, "No paso" (No Exitoso) -> 0
                category_map = {"Exitoso": 1, "No Exitoso": 0}
                df_results['Passed'] = df_results['Categoria Original'].map(category_map)
                # Opcional: manejar casos donde la categoría no sea ninguna de las esperadas
                df_results['Passed'] = df_results['Passed'].fillna(-1).astype(int) # Rellenar con -1 y convertir a entero
                logging.info("Columna 'Passed' creada (1 para Exitoso/Paso, 0 para No Exitoso/No paso).")
            else:
                logging.warning("No se encontró la columna 'Categoria Original'. No se pudo crear la columna 'Passed'.")
                df_results['Passed'] = -1 # Añadir columna con valor por defecto si no existe la original

            # *** MODIFICACIÓN 3: Añadir 'Passed' al orden de columnas deseado ***
            column_order = [
                'archivo', 'Categoria Original', 'Passed', 'status_procesamiento', # <-- 'Passed' añadida aquí
                'Numero de Paginas',
                'Cantidad de Palabras', 'Densidad Informacion (%)', 'Formato Texto (Lineas)',
                'Orden Temporal', 'Fechas Detectadas (Count)', 'Formato Fecha Más Común',
                'Fuente principal', 'Tamaño cuerpo probable', 'Legibilidad general',
                'Variedad de fuentes', 'Variedad de tamaños', 'Consistencia tamaños fuente',
                'Consistencia márgenes (aprox)', 'Uso de negritas (estimado %)', 'Uso de cursivas (estimado %)',
                'Uso de colores (texto)', 'Uso de colores (dibujos)',
                'Porcentaje Lenguaje Técnico', 'Porcentaje Lenguaje Genérico',
                'LinkedIn', 'GitHub', 'Website/Otro',
                'Deteccion Foto Perfil', 'Cantidad de imágenes', 'Tiene Elementos Graficos',
                *[f"Lineas_{k}" for k in SECCIONES_CV_DICT.keys() if f"Lineas_{k}" in df_results.columns],
                 # Añadir columnas de secciones detectadas al final
                 *[f"Seccion_{k}" for k in SECCIONES_CV_DICT.keys() if f"Seccion_{k}" in df_results.columns] # Asegurar que existen
            ]
            # Filtrar para mantener solo columnas existentes en el DataFrame y en el orden deseado
            column_order_existing = [col for col in column_order if col in df_results.columns]
            # Añadir columnas que no estaban en column_order pero sí en el df (por si acaso)
            remaining_cols = [col for col in df_results.columns if col not in column_order_existing]
            final_column_order = column_order_existing + remaining_cols
            df_results = df_results[final_column_order]


            # Guardar DataFrame en un archivo CSV
            output_filename = "../Bases/base_cvs/base_cvs_mejorado.csv"
            try:
                df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
                logging.info(f"Resultados guardados en: {output_filename}")
            except Exception as e_save:
                logging.error(f"Error al guardar los resultados en CSV: {e_save}")

            # Opcional: Mostrar las primeras filas del DataFrame
            print("\n--- Primeras filas de los resultados (con columna 'Passed') ---")
            print(df_results[['archivo', 'Categoria Original', 'Passed']].head()) # Mostrar columnas clave
            print("\n--- Resumen del DataFrame ---")
            print(df_results.info()) # Descomentar si quieres ver el resumen completo

        else:
            logging.info("No se generaron resultados para guardar.")

    logging.info("Script finalizado.")


# In[23]:


df_results

