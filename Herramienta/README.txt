===============================================================================
Herramienta de Análisis de CVs y Asignación de Perfil (Cluster)
===============================================================================

Fecha de Creación: 6/5/2025
Autor: Manuel Vecino
Versión del Script: 1.0

-------------------------------------------------------------------------------
Descripción
-------------------------------------------------------------------------------

Este script en Python analiza un archivo de Curriculum Vitae (CV) en formato PDF
para extraer 4 características clave:
    1. Longitud total del texto extraído (`texto_extraido_len`).
    2. Número de secciones principales detectadas (`secciones_completas`).
    3. Presencia de un enlace a un sitio web personal o portafolio (`Website/Otro`).
    4. Presencia de una sección dedicada a cursos o formación (`Seccion_training_courses`).

Utilizando estas 4 características, el script asigna el CV a uno de los 3 perfiles
(clusters) predefinidos, los cuales han mostrado tener diferentes tasas promedio
de éxito ('Passed') en análisis previos. Los perfiles identificados son:

* **Cluster 0:** "Con Website y Sección Training" (Tasa de 'Passed' más alta)
* **Cluster 1:** "Sin Training (y Mayormente sin Website) - Corto" (Tasa de 'Passed' más baja)
* **Cluster 2:** "Con Sección Training, Sin Website (Grupo Estándar)"

Este análisis proporciona una visión descriptiva del perfil del CV.

-------------------------------------------------------------------------------
Requisitos Previos
-------------------------------------------------------------------------------

Antes de ejecutar el script, asegúrate de tener instalado lo siguiente:

1.  **Python:** Versión 3.8 o superior recomendada. Puedes descargarlo desde [https://www.python.org/downloads/](https://www.python.org/downloads/).
    * Durante la instalación en Windows, asegúrate de marcar la casilla "Add Python to PATH".

2.  **Archivos de Modelo:** Debes tener los siguientes archivos en la misma carpeta que
    el script `Herramienta.py` (o ajustar las rutas en el comando de ejecución):
    * `kmeans_scaler_k3_4f.joblib`: El objeto StandardScaler entrenado.
    * `kmeans_model_k3_4f.joblib`: El modelo KMeans (K=3) entrenado.

-------------------------------------------------------------------------------
Instalación de Dependencias
-------------------------------------------------------------------------------

Este script requiere varias librerías de Python. La forma más sencilla de
instalarlas es usando `pip` dentro de un entorno virtual.

1.  **Abrir una Terminal o Símbolo del Sistema:**
    * **Windows:** Busca "cmd" o "PowerShell".
    * **macOS/Linux:** Abre la aplicación "Terminal".

2.  **Navegar al Directorio del Proyecto:**
    Usa el comando `cd` para moverte a la carpeta donde descargaste este
    script y los archivos de modelo. Ejemplo:
    cd ruta/a/la/carpeta_del_script

3.  **(Recomendado) Crear y Activar un Entorno Virtual:**
    * Crea el entorno (solo una vez):
        ```bash
        python -m venv mi_entorno_cv
        ```
    * Activa el entorno:
        * Windows:
            ```bash
            mi_entorno_cv\Scripts\activate
            ```
        * macOS/Linux:
            ```bash
            source mi_entorno_cv/bin/activate
            ```
        Deberías ver `(mi_entorno_cv)` al inicio del prompt de tu terminal.

4.  **Instalar las Librerías Requeridas:**
    Copia el archivo `requirements.txt` (que debe estar en la misma carpeta que este README)
    y ejecuta:
    ```bash
	python -m pip install --upgrade pip
    
	pip install -r requirements.txt
    ```
    Esto instalará: PyMuPDF, rapidfuzz, numpy, pandas, joblib, scikit-learn, y
    sus dependencias.

-------------------------------------------------------------------------------
Uso del Script
-------------------------------------------------------------------------------

Una vez instaladas las dependencias y con el entorno virtual activado (si creaste uno),
puedes ejecutar el script desde la terminal.

**Comando Básico:**

```bash
python Herramienta.py CV/[AQUI MODIFICA EL NOMBRE POR EL DEL ARCHIVO DE CV A ANALIZAR].pdf --scaler kmeans_scaler_k3_4f.joblib --kmeans kmeans_model_k3_4f.joblib

python Herramienta.py CV/sample.pdf --scaler kmeans_scaler_k3_4f.joblib --kmeans kmeans_model_k3_4f.joblib

Copia y pega esto en la terminal. 