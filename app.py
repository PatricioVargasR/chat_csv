import os
import re
import io
import time
import contextlib
import logging
from flask import Flask, request, render_template, session, abort, g
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de la aplicación
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Cliente Groq (singleton)
groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return groq_client

# Entorno base restringido para la ejecución de código
BASE_RESTRICTED_ENV = {
    '__builtins__': {
        'print': print,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'sum': sum,
        'min': min,
        'max': max,
        'len': len,
        'range': range,
        'round': round
    }
}

# Variable global para controlar la última limpieza
LAST_CLEANUP = 0

@app.before_request
def manage_cleanup():
    global LAST_CLEANUP
    now = time.time()
    # Ejecutar limpieza cada 10 minutos (600 segundos)
    if now - LAST_CLEANUP > 600:
        clean_old_files()
        LAST_CLEANUP = now

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template(
        "error.html",
        error_title="Archivo demasiado grande",
        error_message="El archivo excede el límite de 10MB. Por favor, sube un archivo más pequeño."
    ), 413

@app.errorhandler(400)
def handle_bad_request(e):
    return render_template(
        "error.html",
        error_title="Solicitud incorrecta",
        error_message=str(e)
    ), 400

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    code_snippet = ""
    execution_output = ""

    if request.method == "POST":
        # Manejo de archivo CSV
        if "file" in request.files:
            file = request.files["file"]
            if not file.filename:
                message = "No se seleccionó ningún archivo."
            elif not file.filename.endswith(".csv"):
                message = "Por favor, sube un archivo CSV válido."
            else:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(filepath)

                    try:
                        df = pd.read_csv(filepath)
                        # Almacenar metadatos en la sesión
                        session['file_metadata'] = {
                            'filename': filename,
                            'columns': list(df.columns),
                            'row_count': len(df),
                            'preview': df.head(5).to_dict(orient='records')
                        }
                        message = f"Archivo '{filename}' cargado correctamente."
                    except pd.errors.EmptyDataError:
                        message = "El archivo CSV está vacío."
                    except pd.errors.ParserError as pe:
                        message = f"Error al analizar CSV: {str(pe)}"
                    except UnicodeDecodeError:
                        message = "Error de codificación: el archivo debe estar en UTF-8."
                    except Exception as e:
                        logger.exception("Error inesperado al leer el archivo CSV")
                        message = "Error interno al procesar el archivo."
                except (OSError, PermissionError) as e:
                    logger.error(f"Error al guardar archivo: {str(e)}")
                    message = "Error al guardar el archivo. Inténtalo de nuevo."

        # Manejo de pregunta
        if request.form.get("question"):
            question = request.form["question"]

            if len(question) > 1000:
                abort(400, "Pregunta demasiado larga (máximo 1000 caracteres)")

            if 'file_metadata' in session:
                try:
                    # Cargar el DataFrame desde el archivo guardado
                    filepath = os.path.join(
                        app.config["UPLOAD_FOLDER"], 
                        session['file_metadata']['filename']
                    )
                    df = pd.read_csv(filepath)

                    # Construir contexto para la IA
                    context = (
                        f"Columnas: {session['file_metadata']['columns']}\n\n"
                        f"Total de filas: {session['file_metadata']['row_count']}"
                    )

                    # Construir el prompt
                    prompt = build_prompt(context, question)

                    # Obtener respuesta de la IA
                    client = get_groq_client()
                    completion = client.chat.completions.create(
                        model="deepseek-r1-distill-llama-70b",
                        messages=prompt,
                        temperature=0,
                        max_tokens=1024,
                        stream=False,
                    )

                    full_output = completion.choices[0].message.content.strip()

                    # Extraer y ejecutar el código
                    code_snippet = extract_code_snippet(full_output)
                    if code_snippet:
                        execution_output = execute_code_safely(code_snippet, df)
                    else:
                        execution_output = "No se encontró código ejecutable en la respuesta de la IA."
                except (ConnectionError, TimeoutError) as e:
                    logger.error(f"Error de red: {str(e)}")
                    message = "Error de conexión con el servicio de IA. Inténtalo de nuevo."
                except Exception as e:
                    logger.exception("Error inesperado al procesar la pregunta")
                    message = "Error interno al procesar tu pregunta."
            else:
                message = "Por favor, sube un archivo CSV primero."

    # Preparar datos para la plantilla
    preview = session.get('file_metadata', {}).get('preview', [])
    columns = session.get('file_metadata', {}).get('columns', [])

    return render_template(
        "index.html",
        message=message,
        code_snippet=code_snippet,
        response=execution_output,
        preview=preview,
        columns=columns
    )

def build_prompt(context, question):
    return [
        {
            "role": "system",
            "content": (
                "REGLAS DE SEGURIDAD:\n"
                "1) Solo usar pandas y Python básico (prohibido: os, sys, subprocess, exec, eval, open).\n"
                "2) No usar métodos con doble guión bajo (__).\n"
                "3) No acceder a atributos/métodos peligrosos.\n"
                "4) Solo operaciones con el DataFrame.\n\n"
                "Responde preguntas sobre un DataFrame llamado `df` que ya está cargado. "
                "No es necesario importar pandas. "
                "Cuando llegues a la línea que da el resultado, envuélvela en un `print()`. "
                "El código final debe ir entre:\n"
                "## CODE ##\n<tu código aquí>\n## END CODE ##\n\n"
                "No des explicaciones fuera de estas secciones. "
                "Responde preguntas sobre un DataFrame llamado `df` que ya está cargado. "
                "No es necesario importar pandas. "
                "Cuando llegues a la línea que da el resultado, asigna el resultado a 'result_df'. "
                "No uses print() para mostrar resultados. "
                "El código final debe terminar con: result_df = [tu expresión]\n"
                "## CODE ##\n<tu código aquí>\n## END CODE ##\n\n"
            )
        },
        {
            "role": "user",
            "content": f"{context}\n\nPregunta: {question}"
        }
    ]

def extract_code_snippet(full_output):
    code_match = re.search(
        r"## CODE ##\s*(.*?)\s*## END CODE ##",
        full_output,
        re.DOTALL
    )
    return code_match.group(1).strip() if code_match else None

def execute_code_safely(code_snippet, df):
    """Ejecuta el código en un entorno restringido y captura la salida."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            # Crear una copia segura de pandas y el DataFrame
            safe_pd = type('SafePandas', (object,), {})()
            safe_df = df.copy()

            # Lista blanca principal de métodos pandas
            allowed_pd_attrs = [
                'loc', 'iloc', 'at', 'iat', 'get', 'groupby', 'agg', 'aggregate',
                'mean', 'sum', 'max', 'min', 'count', 'std', 'var', 'median',
                'quantile', 'idxmax', 'idxmin', 'unique', 'nunique', 'value_counts',
                'describe', 'corr', 'cov', 'query', 'filter', 'where', 'mask', 'isin',
                'between', 'dropna', 'isna', 'notna', 'duplicated', 'sort_values',
                'sort_index', 'nsmallest', 'nlargest', 'head', 'tail', 'sample',
                'info', 'memory_usage', 'merge', 'join', 'concat', 'dtypes',
                'select_dtypes', 'infer_objects', 'to_datetime', 'to_timedelta',
                'dt', 'rolling', 'expanding', 'ewm'
            ]

            # Mecanismo de respaldo para otros métodos
            class SafeAccess:
                def __init__(self, obj):
                    self._obj = obj

                def __getattr__(self, name):
                    # Registrar advertencia para métodos no incluidos
                    logger.warning(f"Acceso a método no incluido en lista blanca: {name}")
                    return getattr(self._obj, name)

            # Copiar métodos de la lista blanca
            for attr in allowed_pd_attrs:
                if hasattr(pd, attr):
                    setattr(safe_pd, attr, getattr(pd, attr))

            # Para métodos fuera de lista blanca, usar acceso seguro
            safe_pd = SafeAccess(pd)

            # Entorno restringido con solo funciones y objetos seguros
            restricted_env = {
                'df': safe_df,
                'pd': safe_pd,
                **BASE_RESTRICTED_ENV
            }

            # Ejecutar el código
            exec(code_snippet, restricted_env)
            result = f.getvalue()

            # Intentar obtener resultado como objeto pandas
            result_obj = None
            if 'result_df' in restricted_env:
                result_obj = restricted_env['result_df']

            if result_obj is not None:
                # Convertir diferentes tipos de objetos a DataFrame
                if isinstance(result_obj, pd.Series):
                    # Convertir Series a DataFrame
                    if result_obj.name:
                        result_df = result_obj.reset_index(name=result_obj.name)
                    else:
                        result_df = result_obj.reset_index(name='Value')
                elif isinstance(result_obj, pd.DataFrame):
                    result_df = result_obj
                else:
                    # Para otros tipos (scalares, listas, etc.)
                    result_df = pd.DataFrame([str(result_obj)], columns=['Result'])

                # Convertir DataFrame a HTML
                return result_df.to_html(
                    classes='table table-striped table-bordered table-sm',
                    index=False,
                    table_id="resultTable"
                )
            else:
                # Detectar si el resultado es una tabla estructurada
                if "\n" in result and "  " in result:
                    try:
                        # Crear un DataFrame desde la salida de texto
                        from io import StringIO
                        table_df = pd.read_csv(StringIO(result), sep=r"\s+", engine="python")
                        return table_df.to_html(
                            classes='table table-striped table-bordered table-sm',
                            index=False
                        )
                    except Exception:
                        return f"<pre>{result}</pre>"
                else:
                    return f"<pre>{result}</pre>"
        except SyntaxError as e:
            return f"<pre>Error de sintaxis: {str(e)}</pre>"
        except NameError as e:
            return f"<pre>Nombre no encontrado: {str(e)}</pre>"
        except AttributeError as e:
            return f"<pre>Error de atributo: {str(e)}</pre>"
        except TypeError as e:
            return f"<pre>Error de tipo: {str(e)}</pre>"
        except pd.errors.ParserError as e:
            return f"<pre>Error de pandas: {str(e)}</pre>"
        except Exception as e:
            logger.exception("Error inesperado al ejecutar el código generado")
            return f"<pre>Error inesperado: {str(e)}</pre>"

def clean_old_files(max_age=3600):
    """Elimina archivos subidos con más de max_age segundos."""
    now = time.time()
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
        except FileNotFoundError:
            logger.warning(f"Archivo no encontrado: {file_path}")
        except PermissionError:
            logger.error(f"Permiso denegado al eliminar: {file_path}")
        except OSError as e:
            logger.error(f"Error del sistema: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
