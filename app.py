from flask import Flask, request, render_template
import os
import pandas as pd
import re
import io
import contextlib
import html  # Import nuevo para escapado HTML
from groq import Groq
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite 16MB

# Cliente de Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = None  # DataFrame global temporal

# Entorno seguro para exec
SAFE_GLOBALS = {
    'pd': pd,
    'print': print,
    'len': len,
    'str': str,
    'int': int,
    'float': float,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'bool': bool,
    'range': range
}

@app.route("/", methods=["GET", "POST"])
def index():
    global df
    message = ""
    code_snippet = ""
    execution_output = ""

    if request.method == "POST":
        # Manejo de archivo CSV
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "" and file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                try:
                    df = pd.read_csv(filepath)
                    message = f"Archivo '{file.filename}' cargado correctamente."
                except Exception as e:
                    message = f"Error leyendo CSV: {str(e)}"
            else:
                message = "Por favor, sube un archivo CSV válido."

        # Manejo de pregunta
        if request.form.get("question"):
            question = request.form["question"]
            if df is not None:
                # Contexto: primeras 5 filas
                context = (
                    f"Columnas: {list(df.columns)}\n\n"
                    f"Ejemplo de datos (primeras 5 filas):\n{df.head().to_string(index=False)}"
                )

                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Responde preguntas sobre un DataFrame llamado `df` que ya está cargado. "
                            "No importes pandas. Usa `pd` para operaciones con pandas. "
                            "Envuelve el resultado final en `print()`. "
                            "Razonamiento entre <think> y </think>. "
                            "Código ejecutable entre:\n"
                            "## CODE ##\n<tu código aquí>\n## END CODE ##\n"
                            "Solo código ejecutable entre las marcas."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nPregunta: {question}"
                    }
                ]

                try:
                    completion = client.chat.completions.create(
                        model="deepseek-r1-distill-llama-70b",
                        messages=prompt,
                        temperature=0,
                        max_tokens=1024,
                        top_p=1.0,
                        stream=False,
                    )

                    full_output = completion.choices[0].message.content.strip()

                    # Extraer bloque de código
                    code_match = re.search(
                        r"## CODE ##\s*(.*?)\s*## END CODE ##",
                        full_output,
                        re.DOTALL
                    )
                    if code_match:
                        code_snippet = code_match.group(1).strip()
                    else:
                        code_snippet = "No se encontró código ejecutable."
                        execution_output = "Error: La IA no generó código válido"

                    # Ejecutar solo si se encontró código
                    if code_match:
                        f = io.StringIO()
                        with contextlib.redirect_stdout(f):
                            try:
                                # Entorno restringido
                                local_vars = {'df': df}
                                exec(code_snippet, SAFE_GLOBALS, local_vars)
                                result = f.getvalue().strip()
                                execution_output = html.escape(result) if result else "Código ejecutado sin salida"
                            except Exception as e:
                                execution_output = f"Error al ejecutar: {html.escape(str(e))}"
                except Exception as e:
                    execution_output = f"Error con Groq API: {html.escape(str(e))}"
            else:
                message = "Sube un archivo CSV primero."

    return render_template(
        "index.html",
        message=message,
        code_snippet=code_snippet,
        response=execution_output,
        df=df.head(5) if df is not None else None  # Solo enviar muestra
    )

if __name__ == "__main__":
    app.run(debug=False)  # Debug desactivado en producción