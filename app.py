from flask import Flask, request, render_template
import os
import pandas as pd
import re
import io
import contextlib
from groq import Groq
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Cliente de Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = None  # DataFrame global temporal

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
            if file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                df = pd.read_csv(filepath)
                message = f"Archivo '{file.filename}' cargado correctamente."
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
                            "No es necesario importar pandas ya que ya está importado, la forma de acceder a el es con `pd`."
                            "Cuando llegues a la línea que da el resultado, es necario que la des envuelta en un `print()`, para visualizar su resultado"
                            "Puedes razonar tus pasos antes de dar la solución, pero tu razonamiento debe ir dentro de <think> y </think>. "
                            "El código final que debe ejecutarse debe ir entre:\n"
                            "## CODE ##\n<tu código aquí>\n## END CODE ##\n\n"
                            "No des explicaciones fuera de estas secciones. "
                            "Asegúrate de que el código entre ## CODE ## sea ejecutable por sí solo y produzca la respuesta deseada."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nPregunta: {question}"
                    }
                ]

                completion = client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=prompt,
                    temperature=0,
                    max_completion_tokens=1024,
                    top_p=1.0,
                    stream=False,
                )

                full_output = completion.choices[0].message.content.strip()

                # Extraer bloque entre ## CODE ## y ## END CODE ##
                code_match = re.search(r"## CODE ##\s*(.*?)\s*## END CODE ##", full_output, re.DOTALL)
                code_snippet = code_match.group(1).strip() if code_match else "No se encontró código ejecutable."

                # Ejecutar solo el código extraído
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        exec(code_snippet, {"df": df})
                        execution_output = f.getvalue()
                    except Exception as e:
                        execution_output = f"Error al ejecutar el código: {e}"

            else:
                message = "Sube un archivo CSV primero."

    return render_template(
        "index.html",
        message=message,
        code_snippet=code_snippet,
        response=execution_output,
        df=df
    )

if __name__ == "__main__":
    app.run(debug=True)
