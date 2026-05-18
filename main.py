import pandas as pd
import numpy as np
import re
import nltk
import spacy
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Descargar recursos de NLTK
nltk.download("stopwords")
nltk.download("punkt")

# Cargar modelo de spaCy en español
def cargar_modelo_spacy_es():
    for model_name in ("es_core_news_md", "es_core_news_sm"):
        try:
            print(f"Cargando modelo spaCy: {model_name}")
            return spacy.load(model_name)
        except OSError:
            continue

    print("No se encontro un modelo de spaCy en espanol instalado. Se usara un pipeline basico.")
    return spacy.blank("es")


nlp = cargar_modelo_spacy_es()

print("✅ Librerías cargadas correctamente.")


# ── 2. CARGA DE DATOS ────────────────────────────────────────
df = pd.read_excel("data/Data_arreglada.xlsx", sheet_name="Sheet1")

# Renombrar columnas para facilitar el trabajo
df.columns = ["Categoria", "Resuelta", "Consulta"]

print(f"📦 Shape original: {df.shape}")
print(df.head(5))


# ── 3. LIMPIEZA INICIAL ──────────────────────────────────────
# Eliminar filas donde Consulta o Categoria sean nulas
df = df.dropna(subset=["Consulta", "Categoria"])

# Eliminar filas que no fueron resueltas (Resuelta = 'Sí')
df = df[df["Resuelta"].str.strip().str.lower() != "sí"]
print(f"\n📦 Shape después de eliminar resueltas: {df.shape}")

# Eliminar filas con consulta vacía o solo espacios
df = df[df["Consulta"].str.strip() != ""]
df = df.reset_index(drop=True)

print(f"\n📦 Shape después de eliminar nulos: {df.shape}")

print(f" Shape final del dataset: {df.shape}")
print(df.head(15))


# ── 4. MAPEO DE CATEGORÍAS VÁLIDAS ──────────────────────────
CATEGORIAS_VALIDAS = [
    "Carnetización",
    "Actualización de datos personales",
    "Calendario académico",
    "Certificados",
    "Gestión Académica",
    "Gestión Económica",
    "Reubicación socioeconómica en Pregrado",
    "Aplazamiento de matrícula inicial",
    "Política de gratuidad (matrícula cero) Pregrado",
    "Información general sobre servicios estudiantiles",
]

def mapear_categoria(categoria: str) -> str:
    """
    Mapea cada categoría original del dataset a una de las CATEGORIAS_VALIDAS.
    Usa coincidencia parcial por palabras clave.
    """
    categoria = str(categoria).strip()

    # Mapeos directos y por palabras clave
    mapeo = {
        "Carnetización":                                    ["carnetización", "carnet", "carné"],
        "Actualización de datos personales":                ["actualización de datos", "datos personales", "actualización"],
        "Calendario académico":                             ["calendario académico", "calendario"],
        "Certificados":                                     ["certificado"],
        "Gestión Académica":                                ["gestión académica", "inscripción", "adiciones",
                                                             "cancelaciones", "asignaturas", "sobrecupo",
                                                             "historia académica", "bloqueo", "grado",
                                                             "homologación", "traslado", "aplazamiento",
                                                             "reingreso", "notas", "prueba", "inglés",
                                                             "doble titulación", "posgrado"],
        "Gestión Económica":                                ["gestión económica", "recibo", "pago",
                                                             "fraccionamiento", "unificación", "devolución",
                                                             "financiación", "matrícula", "pbm",
                                                             "descuento", "electoral", "generación e",
                                                             "icetex", "ser pilo", "exención",
                                                             "reexpedición", "cobro", "deuda"],
        "Reubicación socioeconómica en Pregrado":           ["reubicación socioeconómica", "reubicación",
                                                             "socioeconómica", "socioeconómico"],
        "Aplazamiento de matrícula inicial":                ["aplazamiento de matrícula", "aplazamiento inicial",
                                                             "aplazamiento"],
        "Política de gratuidad (matrícula cero) Pregrado": ["matrícula cero", "gratuidad", "matrícula 0",
                                                             "política de gratuidad"],
        "Información general sobre servicios estudiantiles":["información general", "información financiera",
                                                              "servicios estudiantiles", "bienestar",
                                                              "alimentaria", "correo institucional",
                                                              "sia", "bicirún", "sibu"],
    }

    categoria_lower = categoria.lower()

    for cat_valida, palabras_clave in mapeo.items():
        for palabra in palabras_clave:
            if palabra in categoria_lower:
                return cat_valida

    # Si no hay coincidencia, intentar por la consulta (fallback)
    return "Información general sobre servicios estudiantiles"


# Aplicar el mapeo
df["Categoria_Mapeada"] = df["Categoria"].apply(mapear_categoria)

print("\n📊 Distribución de categorías mapeadas:")
print(df["Categoria_Mapeada"].value_counts())


# ── 5. PREPROCESAMIENTO DE TEXTO ─────────────────────────────
STOPWORDS_ES = set(stopwords.words("spanish"))

# Stopwords adicionales específicas del dominio universitario
STOPWORDS_EXTRA = {
    "universidad", "nacional", "colombia", "unal", "sede", "bogotá",
    "favor", "gracias", "buenas", "buenos", "días", "tardes", "noches",
    "cordial", "saludo", "atentamente", "amablemente", "presente",
    "correo", "motivo", "solicito", "solicitud", "quisiera", "quiero",
    "necesito", "requiero", "agradezco", "agradecería", "muchas",
    "manera", "forma", "caso", "parte", "vez", "día", "semestre",
    "periodo", "académico", "estudiante", "programa", "curricular",
    "sia", "dninfoa", "portal", "plataforma", "sistema",
}

STOPWORDS_COMPLETO = STOPWORDS_ES.union(STOPWORDS_EXTRA)


def limpiar_texto(texto: str) -> str:
    """Limpieza básica: minúsculas, eliminar caracteres especiales y números."""
    texto = str(texto).lower()
    # Eliminar URLs
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    # Eliminar correos electrónicos
    texto = re.sub(r"\S+@\S+", " ", texto)
    # Eliminar números y caracteres especiales, conservar letras y espacios
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def eliminar_stopwords(texto: str) -> str:
    """Elimina stopwords del texto."""
    tokens = texto.split()
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS_COMPLETO and len(t) > 2]
    return " ".join(tokens_filtrados)


def lematizar(texto: str) -> str:
    """Lematiza el texto usando spaCy."""
    doc = nlp(texto)
    lemas = [
        token.lemma_ if token.lemma_ else token.text
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_ if token.lemma_ else token.text) > 2
    ]
    return " ".join(lemas)


def preprocesar(texto: str) -> str:
    """Pipeline completo: limpieza → stopwords → lematización."""
    texto = limpiar_texto(texto)
    texto = eliminar_stopwords(texto)
    texto = lematizar(texto)
    return texto


print("\n⚙️  Aplicando preprocesamiento (puede tardar unos minutos)...")
df["Consulta_Procesada"] = df["Consulta"].apply(preprocesar)

print("✅ Preprocesamiento completado.")
print("\nEjemplo de transformación:")
print(f"  ORIGINAL : {df['Consulta'].iloc[0][:100]}...")
print(f"  PROCESADO: {df['Consulta_Procesada'].iloc[0][:100]}...")


# ── 6. PREPARACIÓN DE FEATURES Y ETIQUETAS ──────────────────
# Eliminar filas con texto procesado vacío
df = df[df["Consulta_Procesada"].str.strip() != ""]
df = df.reset_index(drop=True)

X = df["Consulta_Procesada"]
y = df["Categoria_Mapeada"]

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📦 Total de muestras para entrenamiento: {len(X)}")
print(f"🏷️  Clases: {list(le.classes_)}")


# ── 7. VECTORIZACIÓN TF-IDF ──────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=5000,       # Máximo 5000 términos
    ngram_range=(1, 2),      # Unigramas y bigramas
    min_df=2,                # Mínimo 2 documentos
    sublinear_tf=True,       # Escala logarítmica
)

X_tfidf = tfidf.fit_transform(X)
print(f"\n🔢 Matriz TF-IDF: {X_tfidf.shape}")


# ── 8. DIVISIÓN TRAIN / TEST ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print(f"\n📊 Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")


# ── 9. ENTRENAMIENTO DEL MODELO ──────────────────────────────
modelo = LogisticRegression(
    max_iter=1000,
    C=1.0,                   # Regularización
    solver="lbfgs",
    random_state=42,
    class_weight="balanced", # Maneja desbalance de clases
)

modelo.fit(X_train, y_train)
print("\n✅ Modelo entrenado correctamente.")


# ── 10. EVALUACIÓN ───────────────────────────────────────────
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Reporte de Clasificación:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0,
))

print("\n🔲 Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)


# ── 11. FUNCIÓN DE PREDICCIÓN ────────────────────────────────
def predecir_categoria(texto_nuevo: str) -> dict:
    """
    Predice la categoría de una nueva consulta.
    Retorna la categoría predicha y las probabilidades por clase.
    """
    texto_proc = preprocesar(texto_nuevo)
    texto_vec  = tfidf.transform([texto_proc])
    pred_encoded   = modelo.predict(texto_vec)[0]
    pred_categoria = le.inverse_transform([pred_encoded])[0]
    probabilidades = modelo.predict_proba(texto_vec)[0]

    probs_dict = {
        clase: round(float(prob), 4)
        for clase, prob in zip(le.classes_, probabilidades)
    }
    probs_ordenadas = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))

    return {
        "categoria_predicha": pred_categoria,
        "probabilidades": probs_ordenadas,
    }


# ── 12. PRUEBA CON EJEMPLOS ──────────────────────────────────
ejemplos = [
    "No me aparece el recibo de pago en el SIA y necesito pagarlo urgente",
    "Quisiera saber cómo inscribir materias para este semestre",
    "Necesito tramitar mi carné universitario",
    "Soy estrato 2 y quiero saber si aplico para matrícula cero",
    "Solicito certificado de notas para trámite externo",
    "Quiero aplazar mi matrícula por motivos económicos",
]

print("\n🧪 PRUEBAS DE PREDICCIÓN:")
print("=" * 60)
for ejemplo in ejemplos:
    resultado = predecir_categoria(ejemplo)
    print(f"\n📝 Consulta : {ejemplo}")
    print(f"🏷️  Predicción: {resultado['categoria_predicha']}")
    top3 = list(resultado["probabilidades"].items())[:3]
    print(f"📊 Top-3 probs: {top3}")
print("=" * 60)

# ── 13. GUARDAR RESULTADOS ───────────────────────────────────
df_resultado = df[["Consulta", "Categoria", "Categoria_Mapeada", "Consulta_Procesada"]].copy()
df_resultado["Prediccion"] = le.inverse_transform(modelo.predict(X_tfidf))
df_resultado.to_excel("resultados_modelo.xlsx", index=False)
print("\n💾 Resultados guardados en: resultados_modelo.xlsx")
