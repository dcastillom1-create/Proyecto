import pandas as pd
import re
import nltk
import spacy
import unicodedata
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

STOPWORDS_ES = set(stopwords.words("spanish"))
_NLP_ES = None


def _obtener_nlp_es():
    global _NLP_ES
    if _NLP_ES is None:
        try:
            _NLP_ES = spacy.load("es_core_news_md")
        except OSError:
            _NLP_ES = spacy.load("es_core_news_sm")
    return _NLP_ES


def limpiar_stopwords_lematizar(texto: str) -> str:
    """
    Limpia texto, elimina stopwords en espanol y lematiza con spaCy.

    Args:
        texto: texto de entrada.

    Returns:
        Texto normalizado, sin stopwords y lematizado.
    """
    if texto is None or pd.isna(texto):
        return ""

    texto_limpio = str(texto).lower().strip()
    texto_limpio = re.sub(r"https?://\S+|www\.\S+", " ", texto_limpio)
    texto_limpio = re.sub(r"[^a-zA-Záéíóúüñ\s]", " ", texto_limpio)
    texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

    if not texto_limpio:
        return ""

    tokens = word_tokenize(texto_limpio, language="spanish")
    tokens_filtrados = [
        tok for tok in tokens if tok not in STOPWORDS_ES and len(tok) > 2
    ]

    if not tokens_filtrados:
        return ""

    doc = _obtener_nlp_es()(" ".join(tokens_filtrados))
    lemas = [
        tok.lemma_.lower().strip()
        for tok in doc
        if tok.lemma_.strip() and tok.lemma_.lower().strip() not in STOPWORDS_ES and len(tok.lemma_) > 2
    ]

    return " ".join(lemas)

# ── Categorías válidas (extraídas del CSV de respuestas) ────────────────
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
CATEGORIA_POR_DEFECTO = "Información general sobre servicios estudiantiles"
# ── Mapeo de variantes del Excel → categorías estándar ──────────────────
MAPEO_CATEGORIAS = {
    # Espacios invisibles al inicio (carácter ​)
    "​ Gestión Económica - Inconsistencias sobre recibo":
        "Gestión Económica",
    "​ Gestión Económica - Unificación de recibos":
        "Gestión Económica",
    # Sin prefijo especial
    "Gestión Económica - Inconsistencias sobre recibo":
        "Gestión Económica",
    "Gestión Económica - Unificación de recibos":
        "Gestión Económica",
    # Variantes de información general
    "Gestión Económica - Información general":
        "Gestión Económica",
    # Variantes de gestión económica
    "Gestión Económica - Eliminación financiación SPP4":
        "Gestión Económica",
    "Gestión Económica - Devolución por matricula cero":
        "Gestión Económica",
    "Información financiera":
        "Gestión Económica",
    "Fraccionamiento del recibo":
        "Gestión Económica",
    "Gestión económica reexpedición recibo vencido":
        "Gestión Económica",
    "Gestión económica- Validacion de mi estado socieconomico":
        "Gestión Económica",
    "Unificación de los recibos de pago del periodo 2022-2S para proceso de actualización de datos generación E.":
        "Gestión Económica",
    # Gestión Académica
    "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Inscripcion de asignaturas":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Cupos de asignaturas":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Sobre cupo en calculo diferencial":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Error de la información en aplicativo para Solicitud de cupos":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Clase faltante por fallo en el sia":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "Bloqueo de historia académica":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    "bloqueo b-38":
        "Gestión Académica - Inscripción, adiciones y cancelaciones de asignaturas",
    # Reubicación
    "Reubicación socioeconómica":
        "Reubicación socioeconómica en Pregrado",
    "Reubicacion socioeconomica":
        "Reubicación socioeconómica en Pregrado",
    # Matrícula cero
    "Gestión de la matricula cero":
        "Política de gratuidad (matrícula cero) Pregrado",
    "Devolución de dinero por matrícula cero correspondiente a los anteriores periodos académicos":
        "Política de gratuidad (matrícula cero) Pregrado",
    "Devolución dinero pagado en la matrícula del semestre 2021-1":
        "Política de gratuidad (matrícula cero) Pregrado",
    # Aplazamiento
    "Aplazamiento al uso de derecho de matrícula":
        "Aplazamiento de matrícula inicial",
    "Aplazamiento de matrícula inicial":
        "Aplazamiento de matrícula inicial",
    "pausa de un semestre":
        "Aplazamiento de matrícula inicial",
    # Carnetización
    "Carnetización": "Carnetización",
    # Calendario
    "Calendario académico": "Calendario académico",
    # Certificados
    "Certificados": "Certificados",
    # Datos personales
    "🧑‍ Actualización de datos personales":
        "Actualización de datos personales",
    "Actualización Historia Académica":
        "Actualización de datos personales",
    # Información general sobre servicios estudiantiles
    "Información general":
        "Información general sobre servicios estudiantiles",      
}

def _normalizar_texto_categoria(texto: str) -> str:
    if texto is None:
        return ""
    s = str(texto).replace("\u200b", " ").strip().lower()   # quita zero-width
    s = re.sub(r"\s+", " ", s)                              # colapsa espacios
    s = unicodedata.normalize("NFD", s)                     # separa tildes
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quita tildes
    return s

# Diccionario de mapeo normalizado (clave sin tildes/mayúsculas)
MAPEO_CATEGORIAS_NORM = {
    _normalizar_texto_categoria(k): v for k, v in MAPEO_CATEGORIAS.items()
}

def cargar_datos(ruta_xlsx: str | None = None) -> pd.DataFrame:
    """
    Lee el Excel, normaliza categorías y filtra
    solo las filas con categorías válidas y consulta no vacía.

    Returns:
        pd.DataFrame con columnas ['Consulta', 'Categoria_norm']
    """
    ruta = ruta_xlsx or os.getenv("DATA_XLSX", "data/Data_arreglada.xlsx")

    print(f"📂 Leyendo archivo: {ruta}")
    df = pd.read_excel(ruta, sheet_name="Sheet1")
    df.columns = ["Categoria", "Resuelta", "Consulta"]

    # ── Limpiar texto ────────────────────────────────────────────────────
    df = df.dropna(subset=["Consulta"])
    df["Consulta"] = df["Consulta"].astype(str).str.strip()
    df = df[df["Consulta"].str.len() > 30]

    # ── Normalizar categorías ────────────────────────────────────────────
    df["Categoria"] = df["Categoria"].astype(str)
    clave_norm = df["Categoria"].map(_normalizar_texto_categoria)
    df["Categoria_norm"] = clave_norm.map(MAPEO_CATEGORIAS_NORM)

    # Si no se pudo mapear, conserva original para validar después
    df["Categoria_norm"] = df["Categoria_norm"].fillna(df["Categoria"].str.strip())

    # Si no está en categorías válidas, enviar a información general
    mask_invalidas = ~df["Categoria_norm"].isin(CATEGORIAS_VALIDAS)
    df.loc[mask_invalidas, "Categoria_norm"] = CATEGORIA_POR_DEFECTO

    # ── Filtrar solo categorías válidas ──────────────────────────────────
    df_filtrado = df[df["Categoria_norm"].isin(CATEGORIAS_VALIDAS)].copy()
    df_filtrado = df_filtrado[["Consulta", "Categoria_norm"]].reset_index(drop=True) 
    
    print(df_filtrado.head(30))
    

    print(f"✅ Registros válidos cargados: {len(df_filtrado)}")
    print("\n📊 Distribución por categoría:")
    print(df_filtrado["Categoria_norm"].value_counts().to_string())
    print()

    return df_filtrado