import pandas as pd
import unicodedata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import joblib

from funciones import limpiar_stopwords_lematizar


RUTA_MUESTRAS = "muestras_aprendizaje activo.xlsx"
RUTA_DATA_ARREGLADA = "data/Data_arreglada.xlsx"
RUTA_PREDICCION = "prediccion_150_aprendizaje_activo.xlsx"
RUTA_METRICAS = "metricas_aprendizaje_activo_750.xlsx"
RUTA_MODELO = "svm_model_aprendizaje_activo_750.pkl"
RUTA_ENCODER = "label_encoder_aprendizaje_activo_750.pkl"

CATEGORIA_EQUIVALENCIAS = {
    "gestion economica": "Gestión Económica",
    "gestion académica": "Gestión Académica",
    "información general": "Información general sobre servicios estudiantiles",
}


def _normalizar_texto(s: str) -> str:
    t = str(s).strip().lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t


def cargar_muestras(ruta_excel: str) -> pd.DataFrame:
    xls = pd.ExcelFile(ruta_excel)
    hojas = xls.sheet_names

    frames = []
    for hoja in hojas:
        df = pd.read_excel(ruta_excel, sheet_name=hoja)
        if "Consulta" not in df.columns:
            continue

        col_categoria = "Categoria_Manual" if "Categoria_Manual" in df.columns else "Categoria"
        if col_categoria not in df.columns:
            continue

        tmp = df[["Consulta", col_categoria]].copy()
        tmp = tmp.rename(columns={col_categoria: "Categoria"})
        tmp["Hoja"] = hoja
        frames.append(tmp)

    if not frames:
        raise ValueError("No se encontraron hojas con columnas requeridas en muestras_aprendizaje activo.xlsx")

    datos = pd.concat(frames, ignore_index=True)
    datos = datos.dropna(subset=["Consulta", "Categoria"])
    datos["Consulta"] = datos["Consulta"].astype(str).str.strip()
    datos["Categoria"] = datos["Categoria"].astype(str).str.strip()
    datos["Categoria"] = datos["Categoria"].apply(
        lambda c: CATEGORIA_EQUIVALENCIAS.get(_normalizar_texto(c), c)
    )
    datos = datos[(datos["Consulta"] != "") & (datos["Categoria"] != "")]
    datos = datos.reset_index(drop=True)
    return datos


def entrenar_modelo(df_train: pd.DataFrame):
    df_train = df_train.copy()
    df_train["Consulta_procesada"] = df_train["Consulta"].apply(limpiar_stopwords_lematizar)
    df_train = df_train[df_train["Consulta_procesada"].str.strip() != ""].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df_train["Categoria"])
    X = df_train["Consulta_procesada"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    svm_base = LinearSVC(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        dual=True,
    )

    modelo = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=7000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("svm", CalibratedClassifierCV(svm_base, cv=2, method="sigmoid")),
        ]
    )

    print("Entrenando modelo SVM calibrado con 750 registros...")
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = list(range(len(le.classes_)))
    reporte = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    metricas = pd.DataFrame(reporte).transpose()
    metricas.to_excel(RUTA_METRICAS, index=True)

    print(f"Accuracy de validacion interna: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Metricas guardadas en: {RUTA_METRICAS}")

    # Reentrenar con todo el set de 750 para usarlo en la prediccion de nuevos datos.
    modelo.fit(X, y)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(le, RUTA_ENCODER)
    print(f"Modelo guardado en: {RUTA_MODELO}")
    print(f"Encoder guardado en: {RUTA_ENCODER}")

    return modelo, le, df_train


def predecir_150(modelo, le: LabelEncoder, df_train_limpio: pd.DataFrame):
    df_pool = pd.read_excel(RUTA_DATA_ARREGLADA, sheet_name="Sheet1")
    if "Consulta" not in df_pool.columns:
        raise ValueError("No se encontro la columna 'Consulta' en data/Data_arreglada.xlsx")

    df_pool = df_pool.copy()
    df_pool = df_pool.dropna(subset=["Consulta"])
    df_pool["Consulta"] = df_pool["Consulta"].astype(str).str.strip()
    df_pool = df_pool[df_pool["Consulta"] != ""].reset_index(drop=True)

    # Evita evaluar sobre textos usados en entrenamiento.
    consultas_train = set(df_train_limpio["Consulta"].astype(str).str.strip().str.lower())
    df_pool = df_pool[~df_pool["Consulta"].str.lower().isin(consultas_train)].reset_index(drop=True)

    df_pool["Consulta_procesada"] = df_pool["Consulta"].apply(limpiar_stopwords_lematizar)
    df_pool = df_pool[df_pool["Consulta_procesada"].str.strip() != ""].reset_index(drop=True)

    n = min(150, len(df_pool))
    if n == 0:
        raise ValueError("No hay datos disponibles en Data_arreglada para tomar 150 muestras.")

    df_eval = df_pool.sample(n=n, random_state=42).reset_index(drop=True)

    probs = modelo.predict_proba(df_eval["Consulta_procesada"])
    pred_cod = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred = le.inverse_transform(pred_cod)

    salida = pd.DataFrame(
        {
            "Consulta": df_eval["Consulta"],
            "Categoria_origen": df_eval["Categoria"] if "Categoria" in df_eval.columns else "",
            "Prediccion_modelo": pred,
            "Confianza_modelo": conf,
        }
    )

    salida.to_excel(RUTA_PREDICCION, index=False)
    print(f"Predicciones guardadas en: {RUTA_PREDICCION}")
    print("Top 10 predicciones:")
    print(salida.head(10).to_string(index=False))


def main():
    df_muestras = cargar_muestras(RUTA_MUESTRAS)
    print(f"Registros de entrenamiento cargados: {len(df_muestras)}")
    print(df_muestras["Categoria"].value_counts().to_string())

    modelo, le, df_train_limpio = entrenar_modelo(df_muestras)
    predecir_150(modelo, le, df_train_limpio)


if __name__ == "__main__":
    main()
