from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RUTA_EXCEL = Path("muestras_aprendizaje activo.xlsx")
DIR_SALIDA = Path("salidas_aprendizaje_activo_4_1")


def limpiar_texto(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto).lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^a-zA-Z0-9áéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_categoria(texto: str) -> str:
    if pd.isna(texto):
        return ""
    txt = str(texto).strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.lower()
    return txt


def cargar_datos(ruta_excel: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hojas_train = ["Muestra_1", "Muestra_2", "Muestra_3", "Muestra_4"]
    hoja_test = "Muestra_5"

    entrenamientos = []
    for hoja in hojas_train:
        df = pd.read_excel(ruta_excel, sheet_name=hoja)
        df["Origen_Muestra"] = hoja
        entrenamientos.append(df)

    train_df = pd.concat(entrenamientos, ignore_index=True)
    test_df = pd.read_excel(ruta_excel, sheet_name=hoja_test)
    test_df["Origen_Muestra"] = hoja_test

    return train_df, test_df


def preparar(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    columnas_necesarias = ["Consulta"]
    for col in columnas_necesarias:
        if col not in train_df.columns:
            raise ValueError(f"Falta la columna '{col}' en entrenamiento")

    if "Consulta" not in test_df.columns:
        raise ValueError("Falta la columna 'Consulta' en prueba")
    if "Categoria_Manual" not in test_df.columns:
        raise ValueError("Falta la columna 'Categoria_Manual' en prueba")

    columna_objetivo = "Categoria_Manual" if "Categoria_Manual" in train_df.columns else "Categoria"
    if columna_objetivo not in train_df.columns:
        raise ValueError("No se encontro columna objetivo en entrenamiento")

    train_df = train_df.dropna(subset=["Consulta", columna_objetivo]).copy()
    test_df = test_df.dropna(subset=["Consulta", "Categoria_Manual"]).copy()

    train_df["texto_proc"] = train_df["Consulta"].apply(limpiar_texto)
    test_df["texto_proc"] = test_df["Consulta"].apply(limpiar_texto)

    train_df[columna_objetivo] = train_df[columna_objetivo].apply(normalizar_categoria)
    test_df["Categoria_Manual"] = test_df["Categoria_Manual"].apply(normalizar_categoria)

    train_df = train_df[train_df["texto_proc"] != ""].reset_index(drop=True)
    test_df = test_df[test_df["texto_proc"] != ""].reset_index(drop=True)

    return train_df, test_df, columna_objetivo


def entrenar_modelo(cv_calibracion: int, usar_calibracion: bool) -> Pipeline:
    svm_base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    pasos = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        )
    ]

    if usar_calibracion:
        pasos.append(
            (
                "clf",
                CalibratedClassifierCV(svm_base, cv=cv_calibracion, method="sigmoid"),
            )
        )
    else:
        pasos.append(("clf", svm_base))

    modelo = Pipeline(pasos)
    return modelo


def graficar_matriz_confusion(y_true, y_pred, labels, ruta: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusion - Muestra 5")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Categoria manual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def graficar_confianza(confianza, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confianza, bins=20, color="#1f77b4", edgecolor="black")
    ax.set_title("Distribucion de confianza de prediccion")
    ax.set_xlabel("Confianza maxima")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def graficar_f1_por_clase(report_df: pd.DataFrame, ruta: Path) -> None:
    clases = report_df.loc[
        ~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    clases = clases.sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(clases.index, clases["f1-score"], color="#2ca02c")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Categoria")
    ax.set_title("F1-score por categoria - Muestra 5")
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def main() -> None:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    train_df, test_df = cargar_datos(RUTA_EXCEL)
    train_df, test_df, columna_objetivo = preparar(train_df, test_df)

    min_por_clase = int(train_df[columna_objetivo].value_counts().min())
    usar_calibracion = min_por_clase >= 2
    cv_calibracion = max(2, min(3, min_por_clase)) if usar_calibracion else 2

    modelo = entrenar_modelo(
        cv_calibracion=cv_calibracion,
        usar_calibracion=usar_calibracion,
    )
    modelo.fit(train_df["texto_proc"], train_df[columna_objetivo])

    y_true = test_df["Categoria_Manual"].astype(str)
    y_pred = pd.Series(modelo.predict(test_df["texto_proc"]), index=test_df.index).apply(
        normalizar_categoria
    )
    if usar_calibracion:
        y_proba = modelo.predict_proba(test_df["texto_proc"])
        confianza = y_proba.max(axis=1)
    else:
        decision = modelo.decision_function(test_df["texto_proc"])
        if decision.ndim == 1:
            conf = abs(decision)
            conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
            confianza = conf
        else:
            margenes = decision.max(axis=1)
            margenes = (margenes - margenes.min()) / (
                margenes.max() - margenes.min() + 1e-9
            )
            confianza = margenes

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    reporte = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    reporte_df = pd.DataFrame(reporte).transpose()

    resultados = test_df[["Consulta", "Categoria_Manual"]].copy()
    resultados["Prediccion_modelo"] = y_pred
    resultados["Confianza_modelo"] = confianza
    resultados["Coincide"] = resultados["Categoria_Manual"] == resultados["Prediccion_modelo"]

    resumen_metricas = pd.DataFrame(
        [
            {"Metrica": "accuracy", "Valor": accuracy},
            {"Metrica": "f1_macro", "Valor": f1_macro},
            {"Metrica": "f1_weighted", "Valor": f1_weighted},
            {"Metrica": "train_size", "Valor": len(train_df)},
            {"Metrica": "test_size", "Valor": len(test_df)},
        ]
    )

    ruta_excel = DIR_SALIDA / "resultados_muestra5_aprendizaje_activo.xlsx"
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        resultados.to_excel(writer, sheet_name="predicciones", index=False)
        resumen_metricas.to_excel(writer, sheet_name="metricas", index=False)
        reporte_df.to_excel(writer, sheet_name="classification_report", index=True)

    etiquetas = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    graficar_matriz_confusion(y_true, y_pred, etiquetas, DIR_SALIDA / "matriz_confusion.png")
    graficar_confianza(confianza, DIR_SALIDA / "hist_confianza.png")
    graficar_f1_por_clase(reporte_df, DIR_SALIDA / "f1_por_categoria.png")

    print("Entrenamiento con Muestra_1..4 completado.")
    print(f"Columna objetivo usada: {columna_objetivo}")
    print(f"Calibracion habilitada: {usar_calibracion}")
    print(f"CV de calibracion usado: {cv_calibracion if usar_calibracion else 'N/A'}")
    print(f"Filas entrenamiento: {len(train_df)}")
    print(f"Filas prueba (Muestra_5): {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print(f"Resultados guardados en: {ruta_excel}")
    print(f"Graficas guardadas en: {DIR_SALIDA.resolve()}")


if __name__ == "__main__":
    main()
