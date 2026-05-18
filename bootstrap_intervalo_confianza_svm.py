from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MUESTRAS_PATH = BASE_DIR / "muestras.xlsx"
RESULTADOS_PATH = BASE_DIR / "resultados_svm.xlsx"
SALIDA_RESUMEN = BASE_DIR / "intervalo_confianza_bootstrap_svm.xlsx"
SALIDA_GRAFICA = BASE_DIR / "intervalo_confianza_bootstrap_svm.png"

N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def detectar_columna_manual(df: pd.DataFrame) -> str:
    candidatas = ["Categoria_Manual", "Categoria_Mapeada", "Categoria"]
    for columna in candidatas:
        if columna in df.columns:
            return columna
    raise ValueError(
        "No se encontro una columna de categoria manual en la muestra. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def normalizar_texto(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.lower()


def cargar_predicciones() -> pd.DataFrame:
    df_resultados = pd.read_excel(RESULTADOS_PATH)
    columnas_requeridas = {"Consulta", "Prediccion"}
    faltantes = columnas_requeridas.difference(df_resultados.columns)
    if faltantes:
        raise ValueError(
            "resultados_svm.xlsx no contiene las columnas requeridas: "
            f"{sorted(faltantes)}"
        )

    return df_resultados[["Consulta", "Prediccion"]].drop_duplicates(
        subset=["Consulta"], keep="first"
    )


def construir_comparaciones() -> tuple[pd.DataFrame, pd.DataFrame]:
    muestras_dict = pd.read_excel(MUESTRAS_PATH, sheet_name=None)
    predicciones = cargar_predicciones()

    comparaciones = []
    resumen_muestras = []

    for nombre_muestra, df_muestra in muestras_dict.items():
        if "Consulta" not in df_muestra.columns:
            raise ValueError(
                f"La hoja {nombre_muestra} no contiene la columna 'Consulta'."
            )

        columna_manual = detectar_columna_manual(df_muestra)
        comparacion = df_muestra.merge(predicciones, on="Consulta", how="left")
        comparacion["Muestra"] = nombre_muestra
        comparacion["Coincide"] = (
            normalizar_texto(comparacion[columna_manual])
            == normalizar_texto(comparacion["Prediccion"])
        )
        comparacion["Evaluable"] = comparacion["Prediccion"].notna()

        evaluables = comparacion[comparacion["Evaluable"]].copy()
        total = len(comparacion)
        evaluables_n = len(evaluables)
        aciertos = int(evaluables["Coincide"].sum())
        accuracy = aciertos / evaluables_n if evaluables_n else np.nan

        resumen_muestras.append(
            {
                "Muestra": nombre_muestra,
                "Total_registros": total,
                "Registros_evaluables": evaluables_n,
                "Aciertos": aciertos,
                "Accuracy": accuracy,
            }
        )
        comparaciones.append(comparacion)

    return pd.concat(comparaciones, ignore_index=True), pd.DataFrame(resumen_muestras)


def bootstrap_intervalo_binario(
    valores: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict:
    if valores.size == 0:
        raise ValueError("No hay registros evaluables para calcular el bootstrap.")

    rng = np.random.default_rng(seed)
    distribucion = np.empty(n_bootstrap, dtype=float)

    for indice in range(n_bootstrap):
        muestra = rng.choice(valores, size=valores.size, replace=True)
        distribucion[indice] = muestra.mean()

    alpha = 1 - confidence_level
    ic_inferior, ic_superior = np.quantile(distribucion, [alpha / 2, 1 - alpha / 2])

    return {
        "estimacion": float(valores.mean()),
        "ic_inferior": float(ic_inferior),
        "ic_superior": float(ic_superior),
        "distribucion": distribucion,
    }


def bootstrap_media_por_muestra(
    comparaciones_df: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    grupos = []

    for _, grupo in comparaciones_df.groupby("Muestra"):
        evaluables = grupo.loc[grupo["Evaluable"], "Coincide"].astype(int).to_numpy()
        if evaluables.size == 0:
            raise ValueError(
                f"La muestra {grupo['Muestra'].iloc[0]} no tiene registros evaluables."
            )
        grupos.append(evaluables)

    distribucion = np.empty(n_bootstrap, dtype=float)
    for indice in range(n_bootstrap):
        accuracies = []
        for grupo in grupos:
            remuestreo = rng.choice(grupo, size=grupo.size, replace=True)
            accuracies.append(remuestreo.mean())
        distribucion[indice] = float(np.mean(accuracies))

    accuracies_observadas = [grupo.mean() for grupo in grupos]
    alpha = 1 - confidence_level
    ic_inferior, ic_superior = np.quantile(distribucion, [alpha / 2, 1 - alpha / 2])

    return {
        "estimacion": float(np.mean(accuracies_observadas)),
        "ic_inferior": float(ic_inferior),
        "ic_superior": float(ic_superior),
        "distribucion": distribucion,
    }


def guardar_salidas(
    resumen_muestras: pd.DataFrame,
    bootstrap_global: dict,
    bootstrap_promedio: dict,
) -> None:
    resumen_intervalos = pd.DataFrame(
        [
            {
                "Metrica": "Accuracy global agrupada",
                "Estimacion": bootstrap_global["estimacion"],
                "IC_95_inferior": bootstrap_global["ic_inferior"],
                "IC_95_superior": bootstrap_global["ic_superior"],
                "Bootstrap_iteraciones": N_BOOTSTRAP,
            },
            {
                "Metrica": "Promedio de accuracy de las 5 muestras",
                "Estimacion": bootstrap_promedio["estimacion"],
                "IC_95_inferior": bootstrap_promedio["ic_inferior"],
                "IC_95_superior": bootstrap_promedio["ic_superior"],
                "Bootstrap_iteraciones": N_BOOTSTRAP,
            },
        ]
    )

    with pd.ExcelWriter(SALIDA_RESUMEN) as writer:
        resumen_muestras.to_excel(writer, sheet_name="resumen_muestras", index=False)
        resumen_intervalos.to_excel(writer, sheet_name="intervalos_bootstrap", index=False)


def guardar_grafica(bootstrap_promedio: dict) -> None:
    distribucion = bootstrap_promedio["distribucion"] * 100
    estimacion = bootstrap_promedio["estimacion"] * 100
    ic_inferior = bootstrap_promedio["ic_inferior"] * 100
    ic_superior = bootstrap_promedio["ic_superior"] * 100

    plt.figure(figsize=(9, 5))
    plt.hist(distribucion, bins=40, color="#1f77b4", alpha=0.8, edgecolor="white")
    plt.axvline(estimacion, color="black", linestyle="--", linewidth=2, label=f"Media: {estimacion:.2f}%")
    plt.axvline(ic_inferior, color="#d62728", linestyle=":", linewidth=2, label=f"IC 95% inferior: {ic_inferior:.2f}%")
    plt.axvline(ic_superior, color="#2ca02c", linestyle=":", linewidth=2, label=f"IC 95% superior: {ic_superior:.2f}%")
    plt.title("Bootstrap de la accuracy media de las 5 muestras")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA_GRAFICA, dpi=150)
    plt.close()


def main() -> None:
    comparaciones_df, resumen_muestras = construir_comparaciones()
    evaluables_globales = comparaciones_df.loc[
        comparaciones_df["Evaluable"], "Coincide"
    ].astype(int).to_numpy()

    bootstrap_global = bootstrap_intervalo_binario(evaluables_globales)
    bootstrap_promedio = bootstrap_media_por_muestra(comparaciones_df)

    guardar_salidas(resumen_muestras, bootstrap_global, bootstrap_promedio)
    guardar_grafica(bootstrap_promedio)

    print("=" * 72)
    print("INTERVALO DE CONFIANZA BOOTSTRAP - MODELO SVM")
    print("=" * 72)
    print("\nResumen por muestra:")
    print(resumen_muestras.to_string(index=False))

    print("\nBootstrap sobre todos los registros evaluables:")
    print(
        f"Accuracy global: {bootstrap_global['estimacion']:.4f} "
        f"({bootstrap_global['estimacion'] * 100:.2f}%)"
    )
    print(
        f"IC 95%: [{bootstrap_global['ic_inferior']:.4f}, "
        f"{bootstrap_global['ic_superior']:.4f}]"
    )
    print(
        f"IC 95%: [{bootstrap_global['ic_inferior'] * 100:.2f}%, "
        f"{bootstrap_global['ic_superior'] * 100:.2f}%]"
    )

    print("\nBootstrap sobre la media de accuracy de las 5 muestras:")
    print(
        f"Accuracy media: {bootstrap_promedio['estimacion']:.4f} "
        f"({bootstrap_promedio['estimacion'] * 100:.2f}%)"
    )
    print(
        f"IC 95%: [{bootstrap_promedio['ic_inferior']:.4f}, "
        f"{bootstrap_promedio['ic_superior']:.4f}]"
    )
    print(
        f"IC 95%: [{bootstrap_promedio['ic_inferior'] * 100:.2f}%, "
        f"{bootstrap_promedio['ic_superior'] * 100:.2f}%]"
    )

    print(f"\nArchivo Excel generado: {SALIDA_RESUMEN.name}")
    print(f"Grafica generada: {SALIDA_GRAFICA.name}")


if __name__ == "__main__":
    main()