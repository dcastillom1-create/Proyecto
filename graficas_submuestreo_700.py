import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RUTA_DATOS = Path("data/Data_arreglada.xlsx")
DIR_SALIDA = Path("salidas_submuestreo_700")
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

def cargar_datos():
    df = pd.read_excel(RUTA_DATOS, sheet_name="Sheet1")
    df = df.dropna(subset=["Categoria"]).reset_index(drop=True)
    return df

def submuestrear(df, n=700, random_state=42):
    if len(df) < n:
        print(f"Datos disponibles ({len(df)}) < solicitados ({n}). Usando todos.")
        return df
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

def grafica_distribucion_comparativa(df_original, df_submueestra, ruta: Path):
    dist_orig = df_original["Categoria"].value_counts().sort_values(ascending=True)
    dist_sub = df_submueestra["Categoria"].value_counts().sort_values(ascending=True)
    
    todas_cats = sorted(set(dist_orig.index) | set(dist_sub.index))
    orig_vals = [dist_orig.get(c, 0) for c in todas_cats]
    sub_vals = [dist_sub.get(c, 0) for c in todas_cats]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(todas_cats))
    width = 0.35
    
    ax.barh(x - width/2, orig_vals, width, label=f"Original (n={len(df_original)})", color="#1f77b4")
    ax.barh(x + width/2, sub_vals, width, label=f"Submuestreo (n={len(df_submueestra)})", color="#ff7f0e")
    
    ax.set_yticks(x)
    ax.set_yticklabels(todas_cats, fontsize=9)
    ax.set_xlabel("Frecuencia")
    ax.set_title("Distribución de categorías: Original vs Submuestreo (700)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)

def grafica_proporcion_representatividad(df_original, df_submueestra, ruta: Path):
    dist_orig = df_original["Categoria"].value_counts()
    dist_sub = df_submueestra["Categoria"].value_counts()
    
    todas_cats = sorted(set(dist_orig.index) | set(dist_sub.index))
    
    prop_orig = [dist_orig.get(c, 0) / len(df_original) for c in todas_cats]
    prop_sub = [dist_sub.get(c, 0) / len(df_submueestra) for c in todas_cats]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(todas_cats))
    width = 0.35
    
    ax.barh(x - width/2, prop_orig, width, label="Original (proporción)", color="#2ca02c")
    ax.barh(x + width/2, prop_sub, width, label="Submuestreo (proporción)", color="#d62728")
    
    ax.set_yticks(x)
    ax.set_yticklabels(todas_cats, fontsize=9)
    ax.set_xlabel("Proporción")
    ax.set_xlim(0, max(prop_orig + prop_sub) * 1.1)
    ax.set_title("Proporción de categorías: Original vs Submuestreo (700)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)

def grafica_balance_de_clases(df_submueestra, ruta: Path):
    dist = df_submueestra["Categoria"].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colores = plt.cm.tab20c(np.linspace(0, 1, len(dist)))
    ax.barh(range(len(dist)), dist.values, color=colores)
    ax.set_yticks(range(len(dist)))
    ax.set_yticklabels(dist.index, fontsize=9)
    ax.set_xlabel("Número de registros")
    ax.set_title(f"Balance de clases en submuestreo (n={len(df_submueestra)})")
    
    for i, v in enumerate(dist.values):
        ax.text(v + 1, i, str(v), va="center", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)

def grafica_estabilidad_submuestreo(df_original, ruta: Path, n_submuestras=5, tamaño_sub=700):
    distribuciones = []
    for i in range(n_submuestras):
        sub = df_original.sample(n=min(tamaño_sub, len(df_original)), random_state=42+i)
        dist = sub["Categoria"].value_counts(normalize=True).sort_index()
        distribuciones.append(dist)
    
    df_dist = pd.DataFrame(distribuciones).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    df_dist.T.plot(kind="barh", ax=ax, width=0.8, color=plt.cm.Set3(np.linspace(0, 1, n_submuestras)))
    ax.set_xlabel("Proporción")
    ax.set_ylabel("Categoría")
    ax.set_title(f"Estabilidad de proporción en {n_submuestras} submuestras aleatorias (n={tamaño_sub})")
    ax.legend(title="Submueestra", labels=[f"Seed {42+i}" for i in range(n_submuestras)], fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)

def grafica_tamaño_por_categoria(df_submueestra, ruta: Path):
    dist = df_submueestra["Categoria"].value_counts().sort_values(ascending=False).head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfica de barras
    ax1.bar(range(len(dist)), dist.values, color="#9467bd")
    ax1.set_xticks(range(len(dist)))
    ax1.set_xticklabels(dist.index, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Cantidad")
    ax1.set_title("Top 15 categorías (submuestreo 700)")
    ax1.grid(axis="y", alpha=0.3)
    
    # Gráfica de pastel
    otros = df_submueestra["Categoria"].value_counts().iloc[15:].sum()
    labels = list(dist.index) + (["Otros"] if otros > 0 else [])
    sizes = list(dist.values) + ([otros] if otros > 0 else [])
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax2.set_title("Distribución de categorías (submuestreo 700)")
    
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)

def resumen_estadistico(df_original, df_submueestra):
    print("\n" + "="*70)
    print("RESUMEN DE SUBMUESTREO (700 registros)")
    print("="*70)
    print(f"Datos originales: {len(df_original)} registros")
    print(f"Submuestreo: {len(df_submueestra)} registros")
    print(f"Reducción: {(1 - len(df_submueestra)/len(df_original))*100:.1f}%")
    print(f"\nCategorías en original: {df_original['Categoria'].nunique()}")
    print(f"Categorías en submuestreo: {df_submueestra['Categoria'].nunique()}")
    print(f"\nCategoría más frecuente (original): {df_original['Categoria'].value_counts().index[0]} ({df_original['Categoria'].value_counts().values[0]})")
    print(f"Categoría más frecuente (submuestreo): {df_submueestra['Categoria'].value_counts().index[0]} ({df_submueestra['Categoria'].value_counts().values[0]})")
    print(f"\nCategoría menos frecuente (submuestreo): {df_submueestra['Categoria'].value_counts().index[-1]} ({df_submueestra['Categoria'].value_counts().values[-1]})")
    print("="*70 + "\n")

def main():
    df_original = cargar_datos()
    df_submueestra = submuestrear(df_original, n=700)
    
    resumen_estadistico(df_original, df_submueestra)
    
    print("Generando gráficas...")
    grafica_distribucion_comparativa(df_original, df_submueestra, DIR_SALIDA / "01_distribucion_original_vs_submuestreo.png")
    print("✓ Distribución comparativa")
    
    grafica_proporcion_representatividad(df_original, df_submueestra, DIR_SALIDA / "02_proporcion_representatividad.png")
    print("✓ Proporciones")
    
    grafica_balance_de_clases(df_submueestra, DIR_SALIDA / "03_balance_clases_submuestreo.png")
    print("✓ Balance de clases")
    
    grafica_estabilidad_submuestreo(df_original, DIR_SALIDA / "04_estabilidad_submuestras.png", n_submuestras=5, tamaño_sub=700)
    print("✓ Estabilidad (5 submuestras)")
    
    grafica_tamaño_por_categoria(df_submueestra, DIR_SALIDA / "05_top_categorias_pastel.png")
    print("✓ Top categorías")
    
    print(f"\nTodas las gráficas guardadas en: {DIR_SALIDA.resolve()}")

if __name__ == "__main__":
    main()
