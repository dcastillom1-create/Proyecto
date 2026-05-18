import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SALIDA_GRAFICA = BASE_DIR / "comparacion_metricas_modelos.png"

# Datos de métricas por modelo
modelos = ['Random Forest', 'Regresión\nLogística', 'Naive\nBayes', 'SVM\n(LinearSVC)', 'Redes\nNeuronales']
accuracy = [0.694, 0.732, 0.749, 0.765, 0.682]
recall = [0.690, 0.730, 0.740, 0.770, 0.680]
f1_score = [0.700, 0.740, 0.730, 0.750, 0.690]

# Configuración de la gráfica
fig, ax = plt.subplots(figsize=(12, 6))

# Posiciones de las barras
x = np.arange(len(modelos))
ancho_barra = 0.25

# Crear barras
barras1 = ax.bar(x - ancho_barra, accuracy, ancho_barra, label='Accuracy', color='#2C5282', alpha=0.9)
barras2 = ax.bar(x, recall, ancho_barra, label='Recall', color='#4A90E2', alpha=0.9)
barras3 = ax.bar(x + ancho_barra, f1_score, ancho_barra, label='F1-Score', color='#E85D75', alpha=0.9)

# Línea punteada amarilla (baseline SVM Accuracy)
ax.axhline(y=0.765, color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.8)

# Etiquetas en las barras
def agregar_etiquetas(barras):
    for barra in barras:
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura,
                f'{altura:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

agregar_etiquetas(barras1)
agregar_etiquetas(barras2)
agregar_etiquetas(barras3)

# Configuración de etiquetas y título
ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor de la Métrica', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Métricas por Modelo de Clasificación\n(Muestreo completo: 36,468 registros)', 
             fontsize=13, fontweight='bold', pad=20)

# Ubicación de etiquetas en el eje X
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=11)

# Rango del eje Y
ax.set_ylim(0.600, 0.800)

# Leyenda
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# Grid para mejor lectura
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Estilos finales
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(SALIDA_GRAFICA, dpi=300, bbox_inches='tight')
print(f"✅ Gráfica guardada en: {SALIDA_GRAFICA}")
plt.close()


def main() -> None:
    """
    Genera la gráfica comparativa de métricas de modelos de clasificación
    """
    print("=" * 70)
    print("COMPARACIÓN DE MÉTRICAS POR MODELO DE CLASIFICACIÓN")
    print("=" * 70)
    print("\nResumen de Métricas:")
    print("-" * 70)
    
    for i, modelo in enumerate(modelos):
        modelo_limpio = modelo.replace('\n', ' ')
        print(f"{modelo_limpio:25} | Accuracy: {accuracy[i]:.4f} | Recall: {recall[i]:.4f} | F1-Score: {f1_score[i]:.4f}")
    
    print("-" * 70)
    print(f"\n🎯 Mejor Accuracy: {modelos[np.argmax(accuracy)].replace(chr(10), ' ')} ({max(accuracy):.4f})")
    print(f"🎯 Mejor Recall: {modelos[np.argmax(recall)].replace(chr(10), ' ')} ({max(recall):.4f})")
    print(f"🎯 Mejor F1-Score: {modelos[np.argmax(f1_score)].replace(chr(10), ' ')} ({max(f1_score):.4f})")
    
    print(f"\n📊 Gráfica generada: {SALIDA_GRAFICA.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
