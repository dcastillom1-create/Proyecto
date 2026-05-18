# CÓDIGO COMPLETO PARA SELECCIÓN MANUAL DE MUESTRAS DESDE DATASET

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CLASE PRINCIPAL PARA SELECCIÓN MANUAL
# ==========================================

class SelectorManualDataset:
    """
    Clase para selección manual interactiva de muestras desde dataset
    """
    
    def __init__(self, archivo_excel="data/Data_arreglada.xlsx"):
        """
        Inicializa el selector con datos del archivo Excel
        """
        print("🚀 INICIANDO SELECTOR MANUAL DE MUESTRAS")
        print("=" * 50)
        
        # Cargar y preparar datos
        self.df_original = self._cargar_datos(archivo_excel)
        self.X_data, self.y_data, self.vectorizer, self.encoder = self._preparar_datos()
        
        # Inicializar conjuntos
        self._inicializar_conjuntos()
        
        # Estado del modelo
        self.modelo_actual = None
        self.historial_selecciones = []
        self.iteracion_actual = 0
        
        print(f"✅ Inicialización completada")
        print(f"📊 Total muestras: {len(self.df_original)}")
        print(f"🎯 Categorías: {list(self.encoder.classes_)}")
    
    def _cargar_datos(self, archivo_path):
        """Carga y limpia los datos del Excel"""
        try:
            df = pd.read_excel(archivo_path, sheet_name="Sheet1")
            df.columns = ["Categoria", "Resuelta", "Consulta"]
            
            # Limpieza básica
            df = df.dropna(subset=["Consulta", "Categoria"])
            df = df[df["Resuelta"].str.strip().str.lower() != "sí"]
            df = df[df["Consulta"].str.strip() != ""]
            df = df.reset_index(drop=True)
            
            print(f"📁 Archivo cargado: {len(df)} consultas")
            return df
            
        except FileNotFoundError:
            print("❌ Archivo no encontrado. Usando datos simulados...")
            return self._crear_datos_simulados()
    
    def _crear_datos_simulados(self):
        """Crea datos simulados para demostración"""
        categorias = ["Carnetización", "Calendario académico", "Gestión Económica", 
                     "Gestión Académica", "Certificados"]
        
        consultas_base = [
            "Necesito ayuda urgente con mi carné estudiantil",
            "Problema con las fechas del calendario académico",
            "Error en el recibo de pago de matrícula",
            "No puedo inscribir materias este semestre",
            "Solicito certificado oficial de notas",
            "Consulta sobre proceso de matrícula",
            "Ayuda con validación de documentos",
            "Información sobre becas disponibles",
            "Problema con el sistema de inscripción",
            "Solicitud de información académica"
        ]
        
        datos = []
        np.random.seed(42)
        
        for cat in categorias:
            for i in range(1000):  # 1000 por categoría
                consulta_base = np.random.choice(consultas_base)
                variacion = np.random.choice([
                    "por favor ayúdenme", "es urgente", "necesito solución",
                    "tengo dudas sobre", "requiero información de",
                    "problema grave con", "solicito apoyo para"
                ])
                
                consulta = f"{consulta_base} {variacion} {cat.lower()} caso {i}"
                datos.append({
                    'Categoria': cat,
                    'Resuelta': 'No',
                    'Consulta': consulta
                })
        
        return pd.DataFrame(datos)
    
    def _limpiar_texto(self, texto):
        """Limpia el texto de consultas"""
        import re
        if pd.isna(texto):
            return ""
        
        texto = texto.lower()
        texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    def _preparar_datos(self):
        """Prepara los datos para el modelado"""
        # Limpiar texto
        self.df_original['Consulta_limpia'] = self.df_original['Consulta'].apply(self._limpiar_texto)
        
        # Vectorización
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_sparse = vectorizer.fit_transform(self.df_original['Consulta_limpia'])
        X_data = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else X_sparse  # type: ignore[union-attr]
        
        # Codificación de etiquetas
        encoder = LabelEncoder()
        y_data = encoder.fit_transform(self.df_original['Categoria'])
        
        print(f"📊 Matriz TF-IDF: {X_data.shape}")
        print(f"🏷️ Clases codificadas: {len(encoder.classes_)}")
        
        return X_data, y_data, vectorizer, encoder
    
    def _inicializar_conjuntos(self):
        """Inicializa los conjuntos de entrenamiento y pool"""
        # División inicial: 10% entrenamiento, 70% pool, 20% test
        # Usar stratify solo si todas las clases tienen al menos 2 muestras
        conteos = np.bincount(self.y_data)
        stratify_total = self.y_data if conteos.min() >= 2 else None

        X_temp, self.X_test, y_temp, self.y_test, df_temp, self.df_test = train_test_split(
            self.X_data, self.y_data, self.df_original,
            test_size=0.2, stratify=stratify_total, random_state=42
        )

        conteos_temp = np.bincount(y_temp)
        stratify_temp = y_temp if conteos_temp.min() >= 2 else None

        self.X_train, self.X_pool, self.y_train, self.y_pool, self.df_train, self.df_pool = train_test_split(
            X_temp, y_temp, df_temp,
            test_size=0.875, train_size=0.125,  # 10% del total para entrenamiento inicial
            stratify=stratify_temp, random_state=42
        )
        
        print(f"📊 Conjunto inicial entrenamiento: {len(self.X_train)}")
        print(f"📋 Pool disponible: {len(self.X_pool)}")
        print(f"🧪 Conjunto de prueba: {len(self.X_test)}")
    
    def entrenar_modelo(self):
        """Entrena el modelo SVM con los datos actuales"""
        # minlength garantiza que clases ausentes en y_train cuenten como 0
        conteos = np.bincount(self.y_train, minlength=len(self.encoder.classes_))
        min_clase = int(conteos.min())

        if min_clase >= 2:
            n_splits = min(3, min_clase)
            base = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
            cv = StratifiedKFold(n_splits=n_splits)
            self.modelo_actual = CalibratedClassifierCV(base, cv=cv, method="sigmoid")
            self.modelo_actual.fit(self.X_train, self.y_train)
        else:
            # Muy pocas muestras por clase: LogisticRegression da probabilidades directamente
            self.modelo_actual = LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced",
                random_state=42, solver="lbfgs"
            )
            self.modelo_actual.fit(self.X_train, self.y_train)
        
        # Evaluar
        accuracy = self.modelo_actual.score(self.X_test, self.y_test)
        print(f"🎯 Accuracy actual: {accuracy:.4f}")
        
        return self.modelo_actual
    
    def obtener_candidatos_manuales(self, n_candidatos=20, estrategia="incertidumbre", 
                                   filtro_categoria=None, filtro_keywords=None):
        """
        Obtiene candidatos para selección manual con filtros opcionales
        """
        if self.modelo_actual is None:
            self.entrenar_modelo()
        
        if len(self.X_pool) == 0:
            print("❌ No hay más muestras en el pool")
            return pd.DataFrame()
        
        assert self.modelo_actual is not None
        # Calcular probabilidades e incertidumbre
        probabilidades = self.modelo_actual.predict_proba(self.X_pool)
        
        if estrategia == "incertidumbre":
            scores = 1 - np.max(probabilidades, axis=1)
        elif estrategia == "margen":
            prob_ordenadas = np.sort(probabilidades, axis=1)
            scores = -(prob_ordenadas[:, -1] - prob_ordenadas[:, -2])
        else:  # entropia
            prob_clip = np.clip(probabilidades, 1e-15, 1-1e-15)
            scores = -np.sum(prob_clip * np.log(prob_clip), axis=1)
        
        # Crear DataFrame de candidatos
        candidatos = []
        predicciones = self.modelo_actual.predict(self.X_pool)
        
        for i in range(len(self.X_pool)):
            candidato = {
                'indice_pool': i,
                'indice_original': self.df_pool.iloc[i].name,
                'consulta_original': self.df_pool.iloc[i]['Consulta'],
                'consulta_limpia': self.df_pool.iloc[i]['Consulta_limpia'],
                'categoria_real': self.encoder.classes_[self.y_pool[i]],
                'prediccion': self.encoder.classes_[predicciones[i]],
                'confianza_maxima': np.max(probabilidades[i]),
                'score_incertidumbre': scores[i],
                'es_error': self.y_pool[i] != predicciones[i]
            }
            candidatos.append(candidato)
        
        df_candidatos = pd.DataFrame(candidatos)
        
        # Aplicar filtros
        if filtro_categoria:
            if isinstance(filtro_categoria, str):
                filtro_categoria = [filtro_categoria]
            df_candidatos = df_candidatos[df_candidatos['categoria_real'].isin(filtro_categoria)]
        
        if filtro_keywords:
            if isinstance(filtro_keywords, str):
                filtro_keywords = [filtro_keywords]
            patron = '|'.join(filtro_keywords)
            df_candidatos = df_candidatos[
                df_candidatos['consulta_limpia'].str.contains(patron, case=False, na=False)
            ]
        
        # Ordenar y limitar
        df_candidatos = df_candidatos.sort_values('score_incertidumbre', ascending=False)
        df_candidatos = df_candidatos.head(n_candidatos).reset_index(drop=True)
        
        return df_candidatos
    
    def mostrar_candidatos_para_seleccion(self, candidatos_df, mostrar_n=10):
        """
        Muestra candidatos de forma clara para selección manual
        """
        print(f"\n🔍 CANDIDATOS PARA SELECCIÓN MANUAL (Top {min(mostrar_n, len(candidatos_df))})")
        print("=" * 80)
        
        if len(candidatos_df) == 0:
            print("❌ No hay candidatos disponibles")
            return candidatos_df
        
        for idx, row in candidatos_df.head(mostrar_n).iterrows():
            print(f"\n📋 CANDIDATO #{idx}")
            print("-" * 40)
            
            # Información de categorías
            categoria_real = row['categoria_real']
            prediccion = row['prediccion']
            es_error = row['es_error']
            
            print(f"🏷️  Categoría Real: {categoria_real}")
            print(f"🤖 Predicción: {prediccion}")
            
            if es_error:
                print("❌ PREDICCIÓN INCORRECTA - ¡Muy valioso para el aprendizaje!")
            else:
                print("✅ Predicción correcta")
            
            print(f"📊 Confianza: {row['confianza_maxima']:.3f}")
            print(f"❓ Score Incertidumbre: {row['score_incertidumbre']:.3f}")
            
            # Mostrar consulta
            consulta = row['consulta_original']
            if len(consulta) > 150:
                consulta_mostrar = consulta[:150] + "..."
            else:
                consulta_mostrar = consulta
            
            print(f"📝 Consulta:")
            print(f'   "{consulta_mostrar}"')
            
            # Indicador visual de prioridad
            if es_error:
                print("   🔥 ALTA PRIORIDAD - Error de clasificación")
            elif row['score_incertidumbre'] > 0.3:
                print("   ⚠️  MEDIA PRIORIDAD - Alta incertidumbre")
            else:
                print("   ℹ️  BAJA PRIORIDAD - Baja incertidumbre")
        
        print(f"\n💡 Para seleccionar, usa los números de candidato (0-{len(candidatos_df.head(mostrar_n))-1})")
        return candidatos_df.head(mostrar_n)
    
    def seleccionar_muestras_interactivo(self, candidatos_df, indices_seleccionados):
        """
        Selecciona muestras de forma interactiva
        """
        if len(indices_seleccionados) == 0:
            print("⚠️ No se seleccionaron muestras")
            return False
        
        print(f"\n✅ PROCESANDO SELECCIÓN DE {len(indices_seleccionados)} MUESTRAS")
        print("-" * 50)
        
        # Validar índices
        indices_validos = [i for i in indices_seleccionados if 0 <= i < len(candidatos_df)]
        if len(indices_validos) != len(indices_seleccionados):
            print(f"⚠️ Algunos índices no son válidos. Usando: {indices_validos}")
        
        if len(indices_validos) == 0:
            print("❌ No hay índices válidos")
            return False
        
        # Obtener muestras seleccionadas
        candidatos_seleccionados = candidatos_df.iloc[indices_validos]
        indices_pool_originales = candidatos_seleccionados['indice_pool'].values
        
        # Mostrar resumen de selección
        print("📋 MUESTRAS SELECCIONADAS:")
        for i, (_, row) in enumerate(candidatos_seleccionados.iterrows()):
            print(f"  {i+1}. {row['categoria_real']} - {'❌' if row['es_error'] else '✅'}")
        
        # Extraer datos
        X_seleccionadas = self.X_pool[indices_pool_originales]
        y_seleccionadas = self.y_pool[indices_pool_originales]
        df_seleccionadas = self.df_pool.iloc[indices_pool_originales]
        
        # Agregar al entrenamiento
        self.X_train = np.vstack([self.X_train, X_seleccionadas])
        self.y_train = np.hstack([self.y_train, y_seleccionadas])
        self.df_train = pd.concat([self.df_train, df_seleccionadas], ignore_index=True)
        
        # Remover del pool
        mask = np.ones(len(self.X_pool), dtype=bool)
        mask[indices_pool_originales] = False
        
        self.X_pool = self.X_pool[mask]
        self.y_pool = self.y_pool[mask]
        self.df_pool = self.df_pool.iloc[mask].reset_index(drop=True)
        
        # Registrar selección
        self.iteracion_actual += 1
        seleccion_info = {
            'iteracion': self.iteracion_actual,
            'muestras_agregadas': len(indices_validos),
            'total_entrenamiento': len(self.X_train),
            'pool_restante': len(self.X_pool),
            'candidatos_seleccionados': candidatos_seleccionados[['categoria_real', 'es_error', 'score_incertidumbre']].to_dict('records')
        }
        self.historial_selecciones.append(seleccion_info)
        
        print(f"\n📊 ESTADO ACTUALIZADO:")
        print(f"🎯 Muestras agregadas: {len(indices_validos)}")
        print(f"📈 Total entrenamiento: {len(self.X_train)}")
        print(f"📋 Pool restante: {len(self.X_pool)}")
        
        # Reentrenar modelo
        print("\n🔄 Reentrenando modelo...")
        self.entrenar_modelo()
        
        return True
    
    def mostrar_estadisticas(self):
        """
        Muestra estadísticas del proceso de selección
        """
        print(f"\n📊 ESTADÍSTICAS DEL PROCESO DE SELECCIÓN")
        print("=" * 50)
        
        print(f"🔄 Iteraciones completadas: {self.iteracion_actual}")
        print(f"📈 Total muestras entrenamiento: {len(self.X_train)}")
        print(f"📋 Muestras restantes en pool: {len(self.X_pool)}")
        
        # Distribución por categorías
        categorias_train = [self.encoder.classes_[y] for y in self.y_train]
        distribucion = pd.Series(categorias_train).value_counts()
        
        print(f"\n📈 DISTRIBUCIÓN POR CATEGORÍAS EN ENTRENAMIENTO:")
        for cat, count in distribucion.items():
            porcentaje = (count / len(self.y_train)) * 100
            print(f"  {cat}: {count} muestras ({porcentaje:.1f}%)")
        
        # Historial de accuracy
        if self.modelo_actual:
            accuracy_actual = self.modelo_actual.score(self.X_test, self.y_test)
            print(f"\n🎯 Accuracy actual: {accuracy_actual:.4f}")
        
        return {
            'iteraciones': self.iteracion_actual,
            'total_entrenamiento': len(self.X_train),
            'pool_restante': len(self.X_pool),
            'distribucion': distribucion.to_dict(),
            'historial': self.historial_selecciones
        }
    
    def exportar_selecciones(self, archivo_salida="selecciones_manuales.xlsx"):
        """
        Exporta las muestras seleccionadas a un archivo Excel
        """
        try:
            with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
                # Muestras de entrenamiento actuales
                self.df_train.to_excel(writer, sheet_name='Entrenamiento', index=False)
                
                # Pool restante
                self.df_pool.to_excel(writer, sheet_name='Pool_Restante', index=False)
                
                # Historial de selecciones
                if self.historial_selecciones:
                    df_historial = pd.DataFrame(self.historial_selecciones)
                    df_historial.to_excel(writer, sheet_name='Historial', index=False)
            
            print(f"✅ Selecciones exportadas a: {archivo_salida}")
            return True
            
        except Exception as e:
            print(f"❌ Error al exportar: {e}")
            return False

# ==========================================
# FUNCIONES DE USO FÁCIL
# ==========================================

def iniciar_seleccion_manual(archivo_excel="data/Data_arreglada.xlsx"):
    """
    Función principal para iniciar la selección manual
    """
    selector = SelectorManualDataset(archivo_excel)
    
    print(f"\n🎯 SISTEMA DE SELECCIÓN MANUAL LISTO")
    print(f"💡 Comandos principales:")
    print(f"  candidatos = selector.obtener_candidatos_manuales(n_candidatos=15)")
    print(f"  selector.mostrar_candidatos_para_seleccion(candidatos)")
    print(f"  selector.seleccionar_muestras_interactivo(candidatos, [0, 2, 5, 8])")
    print(f"  selector.mostrar_estadisticas()")
    
    return selector

# ==========================================
# EJEMPLO DE USO COMPLETO
# ==========================================

def ejemplo_uso_completo():
    """
    Ejemplo completo de uso del sistema de selección manual
    """
    print("🚀 EJEMPLO DE USO COMPLETO")
    print("=" * 40)
    
    # 1. Inicializar
    selector = iniciar_seleccion_manual()
    
    # 2. Primera iteración
    print(f"\n1️⃣ PRIMERA ITERACIÓN")
    candidatos1 = selector.obtener_candidatos_manuales(n_candidatos=12)
    selector.mostrar_candidatos_para_seleccion(candidatos1, mostrar_n=8)
    
    # Simular selección
    indices_ejemplo1 = [0, 2, 4, 6]  # Seleccionar candidatos 0, 2, 4, 6
    print(f"\n🎯 Simulando selección: {indices_ejemplo1}")
    selector.seleccionar_muestras_interactivo(candidatos1, indices_ejemplo1)
    
    # 3. Segunda iteración con filtros
    print(f"\n2️⃣ SEGUNDA ITERACIÓN - Con filtro de categoría")
    candidatos2 = selector.obtener_candidatos_manuales(
        n_candidatos=10,
        filtro_categoria=["Gestión Económica", "Carnetización"]
    )
    selector.mostrar_candidatos_para_seleccion(candidatos2, mostrar_n=6)
    
    # Simular selección
    indices_ejemplo2 = [0, 1, 3]
    print(f"\n🎯 Simulando selección: {indices_ejemplo2}")
    selector.seleccionar_muestras_interactivo(candidatos2, indices_ejemplo2)
    
    # 4. Mostrar estadísticas finales
    print(f"\n3️⃣ ESTADÍSTICAS FINALES")
    stats = selector.mostrar_estadisticas()
    
    # 5. Exportar resultados
    selector.exportar_selecciones("ejemplo_selecciones.xlsx")
    
    return selector, stats

# ==========================================
# EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    # Ejecutar ejemplo completo
    selector_ejemplo, estadisticas = ejemplo_uso_completo()
    
    print(f"\n🎉 EJEMPLO COMPLETADO")
    print(f"📋 El objeto 'selector_ejemplo' está listo para uso interactivo")
    print(f"💾 Resultados guardados en 'ejemplo_selecciones.xlsx'")