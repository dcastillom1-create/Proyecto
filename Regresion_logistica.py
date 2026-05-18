import pandas as pd
import nltk
import funciones
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Descargar recursos NLTK (silencioso para evitar ruido en consola)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Cargar datos
pd.set_option("display.max_columns", None)
df_filtrado = funciones.cargar_datos()
print(df_filtrado.shape)

# Preprocesar texto: limpieza + stopwords + lematizacion
df_filtrado["Consulta_limpia"] = df_filtrado["Consulta"].apply(
    funciones.limpiar_stopwords_lematizar
)
df_filtrado = df_filtrado[df_filtrado["Consulta_limpia"].str.len() > 0].copy()
print("Registros tras preprocesamiento:", df_filtrado.shape)

# ── Vectorización TF-IDF ─────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=30)
X = vectorizer.fit_transform(df_filtrado["Consulta_limpia"])

tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())  # type: ignore[union-attr]

#print(tfidf_df.head())

def extract_top_tfidf_words(tfidf_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Extracts the top N words with the highest TF-IDF scores for each document in the input DataFrame.

    Args:
        tfidf_df (pd.DataFrame): A DataFrame where rows represent documents and columns are words,
                                 with TF-IDF scores as values.
        top_n (int, optional): Number of top words to extract per document. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame with one row per document and two columns:
                      - 'Top_Words': list of the top N words
                      - 'Top_Scores': list of corresponding TF-IDF scores
    """
    top_words = []
    top_scores = []

    for _, row in tfidf_df.iterrows():
        top_items = row.sort_values(ascending=False).head(top_n)
        top_words.append(list(top_items.index))
        top_scores.append(list(top_items.values))

    return pd.DataFrame({
        "Top_Words": top_words,
        "Top_Scores": top_scores
    })
    
top_tfidf = extract_top_tfidf_words(tfidf_df.head())

# Después:
top_tfidf["Top_Scores"] = top_tfidf["Top_Scores"].apply(
    lambda x: [round(score, 2) for score in x]
)

print(top_tfidf.head())

text = " ".join(df_filtrado["Consulta_limpia"].tolist())
print(text[:500])  # Imprime los primeros 500 caracteres del texto combinado

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
