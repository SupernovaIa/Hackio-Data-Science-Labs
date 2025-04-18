# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import math
import re


# Visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Para modelos NLP
# -----------------------------------------------------------------------
import spacy
from nltk.corpus import stopwords
import nltk
import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV, cross_val_score


# Otros
# -----------------------------------------------------------------------
from collections import Counter

# Descargar stopwords si no están disponibles
nltk.download('stopwords')

# Descargar recursos necesarios de nltk
nltk.download('vader_lexicon')


class ExploracionText:
    def __init__(self, dataframe, text_column, label_column, language="english"):
        """
        Clase para explorar, analizar y preprocesar texto en un DataFrame.

        Args:
        - dataframe: El DataFrame que contiene los datos.
        - text_column: Nombre de la columna que contiene el texto.
        - label_column: Nombre de la columna que contiene las etiquetas o clases.
        - language: Idioma del texto (por defecto, "english").
        """
        self.df = dataframe.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.nlp = spacy.load("en_core_web_sm") if language == "english" else spacy.load("es_core_news_sm")
        self.cleaned_column = "cleaned_text"

    def _limpiar_texto(self, text):
        # Expandir contracciones
        text = contractions.fix(text)  # Convierte "don't" -> "do not"
        
        # Limpieza de texto
        text = text.lower()  # Convertir a minúsculas
        text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
        text = re.sub(r'\d+', '', text)  # Eliminar números
        text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios o saltos de línea por un espacio
        text = text.strip()  # Quitar espacios en blanco al inicio y al final
        doc = self.nlp(text)  # Tokenizar con spaCy
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words]
        return " ".join(tokens)

    def explorar_datos(self):
        # Resumen de datos
        print("Información del DataFrame:")
        print(self.df.info())
        print("\nDescripción del DataFrame:")
        print(self.df.describe(include="all").T)
        print(f"\nDistribución de clases en la columna '{self.label_column}':")
        print(self.df[self.label_column].value_counts())
        
        # Visualización de distribución de clases
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=self.df,
            x=self.label_column,
            order=self.df[self.label_column].value_counts().index,
            palette='viridis'
        )
        plt.title(f'Distribución de Clases - {self.label_column}')
        plt.xlabel('Clase')
        plt.ylabel('Número de Elementos')
        plt.xticks(rotation=45)
        plt.show()

    def limpiar_textos(self, columna_limpia):
        # Aplicar limpieza
        print("Limpiando el texto...")
        self.df[columna_limpia] = self.df[self.text_column].apply(self._limpiar_texto)
        print("Texto limpiado y listo.")


    def generar_wordclouds(self):
        # Contar palabras por cada clase
        word_counts = {}
        for label in self.df[self.label_column].unique():
            all_words = " ".join(self.df[self.df[self.label_column] == label][self.cleaned_column])
            word_counts[label] = Counter(all_words.split())

        # Visualizar nubes de palabras
        print("Generando nubes de palabras...")
        num_labels = len(word_counts)
        rows = 2
        cols = math.ceil(num_labels / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), constrained_layout=True)
        axes = axes.flat

        for ax, (label, counts) in zip(axes, word_counts.items()):
            wordcloud = WordCloud(width=800, 
                                  height=800, 
                                  background_color='white').generate_from_frequencies(counts)
            ax.imshow(wordcloud)
            ax.set_title(f'Nube de Palabras - {label}', fontsize=10)
            ax.axis('off')

        for i in range(len(word_counts), len(axes)):
            fig.delaxes(axes[i])

        plt.show()
    
    def obtener_dataframe_limpio(self):
        """
        Devuelve el DataFrame limpio con la columna de texto procesada.
        """
        return self.df



class RepresentacionTexto:
    def __init__(self, dataframe, text_column, max_features=5000, embedding_model="bert-base-uncased"):
        """
        Clase para representar texto en formas numéricas como Bag of Words, TF-IDF y Embeddings,
        y devolver estas representaciones junto con el DataFrame original.

        Args:
        - dataframe: El DataFrame que contiene los datos.
        - text_column: Columna del DataFrame que contiene el texto.
        - max_features: Número máximo de características para Bag of Words y TF-IDF.
        - embedding_model: Modelo preentrenado para generar embeddings (por defecto, "bert-base-uncased").
        """
        self.df = dataframe
        self.text_column = text_column
        self.max_features = max_features

        # Modelo y tokenizador para embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

    def generar_bow(self):
        """
        Genera representación Bag of Words (BoW) y devuelve un DataFrame con la representación.

        Returns:
        - DataFrame original con la representación Bag of Words como columnas adicionales.
        """
        print("Generando Bag of Words...")
        vectorizer_bow = CountVectorizer(max_features=self.max_features)
        X_bow = vectorizer_bow.fit_transform(self.df[self.text_column])
        bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
        df_bow = pd.concat([self.df.reset_index(drop=True), bow_df], axis=1)
        print("Bag of Words generado.")
        return df_bow

    def generar_tfidf(self):
        """
        Genera representación TF-IDF y devuelve un DataFrame con la representación.

        Returns:
        - DataFrame original con la representación TF-IDF como columnas adicionales.
        """
        print("Generando representación TF-IDF...")
        vectorizer_tfidf = TfidfVectorizer(max_features=self.max_features)
        X_tfidf = vectorizer_tfidf.fit_transform(self.df[self.text_column])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
        df_tfidf = pd.concat([self.df.reset_index(drop=True), tfidf_df], axis=1)
        print("Representación TF-IDF generada.")
        return df_tfidf

    def generar_embeddings(self, max_length=512):
        """
        Genera embeddings usando un modelo preentrenado como BERT y devuelve un DataFrame con la representación.

        Returns:
        - DataFrame original con la representación de embeddings como columnas adicionales.
        """
        print("Generando embeddings...")
        embeddings = []
        for text in self.df[self.text_column]:
            # Tokenizar el texto
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            
            # Pasar los tokens por el modelo
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            
            # Extraer la representación del último nivel (promedio sobre la secuencia)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

        embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings[0].shape[0])])
        df_embeddings = pd.concat([self.df.reset_index(drop=True), embeddings_df], axis=1)
        print("Embeddings generados.")
        return df_embeddings
    

    def transformar_nuevos_textos(self, nuevos_textos, metodo="tfidf"):
        """
        Transforma nuevos textos usando el vectorizador existente.

        Args:
        - nuevos_textos: Lista de textos a transformar.
        - metodo: Método de vectorización ("bow" o "tfidf").

        Returns:
        - Matriz transformada de los textos nuevos.
        """
        if metodo == "bow":
            if not self.vectorizer_bow:
                raise ValueError("El vectorizador Bag of Words no ha sido inicializado. Genera BoW primero.")
            return self.vectorizer_bow.transform(nuevos_textos)
        
        elif metodo == "tfidf":
            if not self.vectorizer_tfidf:
                raise ValueError("El vectorizador TF-IDF no ha sido inicializado. Genera TF-IDF primero.")
            return self.vectorizer_tfidf.transform(nuevos_textos)
        
        else:
            raise ValueError("Método no soportado. Usa 'bow' o 'tfidf'.")


class ReduccionDimensionalidadPCA:
    def __init__(self, n_componentes=50):
        """
        Clase para aplicar PCA (Análisis de Componentes Principales) a un DataFrame.

        Args:
        - n_componentes: Número de componentes principales a conservar.
        """
        self.n_componentes = n_componentes
        self.pca = None

    def ajustar_transformar(self, df, columnas_a_reducir):
        """
        Aplica PCA para reducir la dimensionalidad de las columnas especificadas en un DataFrame.

        Args:
        - df: DataFrame con las columnas a reducir.
        - columnas_a_reducir: Lista de nombres de columnas que serán procesadas.

        Returns:
        - Un nuevo DataFrame con las columnas reducidas.
        """
        print(f"Aplicando PCA para reducir {len(columnas_a_reducir)} dimensiones a {self.n_componentes} dimensiones...")
        
        # Extraer los datos de las columnas especificadas
        datos_a_reducir = df[columnas_a_reducir].values
        
        # Aplicar PCA
        self.pca = PCA(n_components=self.n_componentes)
        datos_reducidos = self.pca.fit_transform(datos_a_reducir)
        
        # Crear nuevas columnas para las dimensiones reducidas
        nuevas_columnas = [f"pca_componente_{i}" for i in range(self.n_componentes)]
        df_reducido = pd.DataFrame(datos_reducidos, columns=nuevas_columnas, index=df.index)
        
        # Combinar el DataFrame reducido con las columnas originales que no fueron procesadas
        df_resto = df.drop(columns=columnas_a_reducir)
        df_final = pd.concat([df_resto, df_reducido], axis=1)

        print("Reducción completada. Nuevas dimensiones agregadas al DataFrame.")
        return df_final

    def explicar_varianza(self, mostrar_grafico=True):
        """
        Muestra la varianza explicada por los componentes principales y genera un gráfico.

        Args:
        - mostrar_grafico: Si es True, muestra un gráfico de la varianza explicada acumulada.

        Returns:
        - Lista de varianza explicada acumulada por cada componente principal.
        """
        if self.pca is None:
            raise ValueError("PCA aún no ha sido ajustado. Llama primero a 'ajustar_transformar'.")
        
        varianza_explicada = self.pca.explained_variance_ratio_
        varianza_acumulada = varianza_explicada.cumsum()

        # Generar gráfico con Seaborn
        if mostrar_grafico:
            componentes = range(1, len(varianza_acumulada) + 1)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=componentes, y=varianza_explicada, alpha=0.6, label="Varianza explicada individual", color="b")
            sns.lineplot(x=componentes, y=varianza_acumulada, marker='o', label="Varianza acumulada", color="r")
            plt.axhline(y=0.95, color='g', linestyle='--', label="Umbral 95%")
            plt.title("Varianza explicada por PCA")
            plt.xlabel("Número de componentes principales")
            plt.ylabel("Varianza explicada")
            plt.xticks(componentes, rotation = 90)
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return varianza_acumulada
    
    def transformar(self, df, columnas_a_reducir):
        """
        Transforma nuevos datos usando un PCA previamente ajustado.

        Args:
        - df: DataFrame con las columnas a transformar.
        - columnas_a_reducir: Lista de nombres de columnas que serán procesadas.

        Returns:
        - Un nuevo DataFrame con las columnas reducidas.
        """
        if self.pca is None:
            raise ValueError("PCA aún no ha sido ajustado. Llama primero a 'ajustar_transformar'.")

        print(f"Transformando nuevas muestras usando PCA...")
        
        # Extraer los datos de las columnas especificadas
        datos_a_reducir = df[columnas_a_reducir].values
        
        # Transformar los datos con el PCA ajustado
        datos_reducidos = self.pca.transform(datos_a_reducir)
        
        # Crear nuevas columnas para las dimensiones reducidas
        nuevas_columnas = [f"pca_componente_{i}" for i in range(self.n_componentes)]
        df_reducido = pd.DataFrame(datos_reducidos, columns=nuevas_columnas, index=df.index)
        
        # Combinar el DataFrame reducido con las columnas originales que no fueron procesadas
        df_resto = df.drop(columns=columnas_a_reducir)
        df_final = pd.concat([df_resto, df_reducido], axis=1)

        print("Transformación completada. Nuevas dimensiones agregadas al DataFrame.")
        return df_final



from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ClasificacionTextoAvanzada:
    def __init__(self, modelo="naive_bayes", naive_bayes_tipo="gaussian"):
        """
        Clase para la clasificación de texto con optimización.

        Args:
        - modelo: Modelo de clasificación ('naive_bayes', 'random_forest', 'logistic_regression').
        - naive_bayes_tipo: Tipo de Naive Bayes ('gaussian' o 'multinomial').
        """
        self.modelo = self._seleccionar_modelo(modelo, naive_bayes_tipo)
        self.pipeline = Pipeline([
            ("classifier", self.modelo)
        ])
    
    def _seleccionar_modelo(self, modelo, naive_bayes_tipo):
        """
        Selecciona el modelo de clasificación basado en la entrada.

        Args:
        - modelo: Modelo especificado ('naive_bayes', 'random_forest', 'logistic_regression').
        - naive_bayes_tipo: Tipo de Naive Bayes ('gaussian' o 'multinomial').

        Returns:
        - Modelo seleccionado.
        """
        if modelo == "naive_bayes":
            if naive_bayes_tipo == "gaussian":
                return GaussianNB()
            elif naive_bayes_tipo == "multinomial":
                return MultinomialNB()
            else:
                raise ValueError("Tipo de Naive Bayes no soportado. Usa 'gaussian' o 'multinomial'.")
        elif modelo == "random_forest":
            return RandomForestClassifier(random_state=42)
        elif modelo == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError("Modelo no soportado.")
        
    def dividir_datos(self, X, y, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
        - X: Datos de entrada (representaciones numéricas, e.g., TF-IDF o PCA).
        - y: Etiquetas.
        - test_size: Proporción del conjunto de prueba (por defecto, 0.2).
        - random_state: Semilla para reproducibilidad (por defecto, 42).

        Returns:
        - X_train, X_test, y_train, y_test: Datos divididos en entrenamiento y prueba.
        """
        print(f"Dividiendo datos: {test_size*100}% para prueba.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
        print(f"Tamaño del conjunto de prueba: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def ajustar_hyperparametros(self, X_train, y_train, params, scoring='f1_weighted'):
        """
        Optimiza los hiperparámetros del modelo utilizando GridSearchCV.

        Args:
        - X_train: Datos de entrada ya representados (e.g., TF-IDF o PCA).
        - y_train: Etiquetas de entrenamiento.
        - params: Diccionario de parámetros para la búsqueda.

        Returns:
        - Mejor modelo ajustado.
        """
        print("Buscando los mejores hiperparámetros...")
        grid = GridSearchCV(self.pipeline, param_grid=params, cv=3, scoring=scoring, n_jobs=-1)
        grid.fit(X_train, y_train)
        print("Mejores parámetros encontrados:", grid.best_params_)
        self.pipeline = grid.best_estimator_
        print("Modelo actualizado con los mejores parámetros.")
    
    def entrenar(self, X_train, y_train):
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
        - X_train: Datos de entrada ya representados (e.g., TF-IDF o PCA).
        - y_train: Etiquetas de entrenamiento.
        """
        print("Entrenando modelo...")
        self.pipeline.fit(X_train, y_train)
        print("Modelo entrenado.")
    
    def validar(self, X_train, y_train):
        """
        Realiza validación cruzada para evaluar el modelo.

        Args:
        - X_train: Datos de entrada ya representados (e.g., TF-IDF o PCA).
        - y_train: Etiquetas de entrenamiento.

        Returns:
        - Promedio de las puntuaciones de validación cruzada.
        """
        print("Realizando validación cruzada...")
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"Puntuaciones de validación cruzada: {scores}")
        print(f"Puntuación media: {scores.mean():.4f}")
        return scores.mean()



class AnalisisSentimientos:
    """
    Clase para realizar análisis de sentimientos en un dataframe
    y generar visualizaciones basadas en los resultados.
    """
    def __init__(self, dataframe, columna_texto):
        """
        Inicializa el analizador de sentimientos y prepara el dataframe.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame que contiene los datos.
        columna_texto : str
            Nombre de la columna que contiene los textos a analizar.
        """
        self.dataframe = dataframe.copy()
        self.columna_texto = columna_texto
        self.sia = SentimentIntensityAnalyzer()
        self._preparar_datos()
    
    def _preparar_datos(self):
        """
        Aplica el análisis de sentimientos y separa las puntuaciones en columnas individuales.
        """
        self.dataframe['scores_sentimientos'] = self.dataframe[self.columna_texto].apply(self._analizar_texto)
        self.dataframe['neg'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['neg'])
        self.dataframe['neu'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['neu'])
        self.dataframe['pos'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['pos'])
        self.dataframe['compound'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['compound'])
        self.dataframe.drop(columns=['scores_sentimientos'], inplace=True)
    
    def _analizar_texto(self, texto):
        """
        Analiza un texto y retorna las puntuaciones de sentimientos.

        Parameters:
        ----------
        texto : str
            Texto a analizar.

        Returns:
        -------
        dict
            Diccionario con las puntuaciones de negatividad, neutralidad, positividad y compound.
        """
        return self.sia.polarity_scores(texto)
    
    def graficar_distribucion_sentimientos(self):
        """
        Genera un gráfico de barras para visualizar la distribución de sentimientos (neg, neu, pos).
        """
        mean_scores = self.dataframe[['neg', 'neu', 'pos']].mean()
        mean_scores.plot(kind='bar', figsize=(8, 5), title='Distribución Promedio de Sentimientos')
        plt.xlabel('Tipo de Sentimiento')
        plt.ylabel('Puntuación Promedio')
        plt.xticks(rotation=0)
        plt.show()

    def graficar_distribucion_compound(self):
        """
        Genera un histograma para visualizar la distribución de las puntuaciones compound.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.dataframe['compound'], bins=20, kde=True, color='blue')
        plt.title('Distribución de Puntuaciones Compound')
        plt.xlabel('Puntuación Compound')
        plt.ylabel('Frecuencia')
        plt.show()

    def graficar_mapa_calor_sentimientos(self):
        """
        Genera un mapa de calor para visualizar las correlaciones entre las puntuaciones de sentimientos.
        """
        matriz_corr = self.dataframe[['neg', 'neu', 'pos', 'compound']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Mapa de Calor de Correlaciones de Sentimientos')
        plt.show()

    def obtener_resumen(self):
        """
        Retorna un resumen estadístico de las puntuaciones de sentimientos.

        Returns:
        -------
        pd.DataFrame
            Resumen estadístico (count, mean, std, min, max) de las puntuaciones.
        """
        return self.dataframe[['neg', 'neu', 'pos', 'compound']].describe()