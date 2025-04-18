# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px

# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon
from sklearn.cluster import DBSCAN # para usar DBSCAN

# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product

# Gestionar warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')


class GestionOutliersUnivariados:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")


    def visualizar_outliers_univariados(self, tipo_grafica, color="blue", whis=1.5, tamano_grafica=(20, 15)):
        """
        Visualiza los outliers univariados mediante boxplots o histogramas.

        Parámetros:
        -----------
        tipo_grafica (str): "B" (boxplot) o "H" (histograma).
        color (str): Color de los gráficos.
        whis (float): Valor para definir el límite de los bigotes en los boxplots.
        tamano_grafica (tuple): Tamaño de la figura.
        """
        tipo_grafica = tipo_grafica.upper()
        
        num_cols = len(self._separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        _, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.select_dtypes(include=np.number).columns):
            if tipo_grafica == "B":
                sns.boxplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], whis=whis,
                            flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            elif tipo_grafica == "H":
                sns.histplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], bins=50)
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.show()

    def detectar_outliers_z_score(self):
        """
        Detecta outliers utilizando z-score.
        """
        diccionario_resultados_z = {}

        for columna in self._separar_variables_tipo()[0].columns:
            z_scores = abs(zscore(self.dataframe[columna]))
            diccionario_resultados_z[columna] = self.dataframe[z_scores > 3]
            print(f"La cantidad de outliers que tenemos para la columna {columna.upper()} es ", 
                  f"{diccionario_resultados_z[columna].shape[0]}")
        return diccionario_resultados_z

    def detectar_outliers_iqr(self, limite_outliers=1.5):
        """
        Detecta outliers utilizando el rango intercuartil (IQR).
        """
        diccionario_iqr = {}
        for columna in self._separar_variables_tipo()[0].columns:
            q1, q3 = np.nanpercentile(self.dataframe[columna], (25, 75))
            iqr = q3 - q1
            limite_inferior = q1 - limite_outliers * iqr
            limite_superior = q3 + limite_outliers * iqr
            df_outliers = self.dataframe[(self.dataframe[columna] < limite_inferior) | (self.dataframe[columna] > limite_superior)]
            if not df_outliers.empty:
                diccionario_iqr[columna] = self.dataframe[self.dataframe.index.isin(df_outliers.index.tolist())]
                print(f"La cantidad de outliers que tenemos para la columna {columna.upper()} es "
                      f"{diccionario_iqr[columna].shape[0]}")
        return diccionario_iqr

    def detectar_outliers(self, limite_outliers = 1.5, metodo="iqr" ):
        """
        Detecta outliers utilizando el método especificado.

        Parámetros:
        -----------
        metodo (str): Método para detectar outliers: "z_score", "z_score_modificado" o "iqr".
        kwargs: Argumentos adicionales para los métodos.

        Returns:
        --------
        dict: Diccionario de columnas con listas de índices de outliers.
        """
        if metodo == "z_score":
            return self.detectar_outliers_z_score()
        elif metodo == "iqr":
            return self.detectar_outliers_iqr(limite_outliers)
        else:
            raise ValueError("Método no válido. Los métodos disponibles son 'z_score', 'z_score_modificado' e 'iqr'.")


class GestionOutliersMultivariados:

    def __init__(self, dataframe, contaminacion = [0.01, 0.05, 0.1, 0.15]):
        self.dataframe = dataframe
        self.contaminacion = contaminacion

    def separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")


    def visualizar_outliers_bivariados(self, vr, tamano_grafica = (20, 15)):

        num_cols = len(self.separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.separar_variables_tipo()[0].columns):
            if columna == vr:
                fig.delaxes(axes[indice])
        
            else:
                sns.scatterplot(x = vr, 
                                y = columna, 
                                data = self.dataframe,
                                ax = axes[indice])
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None, ylabel = None)

        plt.tight_layout()



    def explorar_outliers_lof(self, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], vecinos=[20, 30], colores={-1: "red", 1: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Local Outlier Factor (LOF) y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo LOF. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - vecinos : list of int, opcional. Lista de números de vecinos a usar en el algoritmo LOF. Por defecto es [600, 1200].
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo LOF.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).

        Returns:
        
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada combinación de vecinos y nivel de contaminación especificado.
        """

        # Hacemos una copia del dataframe original para no hacer modificaciones sobre el original
        df_lof = self.dataframe.copy()
        
        # Extraemos las columnas numéricas 
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Generamos todas las posibles combinaciones entre los vecinos y el nivel de contaminación
        combinaciones = list(product(vecinos, indice_contaminacion))

        # Iteramos por cada posible combinación
        for combinacion in combinaciones:
            # Aplicar LOF con un número de vecinos y varias tasas de contaminación
            clf = LocalOutlierFactor(n_neighbors=combinacion[0], contamination=combinacion[1])
            y_pred = clf.fit_predict(self.dataframe[col_numericas])

            # Agregar la predicción de outliers al DataFrame
            df_lof["outlier"] = y_pred

            num_filas = math.ceil(len(col_numericas) / 2)

            fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
            axes = axes.flat

            # Asegurar que la variable dependiente no está en las columnas numéricas
            if var_dependiente in col_numericas:
                col_numericas.remove(var_dependiente)

            for indice, columna in enumerate(col_numericas):
                # Visualizar los outliers en un gráfico
                sns.scatterplot(x=var_dependiente, 
                                y=columna, 
                                data=df_lof,
                                hue="outlier", 
                                palette=colores, 
                                style="outlier", 
                                size=2,
                                ax=axes[indice])
                
                axes[indice].set_title(f"Contaminación = {combinacion[1]} y vecinos {combinacion[0]} y columna {columna.upper()}")
            
            plt.tight_layout()

            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])

            plt.show()



    def explorar_outliers_if(self, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], estimadores=1000, colores={-1: "red", 1: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo Isolation Forest. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - estimadores : int, opcional. Número de estimadores (árboles) a utilizar en el algoritmo Isolation Forest. Por defecto es 1000.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo Isolation Forest.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).
        
        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada valor de contaminación especificado.
        """
    
        df_if = self.dataframe.copy()

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        num_filas = math.ceil(len(col_numericas) / 2)

        for contaminacion in indice_contaminacion: 
            
            ifo = IsolationForest(random_state=42, 
                                n_estimators=estimadores, 
                                contamination=contaminacion,
                                max_samples="auto",  
                                n_jobs=-1)
            ifo.fit(self.dataframe[col_numericas])
            prediccion_ifo = ifo.predict(self.dataframe[col_numericas])
            df_if["outlier"] = prediccion_ifo

            fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
            axes = axes.flat
            for indice, columna in enumerate(col_numericas):
                if columna == var_dependiente:
                    fig.delaxes(axes[indice])

                else:
                    # Visualizar los outliers en un gráfico
                    sns.scatterplot(x=var_dependiente, 
                                    y=columna, 
                                    data=df_if,
                                    hue="outlier", 
                                    palette=colores, 
                                    style="outlier", 
                                    size=2,
                                    ax=axes[indice])
                    
                    axes[indice].set_title(f"Contaminación = {contaminacion} y columna {columna.upper()}")
                    plt.tight_layout()
                
                        
            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])

    def calcular_epsilon_dbscan(self):
        """
        Calcula el valor óptimo de epsilon para el algoritmo DBSCAN utilizando el método del gráfico K-distance.

        Este método separa las variables numéricas del DataFrame, calcula las distancias a los vecinos más cercanos,
        y genera un gráfico de línea que muestra la distancia al segundo vecino más cercano para cada punto.
        El punto donde la curva tiene el mayor cambio de pendiente puede ser un buen valor para epsilon.

        Params:
            No devuelve ningún parámetro

        Retorna:
            Esta función no retorna ningún valor, pero muestra un gráfico de línea interactivo utilizando Plotly.
        """
        df_num = self.separar_variables_tipo()[0]

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(df_num)
        distancias, _ = nbrs.kneighbors(df_num)

        df_distancias = pd.DataFrame(np.sort(distancias, axis=0)[:,1], columns=["epsilon"])

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=df_distancias.index, y="epsilon", data=df_distancias, marker='o')

        # Añadimos el título y etiquetas de los ejes
        plt.title('Gráfico K-distance')
        plt.xlabel('Índice')
        plt.ylabel('Epsilon')

        # Mostramos el gráfico
        plt.tight_layout()


    def explorar_outliers_dbscan(self, epsilon, min_muestras, var_dependiente, colores={-1: "red", 0: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo DBSCAN y visualiza los resultados.

        Params:
            - epsilon : float. El valor de epsilon (radio máximo de la vecindad) para el algoritmo DBSCAN.
        
            - min_muestras : int. El número mínimo de muestras en una vecindad para que un punto sea considerado como un núcleo en DBSCAN.
        
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo DBSCAN.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 0: "grey"}).

        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados utilizando el algoritmo DBSCAN.
        """
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()
        num_filas = math.ceil(len(col_numericas) / 2)

        df_dbscan = self.dataframe.copy()

        model = DBSCAN(eps=epsilon, min_samples=min_muestras).fit(self.dataframe[col_numericas])
        outliers = model.labels_

        df_dbscan["outlier"] = outliers

        fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
        axes = axes.flat

        for indice, columna in enumerate(col_numericas):
            if columna == var_dependiente:
                fig.delaxes(axes[indice])
            else:
                # Visualizar los outliers en un gráfico
                sns.scatterplot(x=var_dependiente, 
                                y=columna, 
                                data=df_dbscan,
                                hue="outlier", 
                                palette=colores, 
                                style="outlier", 
                                size=2,
                                ax=axes[indice])
                
                axes[indice].set_title(f"Columna {columna.upper()}")
                plt.tight_layout()
        
        if len(col_numericas) % 2 != 0:
            fig.delaxes(axes[-1])


    def detectar_outliers_lof(self, n_neighbors, contaminacion):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Local Outlier Factor (LOF).
        """
        df_lof = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contaminacion)
        y_pred = clf.fit_predict(self.dataframe[col_numericas])
        df_lof["outlier"] = y_pred

        return df_lof

    def detectar_outliers_if(self,  contaminacion, n_estimators=1000):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest.
        """
        df_if = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        ifo = IsolationForest(random_state=42, n_estimators=n_estimators, contamination=contaminacion, max_samples="auto", n_jobs=-1)
        ifo.fit(self.dataframe[col_numericas])
        prediccion_ifo = ifo.predict(self.dataframe[col_numericas])
        df_if["outlier"] = prediccion_ifo

        return df_if

    def detectar_outliers_dbscan(self, epsilon, min_samples):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo DBSCAN.
        """
        df_dbscan = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(self.dataframe[col_numericas])
        outliers = model.labels_
        df_dbscan["outlier"] = outliers

        return df_dbscan
    
    def eliminar_outliers(self):
        pass
    
    def imputar_outliers(self, data, metodo='media'):
        """
        Imputa los valores outliers en las columnas numéricas según el método especificado.
        
        Params:
            - data: DataFrame con los datos incluyendo la columna 'outlier'.
            - metodo: str, método de imputación ('media', 'mediana', 'moda').
        
        Returns:
            - DataFrame con los valores outliers imputados.
        """

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Diccionario de métodos de imputación
        metodos_imputacion = {
            'media': lambda x: x.mean(),
            'mediana': lambda x: x.median(),
            'moda': lambda x: x.mode()[0]
        }

        if metodo not in metodos_imputacion:
            raise ValueError("Método de imputación no reconocido. Utilice 'media', 'mediana' o 'moda'.")

        for col in col_numericas:
            valor_imputacion = metodos_imputacion[metodo](data.loc[data['outlier'] != -1, col])
            data.loc[data['outlier'] == -1, col] = valor_imputacion
        
        return data.drop("outlier", axis = 1)

    def capar_outliers(self, data,  lower_percentile=0.01, upper_percentile=0.99):
        """
        Capa los valores outliers en las columnas numéricas según los percentiles especificados.
        
        Params:
            - lower_percentile: float, percentil inferior para capar los valores (por defecto 0.01).
            - upper_percentile: float, percentil superior para capar los valores (por defecto 0.99).
        
        Returns:
            - DataFrame con los valores outliers capados.
        """
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        for col in col_numericas:
            lower_bound = data[col].quantile(lower_percentile)
            upper_bound = data[col].quantile(upper_percentile)
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
        
        return data.drop("outlier", axis = 1)

    def transformar_outliers(self, data, metodo='log'):
        """
        Transforma los valores outliers en las columnas numéricas según el método especificado.
        
        Params:
            - metodo: str, método de transformación ('log', 'sqrt', 'inv').
        
        Returns:
            - DataFrame con los valores outliers transformados.
        """

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Diccionario de métodos de transformación
        metodos_transformacion = {
            'log': np.log1p,  # log(1 + x) para evitar problemas con log(0)
            'sqrt': np.sqrt,
            'inv': lambda x: 1 / (x + np.finfo(float).eps)  # añadir epsilon para evitar división por cero
        }

        if metodo not in metodos_transformacion:
            raise ValueError("Método de transformación no reconocido. Utilice 'log', 'sqrt' o 'inv'.")

        for col in col_numericas:
            transform_func = metodos_transformacion[metodo]
            outlier_indices = data['outlier'] == -1
            data.loc[outlier_indices, col] = transform_func(data.loc[outlier_indices, col])
        
        return data.drop("outlier", axis = 1)

