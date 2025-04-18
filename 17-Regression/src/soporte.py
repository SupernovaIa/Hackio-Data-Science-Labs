import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def separar_df(df):
    """
    Separa un DataFrame en dos DataFrames: uno que contiene columnas numéricas y otro que contiene columnas categóricas.

    Parámetros:
    - df (pd.DataFrame): El DataFrame a separar.

    Retorna:
    - (tuple): Una tupla que contiene:
    - pd.DataFrame: DataFrame con columnas numéricas.
    - pd.DataFrame: DataFrame con columnas categóricas.
    """
    df_num = df.select_dtypes(include = np.number)
    df_cat = df.select_dtypes(include = 'O')

    return df_num, df_cat


def plot_numericas(df, nbins = 20):
    """
    Genera histogramas para cada columna numérica de un DataFrame, organizados en un diseño de cuadrícula.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos a graficar.
    """

    df_num = separar_df(df)[0]

    n_plots = len(df_num.columns)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = (15, 10))

    axes = axes.flat

    for i, col in enumerate(df_num.columns):

        sns.histplot(x = col, data = df_num, ax = axes[i], bins = nbins)
        axes[i].set_title(col)
        axes[i].set_xlabel('')

    # Quitamos el último, si queda vacío
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def plot_categoricas(df):
    """
    Genera gráficos de barras para cada columna categórica de un DataFrame, organizados en un diseño de cuadrícula.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos categóricos a graficar.
    """
    df_cat = separar_df(df)[1]

    n_plots = len(df_cat.columns)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = (15, 10))
    axes = axes.flat

    for i, col in enumerate(df_cat.columns):

        sns.countplot(x = col, 
                      data = df_cat, 
                      ax = axes[i], 
                      palette='mako', 
                      order = df[col].value_counts().index)
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)

    # Quitamos el último, si queda vacío
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def relacion_vr_categoricas(df, vr):
    """
    Genera gráficos de barras para mostrar la relación entre una variable numérica especificada y todas las variables categóricas en un DataFrame.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - vr (str): Nombre de la variable numérica para la cual se calcularán y graficarán los valores medios contra cada variable categórica.
    """
    
    df_cat = df.select_dtypes(include = 'O')
    cols_cat = df_cat.columns

    n_plots = len(cols_cat)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = (15, 10))
    axes = axes.flat

    for i, col in enumerate(cols_cat):

        datos_agrupados = df.groupby(col)[vr].mean().sort_values(ascending=False).reset_index()
        sns.barplot(x = col,
                    y = vr,
                    data = datos_agrupados,
                    ax = axes[i], 
                    palette = 'mako')
        
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)

    # Quitamos el último, si queda vacío
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def relacion_vr_numericas(df, vr):
    """
    Genera gráficos de dispersión para mostrar la relación entre una variable numérica especificada y todas las demás variables numéricas en un DataFrame.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - vr (str): Nombre de la variable numérica que se graficará en el eje y contra cada una de las demás variables numéricas en el eje x.
    """
    
    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = (15, 10))
    axes = axes.flat

    for i, col in enumerate(cols_num):

        if col == vr:
            fig.delaxes(axes[i])

        else:
            sns.scatterplot(x = col,
                        y = vr,
                        data = df_num,
                        ax = axes[i], 
                        palette = 'mako')
            
            axes[i].set_title(col)
            axes[i].set_xlabel('')

    # Quitamos el último, si queda vacío
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def plot_matriz_correlacion(df):
    """
    Genera un mapa de calor triangular que representa la matriz de correlación entre las variables numéricas de un DataFrame.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene las variables para calcular la matriz de correlación.
    """

    matriz_correlacion = df.corr(numeric_only=True)
    plt.figure(figsize=(5, 5))

    # Máscara para que sea triangular
    mascara = np.triu(np.ones_like(matriz_correlacion, dtype = np.bool_))
    sns.heatmap(matriz_correlacion, 
                annot=True, 
                vmin = -1, 
                vmax = 1, 
                mask=mascara)
    

def detectar_outliers(df, vr):
    """
    Genera gráficos de caja (boxplots) para detectar valores atípicos (outliers) en cada variable numérica de un DataFrame.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene las variables numéricas para las cuales se evaluarán los outliers.
    - vr (str): Nombre de la variable que se usará para generar los gráficos de outliers. (No se utiliza directamente en esta función, pero podría indicarse para su futura inclusión).
    """

    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_filas = math.ceil(n_plots/2)

    # Obtener la paleta de colores
    cmap = plt.cm.get_cmap('mako', n_plots)
    color_list = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = (9, 5))
    axes = axes.flat

    for i, col in enumerate(cols_num):

        sns.boxplot(x = col, 
                    data = df_num,
                    ax = axes[i],
                    color=color_list[i]) 
        
        axes[i].set_title(f'Outliers de {col}')
        axes[i].set_xlabel('')

    # Quitamos el último, si queda vacío
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()
    