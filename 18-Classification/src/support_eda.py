# Data processing  
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualization  
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Mathematics  
# -----------------------------------------------------------------------
import math


def plot_numeric_distribution(df, first, last, col, n=1, size = (10, 5), rotation=45):
    """
    Plots the distribution of numeric values in a specified column within a given range, using aligned bins.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - first (int or float): The lower bound of the range to plot.
    - last (int or float): The upper bound of the range to plot.
    - col (str): The name of the column to analyze.
    - n (int, optional): The bin width for the histogram. Defaults to 1.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (10, 5).
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Raises:
    - ValueError: If `n` is not positive or if `first` is not less than `last`.

    Returns:
    - None: Displays a histogram of the numeric distribution within the specified range.
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Bin width (n) must be a positive integer.")
    if first >= last:
        raise ValueError("'first' must be less than 'last'.")
    
    # Define the bin edges to align with ticks
    bin_edges = np.arange(first, last + n, n)
    
    # Filter the data
    filtered_data = df[df[col].between(first, last)][col]
    
    if filtered_data.empty:
        print(f"No data available for the range {first} to {last}.")
        return

    # Set dynamic figure size based on the range
    plt.figure(figsize=size)
    
    # Create the histogram with aligned bins
    sns.histplot(filtered_data, bins=bin_edges, kde=False, color="skyblue", edgecolor="black")
    
    # Add title and labels
    plt.title(f"Distribution of {col} ({first}–{last})")
    plt.xlabel("")
    plt.ylabel("Frequency")
    
    # Set x-ticks to align with bins
    plt.xticks(bin_edges, rotation=rotation)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_categoric_distribution(df, col, size = (8, 4), color='mako', rotation=45):
    """
    Plots the distribution of a categorical column, showing the count of each unique value.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the categorical column to analyze.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (8, 4).
    - color (str, optional): The color palette for the bars. Defaults to 'mako'.
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Returns:
    - None: Displays a bar plot showing the counts of each unique value in the categorical column.
    """

    plt.figure(figsize=size)

    sns.countplot(
        x=col, 
        data=df[col].reset_index(),  
        palette=color, 
        order=df[col].value_counts().index,
        edgecolor="black"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel("")
    plt.ylabel("Number of registrations")

    plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, size = (5, 5)):
    """
    Plots the correlation matrix of numeric columns in a DataFrame as a heatmap.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (5, 5).

    Returns:
    - None: Displays a triangular heatmap of the correlation matrix with annotations.
    """

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=size)

    # Mask to make it triangular
    mask = np.triu(np.ones_like(corr_matrix, dtype = np.bool_))
    sns.heatmap(corr_matrix, 
                annot=True, 
                vmin = -1, 
                vmax = 1, 
                mask=mask)
    

def plot_relation_tv_numeric(df, tv, size = (15, 10)):
    """
    Plots scatter plots showing the relationship between a target variable and all numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - tv (str): The name of the target variable to analyze.
    - size (tuple, optional): The size of the entire plot grid as (width, height). Defaults to (15, 10).

    Returns:
    - None: Displays scatter plots of the target variable against each numeric column.
    """
    
    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = size)
    axes = axes.flat

    for i, col in enumerate(cols_num):

        if col == tv:
            fig.delaxes(axes[i])

        else:
            sns.scatterplot(x = col,
                        y = tv,
                        data = df_num,
                        ax = axes[i], 
                        palette = 'mako')
            
            axes[i].set_title(col)
            axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def plot_outliers(df):
    """
    Plots boxplots for all numeric columns in the DataFrame to visualize outliers.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    - None: Displays a grid of boxplots for each numeric column, showing potential outliers.
    """

    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_rows = math.ceil(n_plots/2)

    cmap = plt.cm.get_cmap('mako', n_plots)
    color_list = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize = (9, 5))
    axes = axes.flat

    for i, col in enumerate(cols_num):

        sns.boxplot(x = col, 
                    data = df_num,
                    ax = axes[i],
                    color=color_list[i]) 
        
        axes[i].set_title(f'{col} outliers')
        axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def value_counts(df, col):
    """
    Calculates the value counts and proportions for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the column to analyze.

    Returns:
    - (pd.DataFrame): A DataFrame with two columns: the counts and proportions (rounded to two decimal places) of each unique value in the specified column.
    """

    return pd.concat([df[col].value_counts(), df[col].value_counts(normalize=True).round(2)], axis=1)


def quick_plot_numeric(df, col, num=10, size=(10,5), rotation=45):
    """
    Generates a quick histogram plot for a numeric column, dividing the data into a specified number of bins.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the numeric column to plot.
    - num (int, optional): The number of bins to divide the range into. Defaults to 10.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (10, 5).
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Returns:
    - None: Displays a histogram of the numeric column if the bin width is greater than or equal to 2.
    """

    max_ = df[col].max()
    min_ = df[col].min()
    n = (max_ - min_) // num

    if n < 2:
        return
    
    plot_numeric_distribution(df, min_, max_, col, n, size=size, rotation=rotation)


def plot_groupby_median(df, groupby, col, max_entries, size=(12, 6), palette='mako'):
    """
    Plots the median values of a numeric column grouped by a specified categorical column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby (str): The name of the categorical column to group by.
    - col (str): The name of the numeric column for which medians are calculated.
    - max_entries (int): The maximum number of group entries to display.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (12, 6).
    - palette (str, optional): The color palette for the bar plot. Defaults to 'mako'.

    Returns:
    - None: Displays a bar plot of the median values per group.
    """
    
    median_by = (
        df.groupby(groupby)[col]
        .median()
        .sort_values()
        .reset_index()
    )

    plt.figure(figsize=size)
    sns.barplot(
        data=median_by.iloc[:max_entries],
        x=col,
        y=groupby,
        palette=palette,
        edgecolor="black"
    )

    plt.title(f'{col} median per {groupby}')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

    plt.show()


def plot_categorical_comparison(data, cat_var1, cat_var2, kind='count', normalize=False):
    """
    Genera un gráfico para comparar dos variables categóricas en un DataFrame.

    Parámetros:
    - data (DataFrame): El DataFrame que contiene los datos.
    - cat_var1 (str): El nombre de la primera variable categórica.
    - cat_var2 (str): El nombre de la segunda variable categórica.
    - kind (str): El tipo de gráfico ('count' para conteos, 'heatmap' para un mapa de calor).
    - normalize (bool): Si se debe normalizar para mostrar proporciones en lugar de conteos (solo para 'heatmap').

    """
    if kind == 'count':
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=cat_var1, hue=cat_var2)
        plt.title(f'Comparación de {cat_var1} y {cat_var2}')
        plt.xlabel(cat_var1)
        plt.ylabel('Frecuencia')
        plt.legend(title=cat_var2)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    elif kind == 'heatmap':
        # Crear una tabla cruzada de las variables
        crosstab = pd.crosstab(data[cat_var1], data[cat_var2])

        if normalize:
            crosstab = crosstab.div(crosstab.sum(axis=1), axis=0)  # Normalizar por fila

        plt.figure(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, fmt='.2f' if normalize else 'd', cmap='coolwarm')
        plt.title(f'Mapa de calor de {cat_var1} y {cat_var2}')
        plt.xlabel(cat_var2)
        plt.ylabel(cat_var1)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("El parámetro 'kind' debe ser 'count' o 'heatmap'")
    

def boxplot_grouped(data, numeric_column, categorical_column, title="Boxplot", xlabel=None, ylabel=None):
    """
    Crea un boxplot de una variable numérica agrupada por una categórica.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene los datos.
    - numeric_column (str): Nombre de la columna numérica.
    - categorical_column (str): Nombre de la columna categórica.
    - title (str): Título del gráfico.
    - xlabel (str): Etiqueta del eje X (opcional).
    - ylabel (str): Etiqueta del eje Y (opcional).
    """
    # Crear el gráfico de boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=categorical_column, y=numeric_column, data=data, palette="mako")

    # Configurar el título y etiquetas
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel if xlabel else categorical_column, fontsize=14)
    plt.ylabel(ylabel if ylabel else numeric_column, fontsize=14)

    # Rotar etiquetas del eje X si es necesario
    plt.xticks(rotation=45)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()