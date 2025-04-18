# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Para visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def get_index_from_value(col, col_value, dataframe):
 
    return dataframe[dataframe[col] == col_value].index[0]


def get_value_from_index(index, col, dataframe):
 
    return dataframe[dataframe.index == index][col].values[0]


def top_10_games(game, dataframe, similarity, plot=True):

    # buscamos el inidice del juego, ya que lo vamos a necesitar para nuestro objetivo
    game_index = get_index_from_value('name', game, dataframe)

    # sacamos los juegos similares
    similar_games = list(enumerate(similarity[game_index]))

    # ordenamos los juegos
    juegos_similares_ordenados = sorted(similar_games,key=lambda x:x[1],reverse=True)[1:11]

    # y ahora buscamos el título
    top_simiar_games = {}
    for i in juegos_similares_ordenados:
        top_simiar_games[get_value_from_index(i[0], 'name', dataframe)] = i[1]

    if plot:

        # visualizamos los resultados
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Crear gráfico de barras
        sns.barplot(
            x=list(top_simiar_games.values()), 
            y=list(top_simiar_games.keys()), 
            palette="mako"
        )

        # Añadir etiquetas y título
        plt.title(f"Top 10 Juegos Similares a {game} Basado en Contenido", fontsize=16, pad=20)
        plt.xlabel("Similitud", fontsize=12)
        plt.ylabel("Películas", fontsize=12)

        # Añadir valores al final de cada barra
        for i, value in enumerate(top_simiar_games.values()):
            plt.text(value + 0.02, i, f"{value:.2f}", va='center', fontsize=10)

        plt.tight_layout()

    return top_simiar_games


# Cambiar a partir de aquí

def plot(peli1, peli2, dataframe):
    """
    Genera un gráfico de dispersión que compara dos películas en un espacio de características.

    Parameters:
    ----------
    peli1 : str
        Nombre de la primera película a comparar.
    peli2 : str
        Nombre de la segunda película a comparar.
    dataframe : pd.DataFrame
        Un dataframe transpuesto donde las columnas representan películas y las filas características.

    Returns:
    -------
    None
        Muestra un gráfico de dispersión con anotaciones para cada película.
    """
    x = dataframe.T[peli1]     
    y = dataframe.T[peli2]

    n = list(dataframe.columns)    

    plt.figure(figsize=(10, 5))

    plt.scatter(x, y, s=0)      

    plt.title('Espacio para {} VS. {}'.format(peli1, peli2), fontsize=14)
    plt.xlabel(peli1, fontsize=14)
    plt.ylabel(peli2, fontsize=14)

    for i, e in enumerate(n):
        plt.annotate(e, (x[i], y[i]), fontsize=12)  

    plt.show();


def filter_data(df):
    """
    Filtra un dataframe de ratings basado en la frecuencia mínima de valoraciones por película y por usuario.

    Parameters:
    ----------
    df : pd.DataFrame
        Un dataframe con columnas 'movieId', 'userId' y 'rating'.

    Returns:
    -------
    pd.DataFrame
        Un dataframe filtrado que contiene solo las películas con al menos 300 valoraciones 
        y los usuarios con al menos 1500 valoraciones.
    """
    ## Ratings Per Movie
    ratings_per_movie = df.groupby('movieId')['rating'].count()
    ## Ratings By Each User
    ratings_per_user = df.groupby('userId')['rating'].count()

    ratings_per_movie_df = pd.DataFrame(ratings_per_movie)
    ratings_per_user_df = pd.DataFrame(ratings_per_user)

    filtered_ratings_per_movie_df = ratings_per_movie_df[ratings_per_movie_df.rating >= 300].index.tolist()
    filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df.rating >= 1500].index.tolist()
    
    df = df[df.movieId.isin(filtered_ratings_per_movie_df)]
    df = df[df.userId.isin(filtered_ratings_per_user_df)]
    return df