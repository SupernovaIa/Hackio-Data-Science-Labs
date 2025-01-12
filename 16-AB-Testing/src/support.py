from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def normalidad_ks(df):

    data_mean = df.mean()
    data_std = df.std()
    
    return stats.kstest(df, 'norm', args=(data_mean, data_std))


def test_no_param(datos, dependencia=False):

    if len(datos) > 2:
        print('Test Kruskal Wallis')
        return stats.kruskal(*datos)

    elif len(datos) == 2 and dependencia:
        print('Wilcoxon')
        return stats.wilcoxon(*datos)

    elif len(datos) == 2 and not dependencia:
        print('Test U')
        return stats.mannwhitneyu(*datos)

    else:
        print('Tienes que pasar al menos 2 grupos')


def diagrama_barras(df, columna):

    df_ciudades = df.groupby('Loyalty Number')[columna].unique().value_counts().reset_index()
    df_ciudades[columna] = df_ciudades[columna].apply(lambda x: x[0])

    plt.figure(figsize=(8, 4))

    sns.barplot(x=columna, y='count', data=df_ciudades, palette='mako', edgecolor='black', linewidth=1.5)

    plt.title(f'Personas por {columna}')
    plt.xlabel('')
    plt.ylabel('Número de personas')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()