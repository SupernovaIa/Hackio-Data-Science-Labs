# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Para las visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

import math

# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

# Para visualizar los dendrogramas
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch


def clustering_metrics(df, labels):
    """
    Calculate clustering metrics and return them in a DataFrame.

    Parameters
    ----------
        - df (pd.DataFrame): The data used for clustering.
        - labels (array-like): Cluster labels assigned to each data point.

    Returns
    -------
        - (pd.DataFrame): A DataFrame containing the silhouette score, Davies-Bouldin index, and cluster cardinality for each cluster.
    """

    silhouette = silhouette_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)

    cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

    df_metrics = pd.DataFrame({
                        "silhouette_score": silhouette,
                        "davies_bouldin_index": davies_bouldin,
                        "cardinality": cardinality
                        })
    
    return df_metrics


def plot_combined_target_distribution(df, target, feature, size=(10, 6)):
    """
    Plots the combined distribution of a feature and the proportion of a binary target variable.

    This function creates a dual-axis histogram visualization. The primary axis shows the distribution of the specified feature, while the secondary axis overlays the proportion of the target variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to plot.
    - target (str): The name of the binary target column in the DataFrame.
    - feature (str): The name of the feature column to plot.
    - figsize (tuple, optional): Figure size for the plot. Defaults to (10, 6).

    Returns:
    - None: The function displays the plot but does not return any value.
    """

    fig, ax = plt.subplots(figsize=size)

    fig.suptitle(f"Proportion of '{target}' by '{feature}' distribution")
    
    # Total count histogram
    sns.histplot(data=df,
                x=feature,
                bins='auto',
                ax=ax)

    # Second histogram for positive class probability
    ax2 = ax.twinx()

    sns.histplot(data=df,
             x=feature,
             hue=target,  
             stat="probability", # Normalize to show proportion
             bins='auto',
             multiple="fill", # Get maximum in the top
             palette={1: "red", 0: "#FFFFFF"}, # Red for possitive, white for negative
             ax=ax2,
             alpha=0.25, # Transparency, important here
             edgecolor=None)

    # Set ylim for proportion
    ax2.set_ylim(0, 1)

    # Remove ax2 legend, we only want one
    ax2.get_legend().remove()
    fig.legend([f"{feature.capitalize()} distribution", f"{target.capitalize()} proportion"], loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_radar(df, columns):
    """
    Plots a radar chart to visualize the mean values of specified columns for each cluster.

    The function calculates the mean values of the provided columns grouped by the 'cluster' column and creates a radar chart to compare these means across clusters.

    Parameters
    ----------
        - df (pd.DataFrame): The DataFrame containing the data, including a 'cluster' column.
        - columns (list of str): A list of column names to include in the radar plot.

    Returns
    -------
        - None: The function displays the radar plot but does not return any value.
    """

    # Group by cluster and compute mean
    cluster_means = df.groupby('cluster')[columns].mean()

    # Repeat first column at the end to close radar
    cluster_means = pd.concat([cluster_means, cluster_means.iloc[:, 0:1]], axis=1)

    # Angles for radarplot
    num_vars = len(columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close radar

    _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # A radar for each cluster
    for i, row in cluster_means.iterrows():
        ax.plot(angles, row, label=f'Cluster {i}')
        ax.fill(angles, row, alpha=0.25)

    # Axis tags
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columns)

    # Legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Clusters radar plot')
    
    plt.tight_layout()
    plt.show()


def plot_clusters(df):
    """
    Plots bar charts showing the mean values of each feature grouped by clusters.

    The function iterates through the features in the DataFrame (excluding the 'cluster' column) and creates a subplot for each, displaying the mean value of the feature per cluster.

    Parameters
    ----------
        - df (pd.DataFrame): The DataFrame containing the data, including a 'cluster' column and other feature columns.

    Returns
    -------
        - None: The function displays the bar chart plots but does not return any value.
    """

    cols = df.drop(columns='cluster').columns.to_list()

    fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(cols)/2), figsize=(20, 8))
    axes = axes.flat

    for i, col in enumerate(cols):

        df_group = df.groupby('cluster')[col].mean().reset_index()

        sns.barplot(x='cluster', y=col, data=df_group, ax=axes[i], palette='coolwarm')
        axes[i].set_title(col, fontsize=18)

    if len(cols) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()


def plot_dendrogram(df, method_list=["average", "complete", "ward", "single"], size=(20, 8)):
    """
    Plots dendrograms using different linkage methods.

    This function generates a subplot for each specified linkage method to visualize the hierarchical clustering of the given DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame containing the data for clustering. Rows represent samples.
    - method_list (list of str, optional): A list of linkage methods to use for generating dendrograms. Defaults to ["average", "complete", "ward", "single"].
    - size (tuple, optional): The size of the entire figure. Defaults to (20, 8).

    Returns
    -------
    - None: The function displays the dendrograms but does not return any value.
    """
    _, axes = plt.subplots(nrows=1, ncols=len(method_list), figsize=size)
    axes = axes.flat

    for i, method in enumerate(method_list):

        sch.dendrogram(sch.linkage(df, method=method),
                        labels=df.index,
                        leaf_rotation=90, leaf_font_size=4,
                        ax=axes[i])
        
        axes[i].set_title(f'Dendrogram using {method}')
        axes[i].set_xlabel('Samples')
        axes[i].set_ylabel('Distance')


def evaluate_balance(cardinality):
    """
    Evaluate the balance of cluster sizes.

    This function computes the ratio of the largest cluster size to the smallest cluster size 
    from a given dictionary of cluster sizes. It is used to assess how evenly distributed 
    the clusters are. The closer the result is to 1, the more balanced the clusters. 

    If there is only one cluster or no clusters, the function returns `float('inf')` to 
    avoid penalizing such cases. Similarly, if any cluster has a size of zero, it also 
    returns `float('inf')` to avoid division by zero.

    Parameters
    ----------
    cardinality : dict
        A dictionary where keys represent cluster identifiers and values represent 
        the sizes of the clusters.

    Returns
    -------
    float
        The balance ratio defined as `max(cluster_sizes) / min(cluster_sizes)`. 
        Returns `float('inf')` if there are fewer than two clusters or if any cluster size is zero.
    """

    cluster_sizes = list(cardinality.values())

    # Avoid penalizing only one cluster
    if len(cluster_sizes) < 2:
        return float('inf')
    
    return max(cluster_sizes) / min(cluster_sizes) if min(cluster_sizes) > 0 else float('inf')


def agglomerative_methods(df, n_min=2, n_max=5, linkage_methods = ['complete',  'ward'], distance_metrics = ['euclidean', 'cosine', 'chebyshev']):
    """
    Performs Agglomerative Clustering using various linkage methods, distance metrics, and cluster counts, and evaluates the results.

    The function iteratively applies Agglomerative Clustering with specified configurations and computes performance metrics such as Silhouette Score and Davies-Bouldin Index for each clustering result. Results are returned as a DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): The input data for clustering, where rows represent samples and columns represent features.
    - n_min (int, optional): The minimum number of clusters to evaluate. Defaults to 2.
    - n_max (int, optional): The maximum number of clusters to evaluate. Defaults to 5.
    - linkage_methods (list of str, optional): A list of linkage methods for clustering. Defaults to ['complete', 'ward'].
    - distance_metrics (list of str, optional): A list of distance metrics to use. Defaults to ['euclidean', 'cosine', 'chebyshev'].

    Returns
    -------
    - pd.DataFrame: A DataFrame containing the results with columns for linkage method, metric, silhouette score, Davies-Bouldin Index, cluster cardinality, and number of clusters.
    """

    # Results storage
    results = []

    for linkage_method in linkage_methods:
        for metric in distance_metrics:
            for cluster in range(n_min,n_max+1):

                try:
                    # Config AgglomerativeClustering model
                    modelo = AgglomerativeClustering(
                        linkage=linkage_method,
                        metric=metric,  
                        distance_threshold=None,  # We use n_clusters
                        n_clusters=cluster,
                    )
                    
                    # Model fit
                    labels = modelo.fit_predict(df)

                    # Initialize metrics
                    silhouette_avg, db_score = None, None

                    # Get metrics if we have more than one cluster
                    if len(np.unique(labels)) > 1:

                        # Silhouette Score
                        silhouette_avg = silhouette_score(df, labels, metric=metric)

                        # Davies-Bouldin Index
                        db_score = davies_bouldin_score(df, labels)

                        # Cardinality
                        cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

                    # If only one cluster
                    else:
                        cluster_cardinality = {'Unique cluster': len(df)}

                    # Store results
                    results.append({
                        'linkage': linkage_method,
                        'metric': metric,
                        'silhouette_score': silhouette_avg,
                        'davies_bouldin_index': db_score,
                        'cluster_cardinality': cluster_cardinality,
                        'n_cluster': cluster
                    })

                except Exception as e:
                    print(f"Error with linkage={linkage_method}, metric={metric}: {e}")

    # Build results dataframe
    results_df = pd.DataFrame(results)
    results_df['balance_score'] = results_df['cluster_cardinality'].apply(evaluate_balance)

    # Ranking based in all metrics
    results_df['ranking_score'] = (
        results_df['silhouette_score'] -  # Max
        results_df['davies_bouldin_index'] -  # Min
        results_df['balance_score']  # Min
    )

    return results_df


def spectral_methods(df, n_clusters_list = [2, 3, 4, 5], assign_labels_options = ["kmeans", "discretize"]):
    """
    Performs Spectral Clustering with various configurations and evaluates the results.

    The function applies Spectral Clustering using different numbers of clusters and label assignment methods, then computes performance metrics such as Silhouette Score and Davies-Bouldin Index for each configuration. Results are returned as a DataFrame.

    Parameters
    ----------
        - df (pd.DataFrame): The input data for clustering, where rows represent samples and columns represent features.
        - n_clusters_list (list of int, optional): A list of the numbers of clusters to evaluate. Defaults to [2, 3, 4, 5].
        - assign_labels_options (list of str, optional): A list of label assignment methods to use. Defaults to ["kmeans", "discretize"].

    Returns
    -------
        - pd.DataFrame: A DataFrame containing the results with columns for number of clusters, label assignment method, Silhouette Score, Davies-Bouldin Index, and cluster cardinality.
    """

    # Results storage
    results = []

    for n_clusters in n_clusters_list:
        for assign_labels in assign_labels_options:

            # Config SpectralClustering model
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                assign_labels=assign_labels,
                random_state=42
            )

            # Model fit
            labels = spectral.fit_predict(df)

            # Initialize metrics
            silhouette, db_score = None, None

            # Get metrics if we have more than one cluster
            if len(np.unique(labels)) > 1:

                # Silhouette Score
                silhouette = silhouette_score(df, labels)

                # Davies-Bouldin Index
                db_score = davies_bouldin_score(df, labels)

                # Cardinality
                cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

            # If only one cluster
            else:
                cluster_cardinality = {'Unique cluster': len(df)}
                    
            # Save results
            results.append({
                "n_clusters": n_clusters,
                "assign_labels": assign_labels,
                "silhouette_score": silhouette,
                "davies_bouldin_score": db_score,
                "cardinality": cluster_cardinality
            })

    # Return metrics
    return pd.DataFrame(results)


def dbscan_methods(df, eps_values=[1, 2, 3, 4, 5], min_samples_values=[5, 10, 15, 20]):
    """
    Applies the DBSCAN clustering algorithm to a given dataset for a range of `eps` and `min_samples` parameter values, evaluating clustering performance using silhouette score and Davies-Bouldin index.

    Parameters
    ----------
    - df (pd.DataFrame): The input dataset to cluster.
    - eps_values (list of float, optional): A list of epsilon values for DBSCAN. Default is [1, 2, 3, 4, 5].
    - min_samples_values (list of int, optional): A list of minimum samples values for DBSCAN. Default is [5, 10, 15, 20].

    Returns
    -------
    - (pd.DataFrame): A dataframe containing the results of clustering for each combination of `eps` and `min_samples`, including silhouette score, Davies-Bouldin index, and cluster cardinality.
    """

    # Results storage
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:

            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

            # Model fit
            labels = dbscan.fit_predict(df)

            # Initialize metrics
            silhouette, db_score = None, None

            # Compute metrics ignoring noise
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):

                # Silhouette Score
                silhouette = silhouette_score(df, labels)

                # Davies-Bouldin Index
                db_score = davies_bouldin_score(df, labels)

                # Cardinality
                cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}

            else:
                silhouette = -1
                cluster_cardinality = {'Unique cluster': len(df)}

            # Save results
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "silhouette_score": silhouette,
                "davies_bouldin_score": db_score,
                "cardinality": cluster_cardinality
            })

    # Return metrics
    return pd.DataFrame(results).sort_values(by="silhouette_score", ascending=False)






class Clustering:
    """
    Clase para realizar varios métodos de clustering en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos sobre el cual se aplicarán los métodos de clustering.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Clustering con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a los que se les aplicarán los métodos de clustering.
        """
        self.dataframe = dataframe
    
    def sacar_clusters_kmeans(self, n_clusters=(2, 15)):
        """
        Utiliza KMeans y KElbowVisualizer para determinar el número óptimo de clusters basado en la métrica de silhouette.

        Params:
            - n_clusters : tuple of int, optional, default: (2, 15). Rango de número de clusters a probar.
        
        Returns:
            None
        """
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=n_clusters, metric='silhouette')
        visualizer.fit(self.dataframe)
        visualizer.show()
    
    def modelo_kmeans(self, dataframe_original, num_clusters):
        """
        Aplica KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters)
        km_fit = kmeans.fit(self.dataframe)
        labels = km_fit.labels_
        dataframe_original["clusters_kmeans"] = labels.astype(str)
        return dataframe_original, labels
    
    def visualizar_dendrogramas(self, lista_metodos=["average", "complete", "ward"]):
        """
        Genera y visualiza dendrogramas para el conjunto de datos utilizando diferentes métodos de distancias.

        Params:
            - lista_metodos : list of str, optional, default: ["average", "complete", "ward"]. Lista de métodos para calcular las distancias entre los clusters. Cada método generará un dendrograma
                en un subplot diferente.

        Returns:
            None
        """
        _, axes = plt.subplots(nrows=1, ncols=len(lista_metodos), figsize=(20, 8))
        axes = axes.flat

        for indice, metodo in enumerate(lista_metodos):
            sch.dendrogram(sch.linkage(self.dataframe, method=metodo),
                           labels=self.dataframe.index, 
                           leaf_rotation=90, leaf_font_size=4,
                           ax=axes[indice])
            axes[indice].set_title(f'Dendrograma usando {metodo}')
            axes[indice].set_xlabel('Muestras')
            axes[indice].set_ylabel('Distancias')
    
    def modelo_aglomerativo(self, num_clusters, metodo_distancias, dataframe_original):
        """
        Aplica clustering aglomerativo al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - num_clusters : int. Número de clusters a formar.
            - metodo_distancias : str. Método para calcular las distancias entre los clusters.
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        modelo = AgglomerativeClustering(
            linkage=metodo_distancias,
            distance_threshold=None,
            n_clusters=num_clusters
        )
        aglo_fit = modelo.fit(self.dataframe)
        labels = aglo_fit.labels_
        dataframe_original["clusters_agglomerative"] = labels.astype(str)
        return dataframe_original
    
    def modelo_divisivo(self, dataframe_original, threshold=0.5, max_clusters=5):
        """
        Implementa el clustering jerárquico divisivo.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - threshold : float, optional, default: 0.5. Umbral para decidir cuándo dividir un cluster.
            - max_clusters : int, optional, default: 5. Número máximo de clusters deseados.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de los clusters.
        """
        def divisive_clustering(data, current_cluster, cluster_labels):
            # Si el número de clusters actuales es mayor o igual al máximo permitido, detener la división
            if len(set(current_cluster)) >= max_clusters:
                return current_cluster

            # Aplicar KMeans con 2 clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calcular la métrica de silueta para evaluar la calidad del clustering
            silhouette_avg = silhouette_score(data, labels)

            # Si la calidad del clustering es menor que el umbral o si el número de clusters excede el máximo, detener la división
            if silhouette_avg < threshold or len(set(current_cluster)) + 1 > max_clusters:
                return current_cluster

            # Crear nuevas etiquetas de clusters
            new_cluster_labels = current_cluster.copy()
            max_label = max(current_cluster)

            # Asignar nuevas etiquetas incrementadas para cada subcluster
            for label in set(labels):
                cluster_indices = np.where(labels == label)[0]
                new_label = max_label + 1 + label
                new_cluster_labels[cluster_indices] = new_label

            # Aplicar recursión para seguir dividiendo los subclusters
            for new_label in set(new_cluster_labels):
                cluster_indices = np.where(new_cluster_labels == new_label)[0]
                new_cluster_labels = divisive_clustering(data[cluster_indices], new_cluster_labels, new_cluster_labels)

            return new_cluster_labels

        # Inicializar las etiquetas de clusters con ceros
        initial_labels = np.zeros(len(self.dataframe))

        # Llamar a la función recursiva para iniciar el clustering divisivo
        final_labels = divisive_clustering(self.dataframe.values, initial_labels, initial_labels)

        # Añadir las etiquetas de clusters al DataFrame original
        dataframe_original["clusters_divisive"] = final_labels.astype(int).astype(str)

        return dataframe_original

    def modelo_espectral(self, dataframe_original, n_clusters=3, assign_labels='kmeans'):
        """
        Aplica clustering espectral al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - n_clusters : int, optional, default: 3. Número de clusters a formar.
            - assign_labels : str, optional, default: 'kmeans'. Método para asignar etiquetas a los puntos. Puede ser 'kmeans' o 'discretize'.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels, random_state=0)
        labels = spectral.fit_predict(self.dataframe)
        dataframe_original["clusters_spectral"] = labels.astype(str)
        return dataframe_original
    
    def modelo_dbscan(self, dataframe_original, eps_values=[0.5, 1.0, 1.5], min_samples_values=[3, 2, 1]):
        """
        Aplica DBSCAN al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - eps_values : list of float, optional, default: [0.5, 1.0, 1.5]. Lista de valores para el parámetro eps de DBSCAN.
            - min_samples_values : list of int, optional, default: [3, 2, 1]. Lista de valores para el parámetro min_samples de DBSCAN.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        best_eps = None
        best_min_samples = None
        best_silhouette = -1  # Usamos -1 porque la métrica de silueta varía entre -1 y 1

        # Iterar sobre diferentes combinaciones de eps y min_samples
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Aplicar DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.dataframe)

                # Calcular la métrica de silueta, ignorando etiquetas -1 (ruido)
                if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                    silhouette = silhouette_score(self.dataframe, labels)
                else:
                    silhouette = -1

                # Mostrar resultados (opcional)
                print(f"eps: {eps}, min_samples: {min_samples}, silhouette: {silhouette}")

                # Actualizar el mejor resultado si la métrica de silueta es mejor
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples

        # Aplicar DBSCAN con los mejores parámetros encontrados
        best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        best_labels = best_dbscan.fit_predict(self.dataframe)

        # Añadir los labels al DataFrame original
        dataframe_original["clusters_dbscan"] = best_labels

        return dataframe_original

    def calcular_metricas(self, labels: np.ndarray):
        """
        Calcula métricas de evaluación del clustering.
        """
        if len(set(labels)) <= 1:
            raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

        silhouette = silhouette_score(self.dataframe, labels)
        davies_bouldin = davies_bouldin_score(self.dataframe, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cardinalidad = dict(zip(unique, counts))

        return pd.DataFrame({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "cardinalidad": cardinalidad
        })