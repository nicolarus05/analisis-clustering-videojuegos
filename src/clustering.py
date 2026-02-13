"""
Módulo de clustering y análisis de componentes principales (PCA).

Este módulo contiene funciones para:
- Método del codo (Elbow Method)
- K-Means clustering
- Análisis de componentes principales (PCA)
- Análisis de centroides
- Caracterización de clusters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


def elbow_method(df_scaled, max_k=10, random_state=42):
    """
    Calcular WCSS para el método del codo.
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        DataFrame con datos normalizados
    max_k : int, default=10
        Número máximo de clusters a probar
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Returns:
    --------
    tuple
        (K_range, wcss, silhouette_scores)
    
    Example:
    --------
    >>> K wcss, sil_scores = elbow_method(df_scaled, max_k=10)
    """
    wcss = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)  # Empezar desde 2 para silhouette
    
    print("\n" + "="*70)
    print("📊 MÉTODO DEL CODO (Elbow Method)")
    print("="*70)
    print(f"\nProbando k desde 2 hasta {max_k}...\n")
    
    # Calcular para k=1 solo WCSS
    kmeans_1 = KMeans(n_clusters=1, random_state=random_state, n_init=10)
    kmeans_1.fit(df_scaled)
    wcss_1 = kmeans_1.inertia_
    
    # Calcular para k >= 2
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(df_scaled)
        
        wcss_k = kmeans.inertia_
        wcss.append(wcss_k)
        
        # Calcular silhouette score
        labels = kmeans.labels_
        sil_score = silhouette_score(df_scaled, labels)
        silhouette_scores.append(sil_score)
        
        print(f"k={k:2d} | WCSS={wcss_k:10.2f} | Silhouette={sil_score:.4f}")
    
    # Añadir k=1 al principio
    wcss = [wcss_1] + wcss
    K_range_full = range(1, max_k + 1)
    
    print("\n✅ Cálculo completado")
    
    return K_range_full, wcss, silhouette_scores


def plot_elbow(K_range, wcss, save_path=None):
    """
    Visualizar el método del codo.
    
    Parameters:
    -----------
    K_range : range
        Rango de valores de k
    wcss : list
        Lista de valores WCSS
    save_path : str, optional
        Ruta donde guardar la figura
    
    Example:
    --------
    >>> plot_elbow(K_range, wcss, save_path='results/figures/elbow_method.png')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(K_range), wcss, 'bo-', linewidth=2, markersize=8)
    
    plt.xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    plt.title('Método del Codo - Determinación de k Óptimo', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(list(K_range))
    
    # Añadir anotaciones
    for i, (k, w) in enumerate(zip(K_range, wcss)):
        if k % 2 == 0:  # Etiquetar solo valores pares
            plt.annotate(f'{w:.0f}', 
                        xy=(k, w), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_silhouette_scores(K_range, silhouette_scores, save_path=None):
    """
    Visualizar Silhouette Scores por k.
    
    Parameters:
    -----------
    K_range : range
        Rango de valores de k (desde 2)
    silhouette_scores : list
        Lista de silhouette scores
    save_path : str, optional
        Ruta donde guardar la figura
    """
    plt.figure(figsize=(10, 6))
    K_list = list(K_range)[1:]  # Empezar desde k=2
    plt.plot(K_list, silhouette_scores, 'go-', linewidth=2, markersize=8)
    
    plt.xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score por Número de Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(K_list)
    
    # Marcar el máximo
    max_idx = np.argmax(silhouette_scores)
    max_k = K_list[max_idx]
    max_score = silhouette_scores[max_idx]
    plt.axvline(x=max_k, color='r', linestyle='--', alpha=0.5, label=f'Máximo: k={max_k}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {save_path}")
    
    plt.show()


def apply_kmeans(df_scaled, n_clusters, random_state=42):
    """
    Aplicar K-Means clustering.
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        DataFrame con datos normalizados
    n_clusters : int
        Número de clusters
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Returns:
    --------
    tuple
        (clusters, kmeans_model)
    
    Example:
    --------
    >>> clusters, kmeans = apply_kmeans(df_scaled, n_clusters=4)
    """
    print(f"\n🎯 Aplicando K-Means con k={n_clusters}...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    clusters = kmeans.fit_predict(df_scaled)
    
    # Métricas de evaluación
    wcss = kmeans.inertia_
    silhouette = silhouette_score(df_scaled, clusters)
    davies_bouldin = davies_bouldin_score(df_scaled, clusters)
    
    print(f"\n📊 Métricas de clustering:")
    print(f"   - WCSS: {wcss:.2f}")
    print(f"   - Silhouette Score: {silhouette:.4f} (rango: -1 a 1, mejor si > 0.5)")
    print(f"   - Davies-Bouldin Index: {davies_bouldin:.4f} (mejor si < 1)")
    
    # Distribución de clusters
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"\n📈 Distribución de videojuegos por cluster:")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(clusters)) * 100
        print(f"   Cluster {cluster_id}: {count:5d} juegos ({pct:5.2f}%)")
    
    print(f"\n✅ Clustering completado")
    
    return clusters, kmeans


def apply_pca(df_scaled, n_components=2):
    """
    Aplicar PCA para reducción dimensional.
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        DataFrame con datos normalizados
    n_components : int, default=2
        Número de componentes principales
    
    Returns:
    --------
    tuple
        (pca_components, pca_model)
    
    Example:
    --------
    >>> pca_components, pca = apply_pca(df_scaled, n_components=2)
    """
    print(f"\n🔍 Aplicando PCA con {n_components} componentes...")
    
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df_scaled)
    
    print(f"\n📊 Varianza explicada por cada componente:")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"   PC{i}: {var*100:6.2f}%")
    
    total_var = sum(pca.explained_variance_ratio_) * 100
    print(f"   {'─'*30}")
    print(f"   Total: {total_var:6.2f}%")
    
    # Mostrar loadings (contribución de cada feature)
    if hasattr(df_scaled, 'columns'):
        print(f"\n📋 Loadings (contribución de features):")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=df_scaled.columns
        )
        print(loadings.round(3))
    
    print(f"\n✅ PCA completado")
    
    return pca_components, pca


def get_cluster_centroids(kmeans, scaler, feature_names):
    """
    Obtener centroides en escala original.
    
    Parameters:
    -----------
    kmeans : KMeans
        Modelo K-Means ajustado
    scaler : StandardScaler
        Escalador utilizado para normalización
    feature_names : list
        Nombres de las features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con los centroides
    
    Example:
    --------
    >>> centroids = get_cluster_centroids(kmeans, scaler, feature_names)
    """
    # Invertir normalización
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Crear DataFrame
    df_centroids = pd.DataFrame(
        centroids_original,
        columns=feature_names
    )
    df_centroids.insert(0, 'Cluster', range(len(df_centroids)))
    
    print("\n" + "="*70)
    print("📍 CENTROIDES DE CADA CLUSTER (Valores Originales)")
    print("="*70)
    print(df_centroids.to_string(index=False))
    print()
    
    return df_centroids


def characterize_clusters(df, cluster_col='Cluster', n_clusters=None):
    """
    Caracterizar cada cluster con estadísticas descriptivas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo con columna de clusters
    cluster_col : str, default='Cluster'
        Nombre de la columna de clusters
    n_clusters : int, optional
        Número de clusters. Si None, se detecta automáticamente.
    
    Example:
    --------
    >>> characterize_clusters(df_clean, cluster_col='Cluster')
    """
    if n_clusters is None:
        n_clusters = df[cluster_col].nunique()
    
    print("\n" + "="*70)
    print("🔍 CARACTERIZACIÓN DE CLUSTERS")
    print("="*70)
    
    for cluster_id in range(n_clusters):
        cluster_data = df[df[cluster_col] == cluster_id]
        n_games = len(cluster_data)
        pct = (n_games / len(df)) * 100
        
        print(f"\n{'─'*70}")
        print(f"🎮 CLUSTER {cluster_id} | {n_games:,} juegos ({pct:.1f}%)")
        print(f"{'─'*70}")
        
        # Ventas promedio
        print(f"\n💰 Ventas Promedio (millones):")
        print(f"   • Global:      ${cluster_data['Global_Sales'].mean():7.2f}M")
        print(f"   • Norteamérica: ${cluster_data['NA_Sales'].mean():7.2f}M")
        print(f"   • Europa:       ${cluster_data['EU_Sales'].mean():7.2f}M")
        print(f"   • Japón:        ${cluster_data['JP_Sales'].mean():7.2f}M")
        print(f"   • Otras:        ${cluster_data['Other_Sales'].mean():7.2f}M")
        
        # Año promedio
        if 'Year' in cluster_data.columns:
            print(f"\n📅 Año promedio: {cluster_data['Year'].mean():.0f}")
        
        # Top géneros
        if 'Genre' in cluster_data.columns:
            print(f"\n🎯 Top 3 Géneros:")
            top_genres = cluster_data['Genre'].value_counts().head(3)
            for i, (genre, count) in enumerate(top_genres.items(), 1):
                pct_genre = (count / n_games) * 100
                print(f"   {i}. {genre:15s} ({count:4d} juegos, {pct_genre:4.1f}%)")
        
        # Top plataformas
        if 'Platform' in cluster_data.columns:
            print(f"\n🎮 Top 3 Plataformas:")
            top_platforms = cluster_data['Platform'].value_counts().head(3)
            for i, (platform, count) in enumerate(top_platforms.items(), 1):
                pct_platform = (count / n_games) * 100
                print(f"   {i}. {platform:15s} ({count:4d} juegos, {pct_platform:4.1f}%)")
        
        # Top juegos
        if 'Name' in cluster_data.columns and 'Global_Sales' in cluster_data.columns:
            print(f"\n🏆 Top 3 Juegos:")
            top_games = cluster_data.nlargest(3, 'Global_Sales')[['Name', 'Global_Sales']]
            for i, (idx, row) in enumerate(top_games.iterrows(), 1):
                print(f"   {i}. {row['Name'][:40]:40s} (${row['Global_Sales']:.2f}M)")


def suggest_optimal_k(wcss, silhouette_scores, K_range):
    """
    Sugerir número óptimo de clusters basado en métricas.
    
    Parameters:
    -----------
    wcss : list
        Lista de WCSS values
    silhouette_scores : list
        Lista de silhouette scores
    K_range : range
        Rango de k evaluados
    
    Returns:
    --------
    int
        Número de clusters sugerido
    """
    # Método del codo: buscar el "codo" usando segunda derivada
    wcss_array = np.array(wcss[1:])  # Excluir k=1
    k_array = np.array(list(K_range)[1:])
    
    # Calcular segunda derivada
    first_deriv = np.diff(wcss_array)
    second_deriv = np.diff(first_deriv)
    
    # El codo está donde la segunda derivada es máxima
    elbow_idx = np.argmax(second_deriv) + 2  # +2 por los diffs
    k_elbow = k_array[elbow_idx - 1] if elbow_idx < len(k_array) else k_array[-1]
    
    # Mejor silhouette score
    k_silhouette = k_array[np.argmax(silhouette_scores)]
    
    print(f"\n💡 Sugerencias de k óptimo:")
    print(f"   • Método del codo:    k = {k_elbow}")
    print(f"   • Mejor Silhouette:   k = {k_silhouette}")
    print(f"   • Recomendación:      k = {k_silhouette} (prioriza cohesión)")
    
    return k_silhouette


if __name__ == "__main__":
    print("📦 Módulo clustering.py cargado correctamente")
    print("Funciones disponibles:")
    print("  - elbow_method()")
    print("  - apply_kmeans()")
    print("  - apply_pca()")
    print("  - characterize_clusters()")
