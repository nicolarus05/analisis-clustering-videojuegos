"""
Módulo de visualización para clustering y regresión.

Este módulo contiene funciones para:
- Visualización interactiva con Plotly
- Gráficos de clustering (PCA, distribuciones)
- Gráficos de regresión (predicciones, residuos)
- Mapas de calor y correlaciones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_pca_clusters(df_pca, pca, cluster_col='Cluster', output_path=None):
    """
    Visualización interactiva de clusters con PCA usando Plotly.
    
    Parameters:
    -----------
    df_pca : pd.DataFrame
        DataFrame con componentes PCA y clusters
    pca : PCA
        Modelo PCA ajustado
    cluster_col : str, default='Cluster'
        Nombre de la columna de clusters
    output_path : str, optional
        Ruta para guardar HTML
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly
    
    Example:
    --------
    >>> fig = plot_pca_clusters(df_pca, pca, output_path='results/figures/pca_clusters.html')
    """
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color=df_pca[cluster_col].astype(str),
        title=f'Segmentación de Videojuegos - Clustering K-Means + PCA',
        labels={
            'PC1': f'Componente Principal 1 ({var1:.1f}% varianza)',
            'PC2': f'Componente Principal 2 ({var2:.1f}% varianza)',
            'color': 'Cluster'
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        width=1000,
        height=600
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.6, line=dict(width=0.5, color='white'))
    )
    
    fig.update_layout(
        font=dict(size=12),
        title_font=dict(size=16, family='Arial', color='#2c3e50'),
        plot_bgcolor='rgba(240,240,240,0.5)',
        legend_title_text='Cluster',
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"💾 Visualización guardada: {output_path}")
    
    fig.show()
    return fig


def plot_cluster_distribution(df, cluster_col='Cluster', save_path=None):
    """
    Visualizar distribución de videojuegos por cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con clusters
    cluster_col : str
        Nombre de la columna de clusters
    save_path : str, optional
        Ruta para guardar imagen
    """
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    
    # Añadir etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Videojuegos', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Videojuegos por Cluster', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(cluster_counts.index)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_sales_by_cluster(df, cluster_col='Cluster', save_path=None):
    """
    Visualizar ventas promedio por cluster y región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con datos de ventas y clusters
    cluster_col : str
        Nombre de columna de clusters
    save_path : str, optional
        Ruta para guardar imagen
    """
    sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    
    # Calcular promedios por cluster
    df_avg = df.groupby(cluster_col)[sales_cols].mean()
    
    # Plot
    ax = df_avg.plot(kind='bar', figsize=(12, 6), width=0.8, 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                      alpha=0.8)
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ventas Promedio (Millones)', fontsize=12, fontweight='bold')
    ax.set_title('Ventas Promedio por Cluster y Región', 
                 fontsize=14, fontweight='bold')
    ax.legend(['Norteamérica', 'Europa', 'Japón', 'Otras'], 
              title='Región', loc='upper right')
    ax.set_xticklabels(df_avg.index, rotation=0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df, features, save_path=None):
    """
    Visualizar matriz de correlación de features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con features
    features : list
        Lista de features a correlacionar
    save_path : str, optional
        Ruta para guardar imagen
    """
    corr_matrix = df[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, center=0)
    
    plt.title('Matriz de Correlación entre Features', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, cluster_id, model_name, save_path=None):
    """
    Gráfico de valores reales vs predichos.
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Valores predichos
    cluster_id : int
        ID del cluster
    model_name : str
        Nombre del modelo
    save_path : str, optional
        Ruta para guardar imagen
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Línea de referencia perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Predicción Perfecta')
    
    plt.xlabel('Valores Reales (Global Sales)', fontsize=12, fontweight='bold')
    plt.ylabel('Valores Predichos', fontsize=12, fontweight='bold')
    plt.title(f'Predicción vs Real - Cluster {cluster_id} ({model_name})', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, cluster_id, model_name, save_path=None):
    """
    Gráfico de residuos.
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Valores predichos
    cluster_id : int
        ID del cluster
    model_name : str
        Nombre del modelo
    save_path : str, optional
        Ruta para guardar imagen
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residuos vs Predicciones
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Valores Predichos', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuos', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Residuos vs Predicciones - Cluster {cluster_id}', 
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histograma de residuos
    axes[1].hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuos', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Distribución de Residuos ({model_name})', 
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_3d_clusters(df_pca, pca, cluster_col='Cluster', output_path=None):
    """
    Visualización 3D de clusters (si se tienen 3+ componentes PCA).
    
    Parameters:
    -----------
    df_pca : pd.DataFrame
        DataFrame con al menos PC1, PC2, PC3
    pca : PCA
        Modelo PCA con 3+ componentes
    cluster_col : str
        Nombre de columna de clusters
    output_path : str, optional
        Ruta para guardar HTML
    """
    if 'PC3' not in df_pca.columns:
        print("⚠️  Se requiere PC3 para visualización 3D")
        return
    
    fig = px.scatter_3d(
        df_pca,
        x='PC1',
        y='PC2',
        z='PC3',
        color=df_pca[cluster_col].astype(str),
        title='Visualización 3D de Clusters (PCA)',
        labels={'color': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        width=900,
        height=700
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    if output_path:
        fig.write_html(output_path)
        print(f"💾 Visualización 3D guardada: {output_path}")
    
    fig.show()
    return fig


def create_dashboard_summary(df, df_results, cluster_col='Cluster'):
    """
    Crear dashboard resumen con múltiples gráficos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    df_results : pd.DataFrame
        Resultados de regresión
    cluster_col : str
        Nombre de columna de clusters
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución por Cluster', 
                       'R² por Cluster',
                       'Ventas Promedio por Región',
                       'RMSE por Cluster'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Plot 1: Distribución
    cluster_counts = df[cluster_col].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=cluster_counts.index, y=cluster_counts.values, 
               name='Juegos', marker_color='steelblue'),
        row=1, col=1
    )
    
    # Plot 2: R²
    fig.add_trace(
        go.Bar(x=df_results['Cluster'], y=df_results['R2_RandomForest'],
               name='R² RF', marker_color='forestgreen'),
        row=1, col=2
    )
    
    # Plot 3: Ventas por región
    sales_avg = df.groupby(cluster_col)['Global_Sales'].mean()
    fig.add_trace(
        go.Bar(x=sales_avg.index, y=sales_avg.values,
               name='Ventas Global', marker_color='coral'),
        row=2, col=1
    )
    
    # Plot 4: RMSE
    fig.add_trace(
        go.Bar(x=df_results['Cluster'], y=df_results['RMSE_RandomForest'],
               name='RMSE RF', marker_color='mediumpurple'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, showlegend=False,
                      title_text="Dashboard Resumen - Clustering y Regresión")
    fig.show()
    
    return fig


if __name__ == "__main__":
    print("📦 Módulo visualization.py cargado correctamente")
    print("Funciones disponibles:")
    print("  - plot_pca_clusters()")
    print("  - plot_cluster_distribution()")
    print("  - plot_sales_by_cluster()")
    print("  - plot_actual_vs_predicted()")
    print("  - plot_residuals()")
