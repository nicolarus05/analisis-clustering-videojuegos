"""
Módulo de modelos de regresión por cluster.

Este módulo contiene funciones para:
- Entrenar modelos de regresión por cluster
- Evaluar performance de modelos
- Comparar modelos (Linear Regression vs Random Forest)
- Visualizar residuos y predicciones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def train_regression_by_cluster(df, features, target, cluster_col='Cluster', 
                                  test_size=0.2, random_state=42):
    """
    Entrenar modelos de regresión para cada cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo con clusters asignados
    features : list
        Lista de nombres de features para X
    target : str
        Nombre de la columna target (y)
    cluster_col : str, default='Cluster'
        Nombre de la columna de clusters
    test_size : float, default=0.2
        Proporción de datos para test
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con resultados por cluster
    
    Example:
    --------
    >>> features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
    >>> results = train_regression_by_cluster(df, features, 'Global_Sales')
    """
    n_clusters = df[cluster_col].nunique()
    results = []
    models = {}
    
    print("\n" + "="*70)
    print("🤖 ENTRENAMIENTO DE MODELOS DE REGRESIÓN POR CLUSTER")
    print("="*70)
    print(f"\nFeatures: {features}")
    print(f"Target: {target}")
    print(f"Test size: {test_size*100:.0f}%\n")
    
    for cluster_id in range(n_clusters):
        print(f"{'─'*70}")
        print(f"📊 Cluster {cluster_id}")
        print(f"{'─'*70}")
        
        # Filtrar datos del cluster
        cluster_mask = df[cluster_col] == cluster_id
        X_cluster = df[cluster_mask][features]
        y_cluster = df[cluster_mask][target]
        
        n_samples = len(X_cluster)
        print(f"Muestras: {n_samples}")
        
        # Verificar mínimo de datos
        if n_samples < 20:
            print(f"⚠️  Cluster {cluster_id}: Datos insuficientes (< 20). Omitiendo...\n")
            continue
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X_cluster, y_cluster, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # ===== MODELO 1: REGRESIÓN LINEAL =====
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        
        print(f"\n📈 Regresión Lineal:")
        print(f"   R²:   {r2_lr:6.4f}")
        print(f"   RMSE: {rmse_lr:6.4f}")
        print(f"   MAE:  {mae_lr:6.4f}")
        
        # ===== MODELO 2: RANDOM FOREST =====
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        
        print(f"\n🌲 Random Forest:")
        print(f"   R²:   {r2_rf:6.4f}")
        print(f"   RMSE: {rmse_rf:6.4f}")
        print(f"   MAE:  {mae_rf:6.4f}")
        
        # Determinar mejor modelo
        best_model = 'Random Forest' if r2_rf > r2_lr else 'Linear Regression'
        print(f"\n🏆 Mejor modelo: {best_model}")
        print()
        
        # Guardar resultados
        results.append({
            'Cluster': cluster_id,
            'n_samples': n_samples,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'R2_Linear': r2_lr,
            'RMSE_Linear': rmse_lr,
            'MAE_Linear': mae_lr,
            'R2_RandomForest': r2_rf,
            'RMSE_RandomForest': rmse_rf,
            'MAE_RandomForest': mae_rf,
            'Best_Model': best_model
        })
        
        # Guardar modelos
        models[cluster_id] = {
            'linear': lr_model,
            'random_forest': rf_model
        }
    
    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    print("="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    return df_results, models


def evaluate_model_performance(df_results):
    """
    Evaluar y mostrar el rendimiento general de los modelos.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame con resultados de regresión por cluster
    """
    print("\n" + "="*70)
    print("📊 RESUMEN DE PERFORMANCE DE MODELOS")
    print("="*70)
    
    print("\n" + df_results.to_string(index=False))
    
    # Promedios
    print("\n" + "─"*70)
    print("📈 PROMEDIOS GENERALES")
    print("─"*70)
    
    avg_r2_lr = df_results['R2_Linear'].mean()
    avg_rmse_lr = df_results['RMSE_Linear'].mean()
    avg_r2_rf = df_results['R2_RandomForest'].mean()
    avg_rmse_rf = df_results['RMSE_RandomForest'].mean()
    
    print(f"\nRegresión Lineal:")
    print(f"   R² promedio:   {avg_r2_lr:.4f}")
    print(f"   RMSE promedio: {avg_rmse_lr:.4f}")
    
    print(f"\nRandom Forest:")
    print(f"   R² promedio:   {avg_r2_rf:.4f}")
    print(f"   RMSE promedio: {avg_rmse_rf:.4f}")
    
    # Mejor modelo por cluster
    print(f"\n{'─'*70}")
    print("🏆 MEJOR MODELO POR CLUSTER")
    print(f"{'─'*70}")
    best_counts = df_results['Best_Model'].value_counts()
    for model, count in best_counts.items():
        print(f"   {model}: {count} cluster(s)")


def plot_regression_performance(df_results, save_path=None):
    """
    Visualizar comparación de performance de modelos.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame con resultados de regresión
    save_path : str, optional
        Ruta donde guardar la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    clusters = df_results['Cluster']
    x_pos = np.arange(len(clusters))
    width = 0.35
    
    # Subplot 1: R² Score
    ax1 = axes[0]
    ax1.bar(x_pos - width/2, df_results['R2_Linear'], width, 
            label='Linear Regression', alpha=0.8, color='steelblue')
    ax1.bar(x_pos + width/2, df_results['R2_RandomForest'], width,
            label='Random Forest', alpha=0.8, color='forestgreen')
    
    ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de R² por Cluster', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(clusters)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Subplot 2: RMSE
    ax2 = axes[1]
    ax2.bar(x_pos - width/2, df_results['RMSE_Linear'], width,
            label='Linear Regression', alpha=0.8, color='coral')
    ax2.bar(x_pos + width/2, df_results['RMSE_RandomForest'], width,
            label='Random Forest', alpha=0.8, color='mediumpurple')
    
    ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de RMSE por Cluster', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(clusters)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {save_path}")
    
    plt.show()


def plot_feature_importance(models, cluster_id, features, top_n=None, save_path=None):
    """
    Visualizar importancia de features del modelo Random Forest.
    
    Parameters:
    -----------
    models : dict
        Diccionario de modelos por cluster
    cluster_id : int
        ID del cluster
    features : list
        Lista de nombres de features
    top_n : int, optional
        Mostrar solo las top_n features más importantes
    save_path : str, optional
        Ruta donde guardar la figura
    """
    rf_model = models[cluster_id]['random_forest']
    importances = rf_model.feature_importances_
    
    # Crear DataFrame
    df_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    if top_n:
        df_importance = df_importance.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(df_importance['Feature'], df_importance['Importance'], color='teal', alpha=0.7)
    plt.xlabel('Importancia', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Importancia de Features - Cluster {cluster_id}', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Gráfico guardado: {save_path}")
    
    plt.show()


def predict_with_best_model(df, models, df_results, features, target, cluster_col='Cluster'):
    """
    Hacer predicciones usando el mejor modelo por cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    models : dict
        Diccionario de modelos entrenados
    df_results : pd.DataFrame
        Resultados de evaluación
    features : list
        Lista de features
    target : str
        Nombre del target
    cluster_col : str
        Nombre de columna de cluster
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con predicciones
    """
    df_pred = df.copy()
    df_pred['Predicted'] = np.nan
    df_pred['Model_Used'] = ''
    
    for _, row in df_results.iterrows():
        cluster_id = row['Cluster']
        best_model_name = row['Best_Model']
        
        # Seleccionar modelo
        if best_model_name == 'Random Forest':
            model = models[cluster_id]['random_forest']
        else:
            model = models[cluster_id]['linear']
        
        # Predecir
        mask = df_pred[cluster_col] == cluster_id
        X_cluster = df_pred.loc[mask, features]
        predictions = model.predict(X_cluster)
        
        df_pred.loc[mask, 'Predicted'] = predictions
        df_pred.loc[mask, 'Model_Used'] = best_model_name
    
    # Calcular residuos
    df_pred['Residual'] = df_pred[target] - df_pred['Predicted']
    df_pred['Abs_Error'] = np.abs(df_pred['Residual'])
    
    return df_pred


def save_regression_results(df_results, models, filepath):
    """
    Guardar resultados de regresión y modelos.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Resultados de evaluación
    models : dict
        Diccionario de modelos
    filepath : str
        Ruta base para guardar archivos
    """
    # Guardar CSV de resultados
    csv_path = filepath.replace('.pkl', '.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"💾 Resultados guardados: {csv_path}")
    
    # Guardar modelos (opcional)
    import joblib
    joblib.dump(models, filepath)
    print(f"💾 Modelos guardados: {filepath}")


if __name__ == "__main__":
    print("📦 Módulo regression.py cargado correctamente")
    print("Funciones disponibles:")
    print("  - train_regression_by_cluster()")
    print("  - evaluate_model_performance()")
    print("  - plot_regression_performance()")
    print("  - predict_with_best_model()")
