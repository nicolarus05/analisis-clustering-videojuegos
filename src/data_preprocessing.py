"""
Módulo de preprocesamiento de datos para Video Game Sales.

Este módulo contiene funciones para:
- Carga de datos
- Limpieza de valores nulos y duplicados
- Selección de features
- Normalización
- Preparación para modelos de ML
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Cargar dataset de videojuegos desde archivo CSV.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV (ej: 'data/raw/vgsales.csv')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos cargados
    
    Example:
    --------
    >>> df = load_data('data/raw/vgsales.csv')
    >>> print(df.shape)
    (16598, 11)
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset cargado exitosamente")
        print(f"   - Filas: {df.shape[0]:,}")
        print(f"   - Columnas: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {filepath}")
        print("   Descarga el dataset desde Kaggle:")
        print("   https://www.kaggle.com/datasets/gregorut/videogamesales")
        return None
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None


def explore_data(df):
    """
    Mostrar información exploratoria del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a explorar
    """
    print("\n" + "="*70)
    print("📊 INFORMACIÓN DEL DATASET")
    print("="*70)
    
    print("\n1. Dimensiones:")
    print(f"   - Filas: {df.shape[0]:,}")
    print(f"   - Columnas: {df.shape[1]}")
    
    print("\n2. Tipos de datos:")
    print(df.dtypes)
    
    print("\n3. Valores nulos:")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({
        'Nulos': null_counts,
        'Porcentaje': null_pct
    })
    print(null_df[null_df['Nulos'] > 0])
    
    print("\n4. Estadísticas descriptivas (ventas en millones):")
    print(df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].describe())
    
    print("\n5. Información categórica:")
    print(f"   - Plataformas únicas: {df['Platform'].nunique()}")
    print(f"   - Géneros únicos: {df['Genre'].nunique()}")
    print(f"   - Publishers únicos: {df['Publisher'].nunique()}")
    
    print("\n6. Rango temporal:")
    if 'Year' in df.columns:
        year_min = df['Year'].min()
        year_max = df['Year'].max()
        print(f"   - Desde: {year_min:.0f}" if not pd.isna(year_min) else "   - Desde: N/A")
        print(f"   - Hasta: {year_max:.0f}" if not pd.isna(year_max) else "   - Hasta: N/A")


def clean_data(df, drop_null_year=True, drop_null_publisher=True, year_range=(1980, 2020)):
    """
    Limpiar dataset eliminando valores nulos y duplicados.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame original
    drop_null_year : bool, default=True
        Si True, elimina filas con Year nulo
    drop_null_publisher : bool, default=True
        Si True, elimina filas con Publisher nulo
    year_range : tuple, default=(1980, 2020)
        Rango de años válidos (min, max)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame limpio
    
    Example:
    --------
    >>> df_clean = clean_data(df)
    >>> print(f"Registros eliminados: {len(df) - len(df_clean)}")
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    print("\n" + "="*70)
    print("🧹 LIMPIEZA DE DATOS")
    print("="*70)
    
    # Eliminar duplicados
    duplicates = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"\n1. Duplicados eliminados: {duplicates}")
    
    # Eliminar valores nulos en Year
    if drop_null_year:
        null_year = df_clean['Year'].isnull().sum()
        df_clean = df_clean.dropna(subset=['Year'])
        print(f"2. Filas con Year nulo eliminadas: {null_year}")
    
    # Eliminar valores nulos en Publisher
    if drop_null_publisher:
        null_publisher = df_clean['Publisher'].isnull().sum()
        df_clean = df_clean.dropna(subset=['Publisher'])
        print(f"3. Filas con Publisher nulo eliminadas: {null_publisher}")
    
    # Filtrar rango de años válidos
    if year_range:
        before_filter = len(df_clean)
        df_clean = df_clean[
            (df_clean['Year'] >= year_range[0]) & 
            (df_clean['Year'] <= year_range[1])
        ]
        print(f"4. Filas fuera del rango {year_range} eliminadas: {before_filter - len(df_clean)}")
    
    # Resetear índice
    df_clean = df_clean.reset_index(drop=True)
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    removed_pct = (removed_rows / initial_rows * 100)
    
    print(f"\n✅ Limpieza completada:")
    print(f"   - Filas iniciales: {initial_rows:,}")
    print(f"   - Filas finales: {final_rows:,}")
    print(f"   - Filas eliminadas: {removed_rows:,} ({removed_pct:.2f}%)")
    
    return df_clean


def select_features_for_clustering(df, features=None):
    """
    Seleccionar features numéricas para clustering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    features : list, optional
        Lista de nombres de columnas. Si None, usa features por defecto.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con solo las features seleccionadas
    
    Example:
    --------
    >>> features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
    >>> df_features = select_features_for_clustering(df, features)
    """
    if features is None:
        features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
    
    print(f"\n📋 Features seleccionadas para clustering:")
    for i, feat in enumerate(features, 1):
        print(f"   {i}. {feat}")
    
    df_features = df[features].copy()
    
    # Verificar valores nulos
    null_check = df_features.isnull().sum()
    if null_check.any():
        print(f"\n⚠️  Advertencia: Se encontraron valores nulos:")
        print(null_check[null_check > 0])
    else:
        print(f"\n✅ No hay valores nulos en las features seleccionadas")
    
    return df_features


def normalize_data(df):
    """
    Normalizar datos usando StandardScaler (z-score normalization).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con features numéricas
    
    Returns:
    --------
    tuple
        (df_scaled, scaler)
        - df_scaled: DataFrame normalizado
        - scaler: objeto StandardScaler ajustado
    
    Example:
    --------
    >>> df_scaled, scaler = normalize_data(df_features)
    >>> print(df_scaled.mean())  # Debe ser ~0
    >>> print(df_scaled.std())   # Debe ser ~1
    """
    scaler = StandardScaler()
    
    # Ajustar y transformar
    data_scaled = scaler.fit_transform(df)
    
    # Crear DataFrame con los mismos nombres de columnas
    df_scaled = pd.DataFrame(
        data_scaled,
        columns=df.columns,
        index=df.index
    )
    
    print("\n" + "="*70)
    print("📐 NORMALIZACIÓN DE DATOS")
    print("="*70)
    print("\nEstadísticas después de normalización:")
    print("\nMedias (deben ser ~0):")
    print(df_scaled.mean().round(4))
    print("\nDesviaciones estándar (deben ser ~1):")
    print(df_scaled.std().round(4))
    
    print("\n✅ Datos normalizados correctamente")
    
    return df_scaled, scaler


def prepare_regression_data(df, target_col='Global_Sales', feature_cols=None):
    """
    Preparar datos para regresión (X, y).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    target_col : str, default='Global_Sales'
        Nombre de la columna objetivo
    feature_cols : list, optional
        Lista de features. Si None, usa todas menos target.
    
    Returns:
    --------
    tuple
        (X, y) donde X son las features e y es el target
    
    Example:
    --------
    >>> X, y = prepare_regression_data(df, target_col='Global_Sales')
    """
    if feature_cols is None:
        feature_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n📊 Datos preparados para regresión:")
    print(f"   - Features (X): {X.shape}")
    print(f"   - Target (y): {y.shape}")
    print(f"   - Target column: {target_col}")
    
    return X, y


def save_processed_data(df, filepath, verbose=True):
    """
    Guardar DataFrame procesado en archivo CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a guardar
    filepath : str
        Ruta donde guardar el archivo
    verbose : bool, default=True
        Si True, muestra mensajes de confirmación
    
    Example:
    --------
    >>> save_processed_data(df_clean, 'data/processed/vgsales_clean.csv')
    """
    try:
        df.to_csv(filepath, index=False)
        if verbose:
            print(f"\n💾 Archivo guardado exitosamente:")
            print(f"   {filepath}")
            print(f"   Tamaño: {len(df):,} filas x {len(df.columns)} columnas")
    except Exception as e:
        print(f"\n❌ Error al guardar archivo: {e}")


def get_data_summary(df):
    """
    Obtener resumen estadístico del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a resumir
    
    Returns:
    --------
    dict
        Diccionario con estadísticas clave
    """
    summary = {
        'n_records': len(df),
        'n_columns': len(df.columns),
        'n_duplicates': df.duplicated().sum(),
        'n_missing': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    if 'Year' in df.columns:
        summary['year_range'] = (df['Year'].min(), df['Year'].max())
    
    if 'Global_Sales' in df.columns:
        summary['total_sales'] = df['Global_Sales'].sum()
        summary['avg_sales'] = df['Global_Sales'].mean()
        summary['top_game'] = df.loc[df['Global_Sales'].idxmax(), 'Name']
    
    return summary


# Función auxiliar para pipeline completo
def preprocessing_pipeline(filepath, save_clean=True, return_scaled=True):
    """
    Pipeline completo de preprocesamiento.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV original
    save_clean : bool, default=True
        Si True, guarda el dataset limpio
    return_scaled : bool, default=True
        Si True, retorna también datos normalizados
    
    Returns:
    --------
    tuple or pd.DataFrame
        Si return_scaled=True: (df_clean, df_scaled, scaler)
        Si return_scaled=False: df_clean
    
    Example:
    --------
    >>> df_clean, df_scaled, scaler = preprocessing_pipeline('data/raw/vgsales.csv')
    """
    # 1. Cargar datos
    df = load_data(filepath)
    if df is None:
        return None
    
    # 2. Explorar
    explore_data(df)
    
    # 3. Limpiar
    df_clean = clean_data(df)
    
    # 4. Guardar datos limpios
    if save_clean:
        save_processed_data(df_clean, 'data/processed/vgsales_clean.csv')
    
    # 5. Preparar para clustering (opcional)
    if return_scaled:
        features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
        df_features = select_features_for_clustering(df_clean, features)
        df_scaled, scaler = normalize_data(df_features)
        
        return df_clean, df_scaled, scaler
    
    return df_clean


if __name__ == "__main__":
    # Ejecutar pipeline completo
    print("🚀 Ejecutando pipeline de preprocesamiento...")
    result = preprocessing_pipeline('data/raw/vgsales.csv')
    
    if result is not None:
        df_clean, df_scaled, scaler = result
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETADO")
        print("="*70)
        print(f"\nDataset limpio: {df_clean.shape}")
        print(f"Dataset normalizado: {df_scaled.shape}")
