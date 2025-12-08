#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script maestro para comparar costo computacional de todos los modelos
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize
import warnings
import time
import psutil
import os
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cargar_datos():
    """
    Carga los datasets de entrenamiento y validación
    """
    print("=" * 70)
    print("ANÁLISIS COMPARATIVO DE COSTO COMPUTACIONAL - TODOS LOS MODELOS")
    print("=" * 70)
    
    # Definir rutas
    ruta_entrenamiento = r"C:\Users\anaka\Downloads\Proyecto ML (2)\Proyecto ML\Porgramas Python\dataset_entrenamiento.csv"
    ruta_validacion = r"C:\Users\anaka\Downloads\Proyecto ML (2)\Proyecto ML\Porgramas Python\dataset_validacion.csv"
    
    try:
        # Cargar datasets
        df_train = pd.read_csv(ruta_entrenamiento)
        df_val = pd.read_csv(ruta_validacion)
        
        print(f"[OK] Dataset entrenamiento: {len(df_train):,} registros")
        print(f"[OK] Dataset validación: {len(df_val):,} registros")
        
    except Exception as e:
        print(f"[ERROR] Error al cargar archivos: {str(e)}")
        return None, None, None, None
    
    # Definir features y target
    features = [
        "HarshBrakingPer100Km_Norm",
        "HarshAccelerationPer100Km_Norm", 
        "HarshTurningPer100Km_Norm",
        "IdlingRatePercentOfIgnitionTime_Norm",
        "SpeedOver95Per100Km_Norm",
        "FuelUnder50PercentPer100Km_Norm",
        "RPMOver1600Per100Km_Norm",
        "FueraDeRutaFRPer100Km_Norm"
    ]
    
    target = "Categoria_Conductor"
    
    # Preparar datos
    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]
    
    # Obtener clases únicas
    clases_unicas = sorted(y_train.unique())
    print(f"[INFO] Clases encontradas: {clases_unicas}")
    
    return X_train, y_train, X_val, y_val

def medir_costo_computacional(modelo, X_train, y_train, X_val, y_val, nombre_modelo):
    """
    Mide el costo computacional del modelo
    
    Args:
        modelo: Modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        nombre_modelo: Nombre del modelo para identificación
    
    Returns:
        dict: Diccionario con métricas computacionales
    """
    print(f"\n[INFO] Midiendo costo computacional para {nombre_modelo}...")
    
    # === MEDICIÓN DEL COSTO COMPUTACIONAL ===
    
    # Medir tiempo de entrenamiento
    inicio_entrenamiento = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio_entrenamiento
    
    # Medir tiempo de predicción
    inicio_prediccion = time.time()
    y_pred = modelo.predict(X_val)
    tiempo_prediccion = time.time() - inicio_prediccion
    
    # Medir uso de memoria (aproximado)
    proceso = psutil.Process(os.getpid())
    memoria_mb = proceso.memory_info().rss / 1024 / 1024  # Convertir a MB
    
    # Medir uso de CPU
    cpu_percent = psutil.cpu_percent(interval=1)

    # Estimar ciclos de reloj a partir de la frecuencia actual de CPU (MHz)
    try:
        freq = psutil.cpu_freq()
        mhz = float(freq.current) if freq and freq.current else 1000.0
    except Exception:
        mhz = 1000.0  # valor por defecto si no disponible
    ciclos_entrenamiento = int(tiempo_entrenamiento * mhz * 1_000_000)
    ciclos_prediccion = int(tiempo_prediccion * mhz * 1_000_000)
    
    # Calcular métricas de rendimiento
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    
    # Calcular ROC AUC promedio
    y_val_bin = label_binarize(y_val, classes=sorted(y_val.unique()))
    y_proba = modelo.predict_proba(X_val)
    roc_auc_scores = []
    for i in range(len(sorted(y_val.unique()))):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
        roc_auc_scores.append(auc(fpr, tpr))
    roc_auc_promedio = np.mean(roc_auc_scores)
    
    # Crear diccionario con resultados
    resultados = {
        'Modelo': nombre_modelo,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'ROC_AUC': roc_auc_promedio,
        'Ciclos_Entrenamiento': ciclos_entrenamiento,
        'Ciclos_Prediccion': ciclos_prediccion,
        'Frecuencia_MHz': mhz,
        'Memoria_MB': memoria_mb,
        'CPU_Percent': cpu_percent
    }
    
    print(f"[OK] Costo computacional medido:")
    print(f"   Ciclos entrenamiento: {ciclos_entrenamiento:,} ciclos (frec. ~{mhz:.1f} MHz)")
    print(f"   Ciclos predicción: {ciclos_prediccion:,} ciclos (frec. ~{mhz:.1f} MHz)")
    print(f"   Memoria utilizada: {memoria_mb:.2f} MB")
    print(f"   CPU utilizado: {cpu_percent:.2f}%")
    
    return resultados

def visualizar_costo_computacional(resultados_df):
    """
    Crea visualización del costo computacional
    
    Args:
        resultados_df: DataFrame con métricas computacionales
    """
    print("\n[INFO] Generando visualización de costo computacional...")
    
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Preparar datos para visualización
    modelos = resultados_df['Modelo'].tolist()
    
    # Subplot 1: Ciclos de entrenamiento
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(modelos, resultados_df['Ciclos_Entrenamiento'], 
                    color='skyblue', alpha=0.7)
    plt.title('Ciclos de Entrenamiento por Modelo')
    plt.ylabel('Ciclos de reloj')
    etiquetas_modelos = [m.replace(' ', '\n') if ' ' in m else m for m in modelos]
    plt.xticks(ticks=range(len(modelos)), labels=etiquetas_modelos, rotation=0, fontsize=10)
    plt.gca().tick_params(axis='x', labelcolor='black')
    for bar, valor in zip(bars1, resultados_df['Ciclos_Entrenamiento']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 * bar.get_height() if bar.get_height() else 1),
                f'{int(valor):,} ciclos', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Ciclos de predicción
    plt.subplot(2, 2, 2)
    bars2 = plt.bar(modelos, resultados_df['Ciclos_Prediccion'], 
                    color='lightgreen', alpha=0.7)
    plt.title('Ciclos de Predicción por Modelo')
    plt.ylabel('Ciclos de reloj')
    etiquetas_modelos = [m.replace(' ', '\n') if ' ' in m else m for m in modelos]
    plt.xticks(ticks=range(len(modelos)), labels=etiquetas_modelos, rotation=0, fontsize=10)
    plt.gca().tick_params(axis='x', labelcolor='black')
    for bar, valor in zip(bars2, resultados_df['Ciclos_Prediccion']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 * bar.get_height() if bar.get_height() else 1),
                f'{int(valor):,} ciclos', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Uso de memoria
    plt.subplot(2, 2, 3)
    bars3 = plt.bar(modelos, resultados_df['Memoria_MB'], 
                    color='lightcoral', alpha=0.7)
    plt.title('Uso de Memoria por Modelo')
    plt.ylabel('Memoria (MB)')
    etiquetas_modelos = [m.replace(' ', '\n') if ' ' in m else m for m in modelos]
    plt.xticks(ticks=range(len(modelos)), labels=etiquetas_modelos, rotation=0, fontsize=10)
    plt.gca().tick_params(axis='x', labelcolor='black')
    for bar, valor in zip(bars3, resultados_df['Memoria_MB']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{valor:.1f}MB', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Uso de CPU
    plt.subplot(2, 2, 4)
    bars4 = plt.bar(modelos, resultados_df['CPU_Percent'], 
                    color='gold', alpha=0.7)
    plt.title('Uso de CPU por Modelo')
    plt.ylabel('CPU (%)')
    etiquetas_modelos = [m.replace(' ', '\n') if ' ' in m else m for m in modelos]
    plt.xticks(ticks=range(len(modelos)), labels=etiquetas_modelos, rotation=0, fontsize=10)
    plt.gca().tick_params(axis='x', labelcolor='black')
    for bar, valor in zip(bars4, resultados_df['CPU_Percent']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{valor:.1f}%', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Costo Computacional - Comparativa de Modelos', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.show()
    
    print("[OK] Visualización de costo computacional generada")

def guardar_resultados(resultados_df, archivo='resultados_costos.csv'):
    """
    Guarda los resultados en un archivo CSV
    
    Args:
        resultados_df: DataFrame con resultados
        archivo: Nombre del archivo CSV
    """
    try:
        resultados_df.to_csv(archivo, index=False)
        print(f"[OK] Resultados guardados en {archivo}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el archivo: {str(e)}")

def mostrar_resumen_costos(resultados_df):
    """
    Muestra resumen de costos computacionales
    
    Args:
        resultados_df: DataFrame con resultados
    """
    print("\n" + "=" * 80)
    print("RESUMEN DE COSTOS COMPUTACIONALES")
    print("=" * 80)
    
    # Mostrar tabla completa
    print("\nTabla completa de resultados:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(resultados_df.round(4))
    
    # Análisis comparativo
    print(f"\nANÁLISIS COMPARATIVO:")
    print("-" * 40)
    
    # Mejor modelo por accuracy
    mejor_accuracy = resultados_df.loc[resultados_df['Accuracy'].idxmax()]
    print(f"Mejor Accuracy: {mejor_accuracy['Modelo']} ({mejor_accuracy['Accuracy']:.4f})")
    
    # Modelo más rápido en entrenamiento
    mas_rapido_entrenamiento = resultados_df.loc[resultados_df['Ciclos_Entrenamiento'].idxmin()]
    print(f"Más rápido en entrenamiento: {mas_rapido_entrenamiento['Modelo']} ({int(mas_rapido_entrenamiento['Ciclos_Entrenamiento']):,} ciclos)")
    
    # Modelo más rápido en predicción
    mas_rapido_prediccion = resultados_df.loc[resultados_df['Ciclos_Prediccion'].idxmin()]
    print(f"Más rápido en predicción: {mas_rapido_prediccion['Modelo']} ({int(mas_rapido_prediccion['Ciclos_Prediccion']):,} ciclos)")
    
    # Modelo con menor uso de memoria
    menor_memoria = resultados_df.loc[resultados_df['Memoria_MB'].idxmin()]
    print(f"Menor uso de memoria: {menor_memoria['Modelo']} ({menor_memoria['Memoria_MB']:.2f}MB)")
    
    print("=" * 80)

def main():
    """
    Función principal que ejecuta el análisis comparativo
    """
    # Cargar datos
    X_train, y_train, X_val, y_val = cargar_datos()
    
    if X_train is None:
        return False
    
    # === ENTRENAMIENTO Y PREDICCIÓN ===
    print("\n[INFO] Iniciando análisis comparativo de modelos...")
    
    # Definir modelos con sus configuraciones óptimas
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(C=100.0, kernel='linear', gamma=0.001, random_state=42, probability=True),
        'Naive Bayes': GaussianNB(var_smoothing=1e-05),
        'Árbol de Decisión': DecisionTreeClassifier(
            max_depth=None, 
            min_samples_split=20, 
            min_samples_leaf=5, 
            random_state=42
        )
    }
    
    # Lista para almacenar resultados
    resultados_lista = []
    
    # Medir costo computacional para cada modelo
    for nombre, modelo in modelos.items():
        try:
            resultados = medir_costo_computacional(
                modelo, X_train, y_train, X_val, y_val, nombre
            )
            resultados_lista.append(resultados)
        except Exception as e:
            print(f"[ERROR] Error con modelo {nombre}: {str(e)}")
            continue
    
    # Crear DataFrame con todos los resultados
    resultados_df = pd.DataFrame(resultados_lista)
    
    # === MÉTRICAS DE RENDIMIENTO ===
    print(f"\n[INFO] Análisis completado para {len(resultados_df)} modelos")
    
    # === VISUALIZACIÓN DE COSTO COMPUTACIONAL ===
    if not resultados_df.empty:
        visualizar_costo_computacional(resultados_df)
        mostrar_resumen_costos(resultados_df)
        guardar_resultados(resultados_df, 'resultados_costos_completo.csv')
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPARATIVO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        exito = main()
        if exito:
            print("\n[OK] El programa se ejecutó correctamente.")
        else:
            print("\n[ERROR] El programa terminó con errores.")
    except KeyboardInterrupt:
        print("\n\n[WARNING] Programa interrumpido por el usuario.")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {str(e)}")

