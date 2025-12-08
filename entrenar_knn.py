#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para entrenar modelo KNN y evaluar diferentes valores de k
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
warnings.filterwarnings('ignore')

def calcular_precision_recall_por_clase(y_true, y_pred, clases):
    """
    Calcula precisión y recall para cada clase
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        clases: Lista de clases únicas
    
    Returns:
        dict: Diccionario con precisión y recall por clase
    """
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    
    precision_recall = {}
    
    for i, clase in enumerate(clases):
        # Precisión = TP / (TP + FP)
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall = TP / (TP + FN)
        fn = cm[i, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall[clase] = {
            'precision': precision,
            'recall': recall
        }
    
    return precision_recall

def imprimir_matriz_confusion(cm, clases, titulo):
    """
    Imprime la matriz de confusión de forma legible
    
    Args:
        cm: Matriz de confusión
        clases: Lista de clases
        titulo: Título para la matriz
    """
    print(f"\n{titulo}")
    print("-" * (len(titulo) + 10))
    
    # Crear encabezados
    header = "Actual\\Pred"
    for clase in clases:
        header += f"{clase:>12}"
    print(header)
    
    # Imprimir filas
    for i, clase_real in enumerate(clases):
        row = f"{clase_real:>10}"
        for j in range(len(clases)):
            row += f"{cm[i, j]:>12}"
        print(row)

def main():
    """
    Función principal que ejecuta todo el proceso de entrenamiento KNN
    """
    print("=" * 70)
    print("PROGRAMA PARA ENTRENAR MODELO KNN CON DIFERENTES VALORES DE K")
    print("=" * 70)
    
    # Definir rutas
    ruta_entrenamiento = r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_entrenamiento.csv"
    ruta_validacion = r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_validacion.csv"
    
    print(f"[INFO] Archivo entrenamiento: {ruta_entrenamiento}")
    print(f"[INFO] Archivo validación: {ruta_validacion}")
    print()
    
    # Paso 1: Cargar datasets
    print("[INFO] Cargando datasets...")
    
    try:
        # Cargar dataset de entrenamiento
        df_entrenamiento = pd.read_csv(ruta_entrenamiento)
        print(f"[OK] Dataset entrenamiento cargado: {len(df_entrenamiento):,} registros")
        
        # Cargar dataset de validación
        df_validacion = pd.read_csv(ruta_validacion)
        print(f"[OK] Dataset validación cargado: {len(df_validacion):,} registros")
        
    except Exception as e:
        print(f"[ERROR] Error al cargar archivos: {str(e)}")
        return False
    
    print()
    
    # Paso 2: Definir features y target
    print("[INFO] Configurando features y target...")
    
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
    
    print(f"   Features ({len(features)}):")
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")
    print(f"   Target: {target}")
    
    # Verificar que las columnas existen
    columnas_faltantes = []
    for feature in features:
        if feature not in df_entrenamiento.columns:
            columnas_faltantes.append(feature)
    
    if target not in df_entrenamiento.columns:
        columnas_faltantes.append(target)
    
    if columnas_faltantes:
        print(f"[ERROR] Columnas faltantes: {columnas_faltantes}")
        return False
    
    print("[OK] Todas las columnas encontradas")
    print()
    
    # Paso 3: Preparar datos
    print("[INFO] Preparando datos...")
    
    try:
        # Separar features y target para entrenamiento
        X_train = df_entrenamiento[features]
        y_train = df_entrenamiento[target]
        
        # Separar features y target para validación
        X_val = df_validacion[features]
        y_val = df_validacion[target]
        
        print(f"[OK] Datos preparados:")
        print(f"   Entrenamiento: {X_train.shape[0]:,} muestras, {X_train.shape[1]} features")
        print(f"   Validación: {X_val.shape[0]:,} muestras, {X_val.shape[1]} features")
        
        # Obtener clases únicas
        clases_unicas = sorted(y_train.unique())
        print(f"   Clases: {clases_unicas}")
        
    except Exception as e:
        print(f"[ERROR] Error al preparar datos: {str(e)}")
        return False
    
    print()
    
    # Paso 4: Entrenar modelos con diferentes valores de k
    print("[INFO] Entrenando modelos KNN con k de 1 a 10...")
    print()
    
    resultados = []
    
    for k in range(1, 11):
        print(f"K = {k}")
        print("=" * 30)
        
        try:
            # Crear y entrenar modelo
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            
            # Predicciones en entrenamiento
            y_train_pred = knn.predict(X_train)
            
            # Predicciones en validación
            y_val_pred = knn.predict(X_val)
            
            # Calcular métricas para entrenamiento
            accuracy_train = accuracy_score(y_train, y_train_pred)
            precision_recall_train = calcular_precision_recall_por_clase(y_train, y_train_pred, clases_unicas)
            
            # Calcular métricas para validación
            accuracy_val = accuracy_score(y_val, y_val_pred)
            precision_recall_val = calcular_precision_recall_por_clase(y_val, y_val_pred, clases_unicas)
            
            # Calcular precision y recall promedio (macro)
            precision_train_avg = np.mean([precision_recall_train[clase]['precision'] for clase in clases_unicas])
            recall_train_avg = np.mean([precision_recall_train[clase]['recall'] for clase in clases_unicas])
            
            precision_val_avg = np.mean([precision_recall_val[clase]['precision'] for clase in clases_unicas])
            recall_val_avg = np.mean([precision_recall_val[clase]['recall'] for clase in clases_unicas])
            
            # Guardar resultados
            resultado = {
                'k': k,
                'accuracy_train': accuracy_train,
                'accuracy_val': accuracy_val,
                'precision_train': precision_train_avg,
                'recall_train': recall_train_avg,
                'precision_val': precision_val_avg,
                'recall_val': recall_val_avg
            }
            resultados.append(resultado)
            
            # Mostrar métricas
            print(f"Accuracy - Entrenamiento: {accuracy_train:.4f}, Validación: {accuracy_val:.4f}")
            print(f"Precision - Entrenamiento: {precision_train_avg:.4f}, Validación: {precision_val_avg:.4f}")
            print(f"Recall - Entrenamiento: {recall_train_avg:.4f}, Validación: {recall_val_avg:.4f}")
            
            # Mostrar matrices de confusión
            cm_train = confusion_matrix(y_train, y_train_pred, labels=clases_unicas)
            cm_val = confusion_matrix(y_val, y_val_pred, labels=clases_unicas)
            
            imprimir_matriz_confusion(cm_train, clases_unicas, "Matriz de Confusión - Entrenamiento")
            imprimir_matriz_confusion(cm_val, clases_unicas, "Matriz de Confusión - Validación")
            
        except Exception as e:
            print(f"[ERROR] Error con k={k}: {str(e)}")
            continue
        
        print()
    
    # Paso 5: Resumen comparativo
    print("=" * 70)
    print("RESUMEN COMPARATIVO - TODOS LOS VALORES DE K")
    print("=" * 70)
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar tabla de resultados
    print("\nTabla de Resultados:")
    print("-" * 80)
    print(f"{'K':<3} {'Acc Train':<10} {'Acc Val':<10} {'Prec Train':<12} {'Prec Val':<12} {'Rec Train':<12} {'Rec Val':<12}")
    print("-" * 80)
    
    for _, row in df_resultados.iterrows():
        print(f"{row['k']:<3} "
              f"{row['accuracy_train']:<10.4f} "
              f"{row['accuracy_val']:<10.4f} "
              f"{row['precision_train']:<12.4f} "
              f"{row['precision_val']:<12.4f} "
              f"{row['recall_train']:<12.4f} "
              f"{row['recall_val']:<12.4f}")
    
    print("-" * 80)
    
    # Paso 6: Encontrar mejor k
    print("\nANÁLISIS DEL MEJOR VALOR DE K:")
    print("-" * 50)
    
    # Mejor k basado en accuracy de validación
    mejor_k_acc = df_resultados.loc[df_resultados['accuracy_val'].idxmax()]
    print(f"Mejor K por Accuracy (Validación): K={mejor_k_acc['k']} (Acc: {mejor_k_acc['accuracy_val']:.4f})")
    
    # Mejor k basado en precision de validación
    mejor_k_prec = df_resultados.loc[df_resultados['precision_val'].idxmax()]
    print(f"Mejor K por Precision (Validación): K={mejor_k_prec['k']} (Prec: {mejor_k_prec['precision_val']:.4f})")
    
    # Mejor k basado en recall de validación
    mejor_k_rec = df_resultados.loc[df_resultados['recall_val'].idxmax()]
    print(f"Mejor K por Recall (Validación): K={mejor_k_rec['k']} (Rec: {mejor_k_rec['recall_val']:.4f})")
    
    # Mejor k balanceado (promedio de métricas de validación)
    df_resultados['score_balanceado'] = (
        df_resultados['accuracy_val'] + 
        df_resultados['precision_val'] + 
        df_resultados['recall_val']
    ) / 3
    
    mejor_k_balanceado = df_resultados.loc[df_resultados['score_balanceado'].idxmax()]
    print(f"Mejor K Balanceado: K={mejor_k_balanceado['k']} (Score: {mejor_k_balanceado['score_balanceado']:.4f})")
    
    # Paso 7: Resumen final
    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Modelos entrenados: {len(resultados)}")
    print(f"[INFO] Features utilizadas: {len(features)}")
    print(f"[INFO] Clases objetivo: {len(clases_unicas)}")
    print(f"[INFO] Registros entrenamiento: {len(X_train):,}")
    print(f"[INFO] Registros validación: {len(X_val):,}")
    print(f"[INFO] Mejor K recomendado: {mejor_k_balanceado['k']}")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        exito = main()
        if exito:
            print("\n[OK] El programa se ejecutó correctamente.")
            sys.exit(0)
        else:
            print("\n[ERROR] El programa terminó con errores.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Programa interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {str(e)}")
        sys.exit(1)
