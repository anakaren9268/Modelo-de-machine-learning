#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para entrenar modelo SVM con búsqueda de hiperparámetros
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import warnings
import sys
from itertools import product
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
    Función principal que ejecuta todo el proceso de entrenamiento SVM
    """
    print("=" * 70)
    print("PROGRAMA PARA ENTRENAR MODELO SVM CON BÚSQUEDA DE HIPERPARÁMETROS")
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
    
    # Paso 4: Definir espacio de búsqueda de hiperparámetros
    print("[INFO] Configurando espacio de búsqueda de hiperparámetros...")
    
    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }
    
    print("   Hiperparámetros a evaluar:")
    print(f"   - C: {param_grid['C']}")
    print(f"   - kernel: {param_grid['kernel']}")
    print(f"   - gamma: {param_grid['gamma']}")
    
    # Calcular número total de combinaciones
    total_combinaciones = len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma'])
    print(f"   - Total de combinaciones posibles: {total_combinaciones}")
    print(f"   - Iteraciones de búsqueda aleatoria: 10")
    
    print()
    
    # Paso 5: Realizar búsqueda aleatoria
    print("[INFO] Iniciando búsqueda aleatoria de hiperparámetros...")
    
    try:
        # Crear modelo SVM base
        svm_base = SVC(random_state=42)
        
        # Configurar RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=svm_base,
            param_distributions=param_grid,
            n_iter=10,  # 10 iteraciones de muestreo aleatorio
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,  # Usar todos los cores disponibles
            verbose=1
        )
        
        # Realizar búsqueda
        print("   Ejecutando RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        
        print("[OK] Búsqueda aleatoria completada")
        
    except Exception as e:
        print(f"[ERROR] Error en búsqueda aleatoria: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Evaluar cada combinación probada
    print("[INFO] Evaluando combinaciones de hiperparámetros...")
    
    resultados = []
    
    # Obtener todas las combinaciones probadas
    resultados_cv = random_search.cv_results_
    
    for i in range(len(resultados_cv['params'])):
        params = resultados_cv['params'][i]
        mean_score = resultados_cv['mean_test_score'][i]
        std_score = resultados_cv['std_test_score'][i]
        
        print(f"\nCombinación {i+1}:")
        print(f"   C: {params['C']}, kernel: {params['kernel']}, gamma: {params['gamma']}")
        print(f"   CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        try:
            # Crear modelo con estos hiperparámetros
            svm_model = SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                random_state=42
            )
            
            # Entrenar modelo
            svm_model.fit(X_train, y_train)
            
            # Predicciones en entrenamiento
            y_train_pred = svm_model.predict(X_train)
            
            # Predicciones en validación
            y_val_pred = svm_model.predict(X_val)
            
            # Calcular métricas para entrenamiento
            accuracy_train = accuracy_score(y_train, y_train_pred)
            precision_train = precision_score(y_train, y_train_pred, average='macro')
            recall_train = recall_score(y_train, y_train_pred, average='macro')
            
            # Calcular métricas para validación
            accuracy_val = accuracy_score(y_val, y_val_pred)
            precision_val = precision_score(y_val, y_val_pred, average='macro')
            recall_val = recall_score(y_val, y_val_pred, average='macro')
            
            # Guardar resultados
            resultado = {
                'combinacion': i+1,
                'C': params['C'],
                'kernel': params['kernel'],
                'gamma': params['gamma'],
                'cv_score': mean_score,
                'accuracy_train': accuracy_train,
                'accuracy_val': accuracy_val,
                'precision_train': precision_train,
                'recall_train': recall_train,
                'precision_val': precision_val,
                'recall_val': recall_val
            }
            resultados.append(resultado)
            
            # Mostrar métricas
            print(f"   Accuracy - Entrenamiento: {accuracy_train:.4f}, Validación: {accuracy_val:.4f}")
            print(f"   Precision - Entrenamiento: {precision_train:.4f}, Validación: {precision_val:.4f}")
            print(f"   Recall - Entrenamiento: {recall_train:.4f}, Validación: {recall_val:.4f}")
            
            # Mostrar matrices de confusión
            cm_train = confusion_matrix(y_train, y_train_pred, labels=clases_unicas)
            cm_val = confusion_matrix(y_val, y_val_pred, labels=clases_unicas)
            
            imprimir_matriz_confusion(cm_train, clases_unicas, "Matriz de Confusión - Entrenamiento")
            imprimir_matriz_confusion(cm_val, clases_unicas, "Matriz de Confusión - Validación")
            
        except Exception as e:
            print(f"   [ERROR] Error con esta combinación: {str(e)}")
            continue
    
    print()
    
    # Paso 7: Resumen tabulado
    print("=" * 70)
    print("RESUMEN TABULADO - TODAS LAS COMBINACIONES EVALUADAS")
    print("=" * 70)
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar tabla de resultados
    print("\nTabla de Resultados:")
    print("-" * 120)
    print(f"{'Comb':<4} {'C':<6} {'Kernel':<8} {'Gamma':<8} {'CV Score':<10} {'Acc Train':<10} {'Acc Val':<10} {'Prec Train':<12} {'Prec Val':<12} {'Rec Train':<12} {'Rec Val':<12}")
    print("-" * 120)
    
    for _, row in df_resultados.iterrows():
        print(f"{row['combinacion']:<4} "
              f"{row['C']:<6} "
              f"{row['kernel']:<8} "
              f"{str(row['gamma']):<8} "
              f"{row['cv_score']:<10.4f} "
              f"{row['accuracy_train']:<10.4f} "
              f"{row['accuracy_val']:<10.4f} "
              f"{row['precision_train']:<12.4f} "
              f"{row['precision_val']:<12.4f} "
              f"{row['recall_train']:<12.4f} "
              f"{row['recall_val']:<12.4f}")
    
    print("-" * 120)
    
    # Paso 8: Identificar mejor configuración
    print("\nANÁLISIS DE LA MEJOR CONFIGURACIÓN:")
    print("-" * 50)
    
    # Mejor configuración por accuracy de validación
    mejor_acc = df_resultados.loc[df_resultados['accuracy_val'].idxmax()]
    print(f"Mejor por Accuracy (Validación):")
    print(f"   Combinación: {mejor_acc['combinacion']}")
    print(f"   C: {mejor_acc['C']}, kernel: {mejor_acc['kernel']}, gamma: {mejor_acc['gamma']}")
    print(f"   Accuracy: {mejor_acc['accuracy_val']:.4f}")
    
    # Mejor configuración por precision de validación
    mejor_prec = df_resultados.loc[df_resultados['precision_val'].idxmax()]
    print(f"\nMejor por Precision (Validación):")
    print(f"   Combinación: {mejor_prec['combinacion']}")
    print(f"   C: {mejor_prec['C']}, kernel: {mejor_prec['kernel']}, gamma: {mejor_prec['gamma']}")
    print(f"   Precision: {mejor_prec['precision_val']:.4f}")
    
    # Mejor configuración por recall de validación
    mejor_rec = df_resultados.loc[df_resultados['recall_val'].idxmax()]
    print(f"\nMejor por Recall (Validación):")
    print(f"   Combinación: {mejor_rec['combinacion']}")
    print(f"   C: {mejor_rec['C']}, kernel: {mejor_rec['kernel']}, gamma: {mejor_rec['gamma']}")
    print(f"   Recall: {mejor_rec['recall_val']:.4f}")
    
    # Mejor configuración balanceada
    df_resultados['score_balanceado'] = (
        df_resultados['accuracy_val'] + 
        df_resultados['precision_val'] + 
        df_resultados['recall_val']
    ) / 3
    
    mejor_balanceado = df_resultados.loc[df_resultados['score_balanceado'].idxmax()]
    print(f"\nMejor Configuración Balanceada:")
    print(f"   Combinación: {mejor_balanceado['combinacion']}")
    print(f"   C: {mejor_balanceado['C']}, kernel: {mejor_balanceado['kernel']}, gamma: {mejor_balanceado['gamma']}")
    print(f"   Score Balanceado: {mejor_balanceado['score_balanceado']:.4f}")
    print(f"   Accuracy: {mejor_balanceado['accuracy_val']:.4f}")
    print(f"   Precision: {mejor_balanceado['precision_val']:.4f}")
    print(f"   Recall: {mejor_balanceado['recall_val']:.4f}")
    
    # Paso 9: Resumen final
    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Combinaciones evaluadas: {len(resultados)}")
    print(f"[INFO] Features utilizadas: {len(features)}")
    print(f"[INFO] Clases objetivo: {len(clases_unicas)}")
    print(f"[INFO] Registros entrenamiento: {len(X_train):,}")
    print(f"[INFO] Registros validación: {len(X_val):,}")
    print(f"[INFO] Mejor configuración recomendada: Combinación {mejor_balanceado['combinacion']}")
    print(f"[INFO] Mejor accuracy en validación: {mejor_acc['accuracy_val']:.4f}")
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
