#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para entrenar modelo Árbol de Decisión con búsqueda de hiperparámetros
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import warnings
import sys
from itertools import product
warnings.filterwarnings('ignore')

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
    Función principal que ejecuta todo el proceso de entrenamiento Árbol de Decisión
    """
    print("=" * 70)
    print("PROGRAMA PARA ENTRENAR MODELO ÁRBOL DE DECISIÓN CON BÚSQUEDA DE HIPERPARÁMETROS")
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
    
    # Definir valores para cada hiperparámetro
    max_depth_values = [3, 5, 7, 10, 15, 20, None]
    min_samples_split_values = [2, 5, 10, 20, 50]
    min_samples_leaf_values = [1, 2, 5, 10, 20]
    
    print("   Hiperparámetros a evaluar:")
    print(f"   - max_depth: {max_depth_values}")
    print(f"   - min_samples_split: {min_samples_split_values}")
    print(f"   - min_samples_leaf: {min_samples_leaf_values}")
    
    # Calcular número total de combinaciones
    total_combinaciones = len(max_depth_values) * len(min_samples_split_values) * len(min_samples_leaf_values)
    print(f"   - Total de combinaciones posibles: {total_combinaciones}")
    
    print()
    
    # Paso 5: Realizar búsqueda de hiperparámetros
    print("[INFO] Iniciando búsqueda de hiperparámetros...")
    
    resultados = []
    combinacion_actual = 0
    
    # Crear todas las combinaciones posibles
    combinaciones = list(product(max_depth_values, min_samples_split_values, min_samples_leaf_values))
    
    print(f"   Evaluando {len(combinaciones)} combinaciones...")
    print()
    
    # Iterar sobre todas las combinaciones
    for max_depth, min_samples_split, min_samples_leaf in combinaciones:
        combinacion_actual += 1
        
        print(f"[INFO] Combinación {combinacion_actual}/{len(combinaciones)}")
        print(f"   max_depth: {max_depth}")
        print(f"   min_samples_split: {min_samples_split}")
        print(f"   min_samples_leaf: {min_samples_leaf}")
        
        try:
            # Crear modelo con estos hiperparámetros
            dt_model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            # Entrenar modelo
            dt_model.fit(X_train, y_train)
            
            # Predicciones en entrenamiento
            y_train_pred = dt_model.predict(X_train)
            
            # Predicciones en validación
            y_val_pred = dt_model.predict(X_val)
            
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
                'combinacion': combinacion_actual,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
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
            
            print("-" * 70)
            
        except Exception as e:
            print(f"   [ERROR] Error con esta combinación: {str(e)}")
            continue
    
    print()
    
    # Paso 6: Resumen de resultados
    print("=" * 70)
    print("RESUMEN DE RESULTADOS - TODAS LAS COMBINACIONES EVALUADAS")
    print("=" * 70)
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar tabla de resultados
    print("\nTabla de Resultados:")
    print("-" * 140)
    print(f"{'Comb':<4} {'Max Depth':<10} {'Min Split':<10} {'Min Leaf':<10} {'Acc Train':<10} {'Acc Val':<10} {'Prec Train':<12} {'Prec Val':<12} {'Rec Train':<12} {'Rec Val':<12}")
    print("-" * 140)
    
    for _, row in df_resultados.iterrows():
        max_depth_str = str(row['max_depth']) if row['max_depth'] is not None else 'None'
        print(f"{row['combinacion']:<4} "
              f"{max_depth_str:<10} "
              f"{row['min_samples_split']:<10} "
              f"{row['min_samples_leaf']:<10} "
              f"{row['accuracy_train']:<10.4f} "
              f"{row['accuracy_val']:<10.4f} "
              f"{row['precision_train']:<12.4f} "
              f"{row['precision_val']:<12.4f} "
              f"{row['recall_train']:<12.4f} "
              f"{row['recall_val']:<12.4f}")
    
    print("-" * 140)
    
    # Paso 7: Identificar mejor configuración
    print("\nANÁLISIS DE LA MEJOR CONFIGURACIÓN:")
    print("-" * 50)
    
    # Mejor configuración por accuracy de validación
    mejor_acc = df_resultados.loc[df_resultados['accuracy_val'].idxmax()]
    print(f"Mejor por Accuracy (Validación):")
    print(f"   Combinación: {mejor_acc['combinacion']}")
    print(f"   max_depth: {mejor_acc['max_depth']}")
    print(f"   min_samples_split: {mejor_acc['min_samples_split']}")
    print(f"   min_samples_leaf: {mejor_acc['min_samples_leaf']}")
    print(f"   Accuracy: {mejor_acc['accuracy_val']:.4f}")
    
    # Mejor configuración por precision de validación
    mejor_prec = df_resultados.loc[df_resultados['precision_val'].idxmax()]
    print(f"\nMejor por Precision (Validación):")
    print(f"   Combinación: {mejor_prec['combinacion']}")
    print(f"   max_depth: {mejor_prec['max_depth']}")
    print(f"   min_samples_split: {mejor_prec['min_samples_split']}")
    print(f"   min_samples_leaf: {mejor_prec['min_samples_leaf']}")
    print(f"   Precision: {mejor_prec['precision_val']:.4f}")
    
    # Mejor configuración por recall de validación
    mejor_rec = df_resultados.loc[df_resultados['recall_val'].idxmax()]
    print(f"\nMejor por Recall (Validación):")
    print(f"   Combinación: {mejor_rec['combinacion']}")
    print(f"   max_depth: {mejor_rec['max_depth']}")
    print(f"   min_samples_split: {mejor_rec['min_samples_split']}")
    print(f"   min_samples_leaf: {mejor_rec['min_samples_leaf']}")
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
    print(f"   max_depth: {mejor_balanceado['max_depth']}")
    print(f"   min_samples_split: {mejor_balanceado['min_samples_split']}")
    print(f"   min_samples_leaf: {mejor_balanceado['min_samples_leaf']}")
    print(f"   Score Balanceado: {mejor_balanceado['score_balanceado']:.4f}")
    print(f"   Accuracy: {mejor_balanceado['accuracy_val']:.4f}")
    print(f"   Precision: {mejor_balanceado['precision_val']:.4f}")
    print(f"   Recall: {mejor_balanceado['recall_val']:.4f}")
    
    # Análisis de overfitting
    print(f"\nANÁLISIS DE OVERFITTING:")
    print("-" * 30)
    
    # Calcular diferencia entre train y val
    df_resultados['diferencia_acc'] = df_resultados['accuracy_train'] - df_resultados['accuracy_val']
    df_resultados['diferencia_prec'] = df_resultados['precision_train'] - df_resultados['precision_val']
    df_resultados['diferencia_rec'] = df_resultados['recall_train'] - df_resultados['recall_val']
    
    # Encontrar configuración con menor overfitting
    menor_overfitting = df_resultados.loc[df_resultados['diferencia_acc'].idxmin()]
    print(f"Menor Overfitting (Accuracy):")
    print(f"   Combinación: {menor_overfitting['combinacion']}")
    print(f"   max_depth: {menor_overfitting['max_depth']}")
    print(f"   min_samples_split: {menor_overfitting['min_samples_split']}")
    print(f"   min_samples_leaf: {menor_overfitting['min_samples_leaf']}")
    print(f"   Diferencia Train-Val: {menor_overfitting['diferencia_acc']:.4f}")
    
    # Top 5 mejores configuraciones
    print(f"\nTOP 5 MEJORES CONFIGURACIONES (por Accuracy Validación):")
    print("-" * 60)
    
    top_5 = df_resultados.nlargest(5, 'accuracy_val')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        max_depth_str = str(row['max_depth']) if row['max_depth'] is not None else 'None'
        print(f"{i}. Comb {row['combinacion']}: "
              f"max_depth={max_depth_str}, "
              f"min_split={row['min_samples_split']}, "
              f"min_leaf={row['min_samples_leaf']} "
              f"-> Acc: {row['accuracy_val']:.4f}")
    
    # Paso 8: Resumen final
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
