#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para entrenar modelos Naive Bayes con búsqueda de hiperparámetros
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import RandomizedSearchCV
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
    Función principal que ejecuta todo el proceso de entrenamiento Naive Bayes
    """
    print("=" * 70)
    print("PROGRAMA PARA ENTRENAR MODELOS NAIVE BAYES CON BÚSQUEDA DE HIPERPARÁMETROS")
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
    
    # Definir parámetros para cada modelo
    param_grid_gaussian = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    
    param_grid_complement = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    print("   Hiperparámetros a evaluar:")
    print(f"   - GaussianNB var_smoothing: {param_grid_gaussian['var_smoothing']}")
    print(f"   - ComplementNB alpha: {param_grid_complement['alpha']}")
    
    # Calcular número total de combinaciones
    total_combinaciones = len(param_grid_gaussian['var_smoothing']) + len(param_grid_complement['alpha'])
    print(f"   - Total de combinaciones posibles: {total_combinaciones}")
    print(f"   - Iteraciones de búsqueda aleatoria: 10")
    
    print()
    
    # Paso 5: Realizar búsqueda aleatoria para GaussianNB
    print("[INFO] Iniciando búsqueda aleatoria para GaussianNB...")
    
    resultados = []
    
    try:
        # Crear modelo GaussianNB base
        gaussian_base = GaussianNB()
        
        # Configurar RandomizedSearchCV para GaussianNB
        random_search_gaussian = RandomizedSearchCV(
            estimator=gaussian_base,
            param_distributions=param_grid_gaussian,
            n_iter=5,  # Evaluar todos los valores de var_smoothing
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,  # Usar todos los cores disponibles
            verbose=1
        )
        
        # Realizar búsqueda
        print("   Ejecutando RandomizedSearchCV para GaussianNB...")
        random_search_gaussian.fit(X_train, y_train)
        
        print("[OK] Búsqueda aleatoria para GaussianNB completada")
        
    except Exception as e:
        print(f"[ERROR] Error en búsqueda aleatoria GaussianNB: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Realizar búsqueda aleatoria para ComplementNB
    print("[INFO] Iniciando búsqueda aleatoria para ComplementNB...")
    
    try:
        # Crear modelo ComplementNB base
        complement_base = ComplementNB()
        
        # Configurar RandomizedSearchCV para ComplementNB
        random_search_complement = RandomizedSearchCV(
            estimator=complement_base,
            param_distributions=param_grid_complement,
            n_iter=5,  # Evaluar todos los valores de alpha
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,  # Usar todos los cores disponibles
            verbose=1
        )
        
        # Realizar búsqueda
        print("   Ejecutando RandomizedSearchCV para ComplementNB...")
        random_search_complement.fit(X_train, y_train)
        
        print("[OK] Búsqueda aleatoria para ComplementNB completada")
        
    except Exception as e:
        print(f"[ERROR] Error en búsqueda aleatoria ComplementNB: {str(e)}")
        return False
    
    print()
    
    # Paso 7: Evaluar cada combinación probada
    print("[INFO] Evaluando combinaciones de hiperparámetros...")
    
    # Evaluar GaussianNB
    print("\n" + "="*50)
    print("EVALUACIÓN DE GAUSSIANNB")
    print("="*50)
    
    resultados_cv_gaussian = random_search_gaussian.cv_results_
    
    for i in range(len(resultados_cv_gaussian['params'])):
        params = resultados_cv_gaussian['params'][i]
        mean_score = resultados_cv_gaussian['mean_test_score'][i]
        std_score = resultados_cv_gaussian['std_test_score'][i]
        
        print(f"\nGaussianNB - Combinación {i+1}:")
        print(f"   var_smoothing: {params['var_smoothing']}")
        print(f"   CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        try:
            # Crear modelo con estos hiperparámetros
            gaussian_model = GaussianNB(var_smoothing=params['var_smoothing'])
            
            # Entrenar modelo
            gaussian_model.fit(X_train, y_train)
            
            # Predicciones en entrenamiento
            y_train_pred = gaussian_model.predict(X_train)
            
            # Predicciones en validación
            y_val_pred = gaussian_model.predict(X_val)
            
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
                'modelo': 'GaussianNB',
                'combinacion': i+1,
                'hiperparametro': 'var_smoothing',
                'valor': params['var_smoothing'],
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
    
    # Evaluar ComplementNB
    print("\n" + "="*50)
    print("EVALUACIÓN DE COMPLEMENTNB")
    print("="*50)
    
    resultados_cv_complement = random_search_complement.cv_results_
    
    for i in range(len(resultados_cv_complement['params'])):
        params = resultados_cv_complement['params'][i]
        mean_score = resultados_cv_complement['mean_test_score'][i]
        std_score = resultados_cv_complement['std_test_score'][i]
        
        print(f"\nComplementNB - Combinación {i+1}:")
        print(f"   alpha: {params['alpha']}")
        print(f"   CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        try:
            # Crear modelo con estos hiperparámetros
            complement_model = ComplementNB(alpha=params['alpha'])
            
            # Entrenar modelo
            complement_model.fit(X_train, y_train)
            
            # Predicciones en entrenamiento
            y_train_pred = complement_model.predict(X_train)
            
            # Predicciones en validación
            y_val_pred = complement_model.predict(X_val)
            
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
                'modelo': 'ComplementNB',
                'combinacion': i+1,
                'hiperparametro': 'alpha',
                'valor': params['alpha'],
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
    
    # Paso 8: Resumen tabulado
    print("=" * 70)
    print("RESUMEN TABULADO - TODAS LAS COMBINACIONES EVALUADAS")
    print("=" * 70)
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar tabla de resultados
    print("\nTabla de Resultados:")
    print("-" * 130)
    print(f"{'Modelo':<12} {'Comb':<4} {'Hiperparam':<12} {'Valor':<10} {'CV Score':<10} {'Acc Train':<10} {'Acc Val':<10} {'Prec Train':<12} {'Prec Val':<12} {'Rec Train':<12} {'Rec Val':<12}")
    print("-" * 130)
    
    for _, row in df_resultados.iterrows():
        print(f"{row['modelo']:<12} "
              f"{row['combinacion']:<4} "
              f"{row['hiperparametro']:<12} "
              f"{str(row['valor']):<10} "
              f"{row['cv_score']:<10.4f} "
              f"{row['accuracy_train']:<10.4f} "
              f"{row['accuracy_val']:<10.4f} "
              f"{row['precision_train']:<12.4f} "
              f"{row['precision_val']:<12.4f} "
              f"{row['recall_train']:<12.4f} "
              f"{row['recall_val']:<12.4f}")
    
    print("-" * 130)
    
    # Paso 9: Identificar mejor configuración
    print("\nANÁLISIS DE LA MEJOR CONFIGURACIÓN:")
    print("-" * 50)
    
    # Mejor configuración por accuracy de validación
    mejor_acc = df_resultados.loc[df_resultados['accuracy_val'].idxmax()]
    print(f"Mejor por Accuracy (Validación):")
    print(f"   Modelo: {mejor_acc['modelo']}")
    print(f"   Combinación: {mejor_acc['combinacion']}")
    print(f"   {mejor_acc['hiperparametro']}: {mejor_acc['valor']}")
    print(f"   Accuracy: {mejor_acc['accuracy_val']:.4f}")
    
    # Mejor configuración por precision de validación
    mejor_prec = df_resultados.loc[df_resultados['precision_val'].idxmax()]
    print(f"\nMejor por Precision (Validación):")
    print(f"   Modelo: {mejor_prec['modelo']}")
    print(f"   Combinación: {mejor_prec['combinacion']}")
    print(f"   {mejor_prec['hiperparametro']}: {mejor_prec['valor']}")
    print(f"   Precision: {mejor_prec['precision_val']:.4f}")
    
    # Mejor configuración por recall de validación
    mejor_rec = df_resultados.loc[df_resultados['recall_val'].idxmax()]
    print(f"\nMejor por Recall (Validación):")
    print(f"   Modelo: {mejor_rec['modelo']}")
    print(f"   Combinación: {mejor_rec['combinacion']}")
    print(f"   {mejor_rec['hiperparametro']}: {mejor_rec['valor']}")
    print(f"   Recall: {mejor_rec['recall_val']:.4f}")
    
    # Mejor configuración balanceada
    df_resultados['score_balanceado'] = (
        df_resultados['accuracy_val'] + 
        df_resultados['precision_val'] + 
        df_resultados['recall_val']
    ) / 3
    
    mejor_balanceado = df_resultados.loc[df_resultados['score_balanceado'].idxmax()]
    print(f"\nMejor Configuración Balanceada:")
    print(f"   Modelo: {mejor_balanceado['modelo']}")
    print(f"   Combinación: {mejor_balanceado['combinacion']}")
    print(f"   {mejor_balanceado['hiperparametro']}: {mejor_balanceado['valor']}")
    print(f"   Score Balanceado: {mejor_balanceado['score_balanceado']:.4f}")
    print(f"   Accuracy: {mejor_balanceado['accuracy_val']:.4f}")
    print(f"   Precision: {mejor_balanceado['precision_val']:.4f}")
    print(f"   Recall: {mejor_balanceado['recall_val']:.4f}")
    
    # Análisis por modelo
    print(f"\nANÁLISIS POR MODELO:")
    print("-" * 30)
    
    # Mejor GaussianNB
    gaussian_results = df_resultados[df_resultados['modelo'] == 'GaussianNB']
    if not gaussian_results.empty:
        mejor_gaussian = gaussian_results.loc[gaussian_results['accuracy_val'].idxmax()]
        print(f"Mejor GaussianNB:")
        print(f"   {mejor_gaussian['hiperparametro']}: {mejor_gaussian['valor']}")
        print(f"   Accuracy: {mejor_gaussian['accuracy_val']:.4f}")
    
    # Mejor ComplementNB
    complement_results = df_resultados[df_resultados['modelo'] == 'ComplementNB']
    if not complement_results.empty:
        mejor_complement = complement_results.loc[complement_results['accuracy_val'].idxmax()]
        print(f"\nMejor ComplementNB:")
        print(f"   {mejor_complement['hiperparametro']}: {mejor_complement['valor']}")
        print(f"   Accuracy: {mejor_complement['accuracy_val']:.4f}")
    
    # Paso 10: Resumen final
    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Combinaciones evaluadas: {len(resultados)}")
    print(f"[INFO] Features utilizadas: {len(features)}")
    print(f"[INFO] Clases objetivo: {len(clases_unicas)}")
    print(f"[INFO] Registros entrenamiento: {len(X_train):,}")
    print(f"[INFO] Registros validación: {len(X_val):,}")
    print(f"[INFO] Mejor modelo recomendado: {mejor_balanceado['modelo']}")
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
