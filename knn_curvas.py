#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar curvas y métricas del modelo KNN
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
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

def main():
    """
    Función principal que ejecuta todo el análisis del modelo KNN
    """
    print("=" * 70)
    print("ANÁLISIS DE CURVAS Y MÉTRICAS - MODELO KNN")
    print("=" * 70)
    
    # === CARGA DE DATOS ===
    print("\n[INFO] Cargando datasets...")
    
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
        return False
    
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
    
    # === ENTRENAMIENTO DEL MODELO ===
    print("\n[INFO] Entrenando modelo KNN...")
    
    # Configuración del modelo según resultados anteriores
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    # Entrenar modelo
    knn_model.fit(X_train, y_train)
    print("[OK] Modelo KNN entrenado con k=5")
    
    # === VALIDACIÓN CRUZADA ===
    print("\n[INFO] Realizando validación cruzada...")
    
    # Validación cruzada con k=10
    cv_scores_acc = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores_prec = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='precision_macro')
    cv_scores_rec = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='recall_macro')
    cv_scores_f1 = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='f1_macro')
    
    print(f"[OK] Validación cruzada completada:")
    print(f"   Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    print(f"   Precision: {cv_scores_prec.mean():.4f} (+/- {cv_scores_prec.std() * 2:.4f})")
    print(f"   Recall: {cv_scores_rec.mean():.4f} (+/- {cv_scores_rec.std() * 2:.4f})")
    print(f"   F1-Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
    
    # === ANÁLISIS DE VALIDACIÓN CRUZADA ===
    print("\n[ANÁLISIS] Validación Cruzada (10 folds):")
    print("-" * 60)
    print(f"  Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    print(f"    Rango: [{cv_scores_acc.min():.4f}, {cv_scores_acc.max():.4f}]")
    print(f"    Coeficiente de variación: {(cv_scores_acc.std()/cv_scores_acc.mean()*100):.2f}%")
    
    print(f"\n  Precision: {cv_scores_prec.mean():.4f} (+/- {cv_scores_prec.std() * 2:.4f})")
    print(f"    Rango: [{cv_scores_prec.min():.4f}, {cv_scores_prec.max():.4f}]")
    
    print(f"\n  Recall: {cv_scores_rec.mean():.4f} (+/- {cv_scores_rec.std() * 2:.4f})")
    print(f"    Rango: [{cv_scores_rec.min():.4f}, {cv_scores_rec.max():.4f}]")
    
    print(f"\n  F1-Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
    print(f"    Rango: [{cv_scores_f1.min():.4f}, {cv_scores_f1.max():.4f}]")
    
    # Crear gráfico de validación cruzada con labels más grandes
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({'font.size': 12})
    
    # Subplot 1: Distribución de scores por fold
    plt.subplot(2, 2, 1)
    folds = range(1, 11)
    plt.plot(folds, cv_scores_acc, 'o-', label='Accuracy', alpha=0.8, linewidth=2.5, markersize=8)
    plt.plot(folds, cv_scores_prec, 's-', label='Precision', alpha=0.8, linewidth=2.5, markersize=8)
    plt.plot(folds, cv_scores_rec, '^-', label='Recall', alpha=0.8, linewidth=2.5, markersize=8)
    plt.plot(folds, cv_scores_f1, 'd-', label='F1-Score', alpha=0.8, linewidth=2.5, markersize=8)
    plt.xlabel('Fold', fontsize=13, fontweight='bold')
    plt.ylabel('Score', fontsize=13, fontweight='bold')
    plt.title('Scores por Fold - Validación Cruzada', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=11, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(folds, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Subplot 2: Boxplot de métricas
    plt.subplot(2, 2, 2)
    data_cv = [cv_scores_acc, cv_scores_prec, cv_scores_rec, cv_scores_f1]
    labels_cv = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bp = plt.boxplot(data_cv, labels=labels_cv, patch_artist=True, widths=0.6)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    plt.ylabel('Score', fontsize=13, fontweight='bold')
    plt.title('Distribución de Métricas - Validación Cruzada', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(fontsize=10)
    
    # Subplot 3: Comparación de medias
    plt.subplot(2, 2, 3)
    means = [cv_scores_acc.mean(), cv_scores_prec.mean(), cv_scores_rec.mean(), cv_scores_f1.mean()]
    stds = [cv_scores_acc.std(), cv_scores_prec.std(), cv_scores_rec.std(), cv_scores_f1.std()]
    bars = plt.bar(labels_cv, means, yerr=stds, capsize=8, alpha=0.7, color=colors, 
                   edgecolor='black', linewidth=1.5)
    plt.ylabel('Score Promedio', fontsize=13, fontweight='bold')
    plt.title('Métricas Promedio con Desviación Estándar', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(fontsize=10)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Heatmap de correlación entre métricas
    plt.subplot(2, 2, 4)
    cv_data = np.array([cv_scores_acc, cv_scores_prec, cv_scores_rec, cv_scores_f1])
    correlation_matrix = np.corrcoef(cv_data)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=labels_cv, yticklabels=labels_cv,
                fmt='.3f', annot_kws={'size': 12, 'weight': 'bold'},
                cbar_kws={'label': 'Correlación'})
    plt.title('Correlación entre Métricas', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4, w_pad=4)
    plt.show()
    
    # === CURVAS ROC Y PR ===
    print("\n[INFO] Generando curvas ROC y Precision-Recall...")
    
    # Binarizar las etiquetas para ROC multiclase
    y_train_bin = label_binarize(y_train, classes=clases_unicas)
    y_val_bin = label_binarize(y_val, classes=clases_unicas)
    
    # Obtener probabilidades de predicción
    y_train_proba = knn_model.predict_proba(X_train)
    y_val_proba = knn_model.predict_proba(X_val)
    
    # Calcular ROC AUC para cada clase
    fpr_train = dict()
    tpr_train = dict()
    roc_auc_train = dict()
    
    fpr_val = dict()
    tpr_val = dict()
    roc_auc_val = dict()
    
    for i, clase in enumerate(clases_unicas):
        # Entrenamiento
        fpr_train[clase], tpr_train[clase], _ = roc_curve(y_train_bin[:, i], y_train_proba[:, i])
        roc_auc_train[clase] = auc(fpr_train[clase], tpr_train[clase])
        
        # Validación
        fpr_val[clase], tpr_val[clase], _ = roc_curve(y_val_bin[:, i], y_val_proba[:, i])
        roc_auc_val[clase] = auc(fpr_val[clase], tpr_val[clase])
    
    # Calcular ROC AUC promedio
    roc_auc_train_avg = np.mean(list(roc_auc_train.values()))
    roc_auc_val_avg = np.mean(list(roc_auc_val.values()))
    
    # === ANÁLISIS DE CURVAS ROC ===
    print("\n[ANÁLISIS] Curvas ROC - Entrenamiento:")
    print("-" * 60)
    for clase in clases_unicas:
        print(f"  {clase}: ROC AUC = {roc_auc_train[clase]:.4f}")
        print(f"    Interpretación: {'Excelente' if roc_auc_train[clase] > 0.9 else 'Bueno' if roc_auc_train[clase] > 0.8 else 'Regular' if roc_auc_train[clase] > 0.7 else 'Pobre'}")
    print(f"  ROC AUC Promedio Entrenamiento: {roc_auc_train_avg:.4f}")
    
    print("\n[ANÁLISIS] Curvas ROC - Validación:")
    print("-" * 60)
    for clase in clases_unicas:
        print(f"  {clase}: ROC AUC = {roc_auc_val[clase]:.4f}")
        print(f"    Interpretación: {'Excelente' if roc_auc_val[clase] > 0.9 else 'Bueno' if roc_auc_val[clase] > 0.8 else 'Regular' if roc_auc_val[clase] > 0.7 else 'Pobre'}")
    print(f"  ROC AUC Promedio Validación: {roc_auc_val_avg:.4f}")
    
    # Calcular curvas Precision-Recall para análisis
    precision_recall_data = {}
    for i, clase in enumerate(clases_unicas):
        precision_train, recall_train, _ = precision_recall_curve(y_train_bin[:, i], y_train_proba[:, i])
        precision_recall_data[clase] = {
            'precision': precision_train,
            'recall': recall_train,
            'ap_score': np.trapz(precision_train, recall_train)
        }
    
    print("\n[ANÁLISIS] Curvas Precision-Recall:")
    print("-" * 60)
    for clase in clases_unicas:
        ap_score = precision_recall_data[clase]['ap_score']
        print(f"  {clase}: AP Score = {ap_score:.4f}")
        print(f"    Interpretación: {'Excelente' if ap_score > 0.9 else 'Bueno' if ap_score > 0.8 else 'Regular' if ap_score > 0.7 else 'Pobre'}")
    
    # Crear gráfico de curvas ROC con labels más grandes y mejor espaciado
    plt.figure(figsize=(18, 12))
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
    
    # Subplot 1: Curvas ROC - Entrenamiento
    plt.subplot(2, 3, 1)
    for clase in clases_unicas:
        plt.plot(fpr_train[clase], tpr_train[clase], linewidth=2.5,
                label=f'{clase} (AUC = {roc_auc_train[clase]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('Curva ROC - Entrenamiento', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4, w_pad=4)
    
    # Subplot 2: Curvas ROC - Validación
    plt.subplot(2, 3, 2)
    for clase in clases_unicas:
        plt.plot(fpr_val[clase], tpr_val[clase], linewidth=2.5,
                label=f'{clase} (AUC = {roc_auc_val[clase]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('Curva ROC - Validación', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Curvas Precision-Recall
    plt.subplot(2, 3, 3)
    for i, clase in enumerate(clases_unicas):
        precision_train, recall_train, _ = precision_recall_curve(y_train_bin[:, i], y_train_proba[:, i])
        ap_score = precision_recall_data[clase]['ap_score']
        plt.plot(recall_train, precision_train, linewidth=2.5,
                label=f'{clase} (AP = {ap_score:.3f})')
    plt.xlabel('Recall', fontsize=13, fontweight='bold')
    plt.ylabel('Precision', fontsize=13, fontweight='bold')
    plt.title('Curva Precision-Recall - Entrenamiento', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Comparación TPR/FPR
    plt.subplot(2, 3, 4)
    tpr_values = [tpr_val[clase][-1] for clase in clases_unicas]
    fpr_values = [fpr_val[clase][-1] for clase in clases_unicas]
    plt.scatter(fpr_values, tpr_values, s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
    for i, clase in enumerate(clases_unicas):
        plt.annotate(clase, (fpr_values[i], tpr_values[i]), 
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('TPR vs FPR - Validación', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: ROC AUC por clase
    plt.subplot(2, 3, 5)
    clases_nombres = list(roc_auc_val.keys())
    auc_valores = list(roc_auc_val.values())
    bars = plt.bar(clases_nombres, auc_valores, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.ylabel('ROC AUC', fontsize=13, fontweight='bold')
    plt.title('ROC AUC por Clase - Validación', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha='right')
    for bar, valor in zip(bars, auc_valores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{valor:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 6: Comparación Train vs Val
    plt.subplot(2, 3, 6)
    x_pos = np.arange(len(clases_unicas))
    width = 0.35
    bars1 = plt.bar(x_pos - width/2, list(roc_auc_train.values()), width, 
            label='Entrenamiento', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x_pos + width/2, list(roc_auc_val.values()), width, 
            label='Validación', alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.xlabel('Clases', fontsize=13, fontweight='bold')
    plt.ylabel('ROC AUC', fontsize=13, fontweight='bold')
    plt.title('Comparación ROC AUC', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x_pos, clases_unicas, rotation=15, ha='right', fontsize=11)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4, w_pad=4)
    plt.show()
    
    print(f"[OK] Curvas ROC generadas:")
    print(f"   ROC AUC promedio - Entrenamiento: {roc_auc_train_avg:.4f}")
    print(f"   ROC AUC promedio - Validación: {roc_auc_val_avg:.4f}")
    
    # === MATRIZ DE CONFUSIÓN ===
    print("\n[INFO] Generando matriz de confusión...")
    
    # Predicciones
    y_train_pred = knn_model.predict(X_train)
    y_val_pred = knn_model.predict(X_val)
    
    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred, labels=clases_unicas)
    cm_val = confusion_matrix(y_val, y_val_pred, labels=clases_unicas)
    
    # === ANÁLISIS DE MATRIZ DE CONFUSIÓN ===
    print("\n[ANÁLISIS] Matriz de Confusión - Entrenamiento:")
    print("-" * 60)
    for i, clase_real in enumerate(clases_unicas):
        total_real = cm_train[i, :].sum()
        correctos = cm_train[i, i]
        porcentaje = (correctos / total_real * 100) if total_real > 0 else 0
        print(f"  {clase_real}: {correctos}/{total_real} correctos ({porcentaje:.2f}%)")
        for j, clase_pred in enumerate(clases_unicas):
            if i != j and cm_train[i, j] > 0:
                print(f"    - Confundido con {clase_pred}: {cm_train[i, j]} veces")
    
    print("\n[ANÁLISIS] Matriz de Confusión - Validación:")
    print("-" * 60)
    for i, clase_real in enumerate(clases_unicas):
        total_real = cm_val[i, :].sum()
        correctos = cm_val[i, i]
        porcentaje = (correctos / total_real * 100) if total_real > 0 else 0
        print(f"  {clase_real}: {correctos}/{total_real} correctos ({porcentaje:.2f}%)")
        for j, clase_pred in enumerate(clases_unicas):
            if i != j and cm_val[i, j] > 0:
                print(f"    - Confundido con {clase_pred}: {cm_val[i, j]} veces")
    
    # Crear gráfico de matrices de confusión con labels más grandes
    plt.figure(figsize=(16, 7))
    plt.rcParams.update({'font.size': 12})
    
    # Matriz de confusión - Entrenamiento
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases_unicas, yticklabels=clases_unicas,
                annot_kws={'size': 13, 'weight': 'bold'},
                cbar_kws={'label': 'Cantidad'})
    plt.title('Matriz de Confusión - Entrenamiento', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=13, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Matriz de confusión - Validación
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases_unicas, yticklabels=clases_unicas,
                annot_kws={'size': 13, 'weight': 'bold'},
                cbar_kws={'label': 'Cantidad'})
    plt.title('Matriz de Confusión - Validación', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=13, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3, w_pad=4)
    plt.show()
    
    # Calcular métricas de validación
    accuracy_val = accuracy_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred, average='macro')
    recall_val = recall_score(y_val, y_val_pred, average='macro')
    f1_val = f1_score(y_val, y_val_pred, average='macro')
    
    print(f"\n[OK] Matriz de confusión generada:")
    print(f"   Accuracy validación: {accuracy_val:.4f}")
    print(f"   Precision validación: {precision_val:.4f}")
    print(f"   Recall validación: {recall_val:.4f}")
    print(f"   F1-Score validación: {f1_val:.4f}")
    
    # === CURVA DE APRENDIZAJE ===
    print("\n[INFO] Generando curva de aprendizaje...")
    
    # Calcular curva de aprendizaje
    train_sizes, train_scores, val_scores = learning_curve(
        knn_model, X_train, y_train, cv=10, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Calcular medias y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # === ANÁLISIS DE CURVA DE APRENDIZAJE ===
    print("\n[ANÁLISIS] Curva de Aprendizaje:")
    print("-" * 60)
    print(f"  Tamaño mínimo de datos (10%):")
    print(f"    Entrenamiento: {train_mean[0]:.4f} (+/- {train_std[0]:.4f})")
    print(f"    Validación: {val_mean[0]:.4f} (+/- {val_std[0]:.4f})")
    print(f"    Diferencia: {abs(train_mean[0] - val_mean[0]):.4f}")
    
    print(f"\n  Tamaño máximo de datos (100%):")
    print(f"    Entrenamiento: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
    print(f"    Validación: {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")
    print(f"    Diferencia: {abs(train_mean[-1] - val_mean[-1]):.4f}")
    
    # Detectar overfitting o underfitting
    gap_final = train_mean[-1] - val_mean[-1]
    if gap_final > 0.1:
        print(f"\n  [ADVERTENCIA] Posible overfitting detectado:")
        print(f"    Gap entre entrenamiento y validación: {gap_final:.4f}")
    elif gap_final < 0.05:
        print(f"\n  [OK] Modelo bien balanceado:")
        print(f"    Gap entre entrenamiento y validación: {gap_final:.4f}")
    
    # Mejora del modelo
    mejora = val_mean[-1] - val_mean[0]
    print(f"\n  Mejora del modelo con más datos:")
    print(f"    Accuracy inicial: {val_mean[0]:.4f}")
    print(f"    Accuracy final: {val_mean[-1]:.4f}")
    print(f"    Mejora: {mejora:.4f} ({mejora*100:.2f}%)")
    
    # Crear gráfico de curva de aprendizaje con labels más grandes
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 12})
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento', 
             linewidth=2.5, markersize=8)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validación', 
             linewidth=2.5, markersize=8)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Curva de Aprendizaje - KNN', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0.2, 1.0)
    
    # Agregar anotaciones para valores importantes
    plt.annotate(f'Inicial: {val_mean[0]:.3f}', 
                xy=(train_sizes[0], val_mean[0]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    plt.annotate(f'Final: {val_mean[-1]:.3f}', 
                xy=(train_sizes[-1], val_mean[-1]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    print("\n[OK] Curva de aprendizaje generada")
    
    # === RESUMEN FINAL ===
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - MODELO KNN")
    print("=" * 70)
    
    print(f"\n📊 MÉTRICAS DE VALIDACIÓN CRUZADA (k=10):")
    print(f"   Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    print(f"   Precision: {cv_scores_prec.mean():.4f} (+/- {cv_scores_prec.std() * 2:.4f})")
    print(f"   Recall: {cv_scores_rec.mean():.4f} (+/- {cv_scores_rec.std() * 2:.4f})")
    print(f"   F1-Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
    
    print(f"\n📊 MÉTRICAS DE VALIDACIÓN:")
    print(f"   Accuracy: {accuracy_val:.4f}")
    print(f"   Precision: {precision_val:.4f}")
    print(f"   Recall: {recall_val:.4f}")
    print(f"   F1-Score: {f1_val:.4f}")
    print(f"   ROC AUC promedio: {roc_auc_val_avg:.4f}")
    
    print(f"\n📊 ROC AUC POR CLASE:")
    for clase in clases_unicas:
        print(f"   {clase}: {roc_auc_val[clase]:.4f}")
    
    print(f"\n📊 CONFIGURACIÓN DEL MODELO:")
    print(f"   Algoritmo: K-Nearest Neighbors")
    print(f"   k: 5")
    print(f"   Features: {len(features)}")
    print(f"   Clases: {len(clases_unicas)}")
    print(f"   Registros entrenamiento: {len(X_train):,}")
    print(f"   Registros validación: {len(X_val):,}")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    # === MEDICIÓN DEL COSTO COMPUTACIONAL ===
    print("\n[INFO] Iniciando medición de costo computacional...")
    
    # Crear modelo nuevo para medición (sin entrenar)
    knn_model_medicion = KNeighborsClassifier(n_neighbors=5)
    
    # Medir costo computacional
    resultados_knn = medir_costo_computacional(
        knn_model_medicion, X_train, y_train, X_val, y_val, "KNN"
    )
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame([resultados_knn])
    
    # Mostrar resumen de costos
    mostrar_resumen_costos(resultados_df)
    
    # Guardar resultados
    guardar_resultados(resultados_df, 'resultados_costos_knn.csv')
    
    return True

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
        'Tiempo_Entrenamiento': tiempo_entrenamiento,
        'Tiempo_Prediccion': tiempo_prediccion,
        'Memoria_MB': memoria_mb,
        'CPU_Percent': cpu_percent
    }
    
    print(f"[OK] Costo computacional medido:")
    print(f"   Tiempo entrenamiento: {tiempo_entrenamiento:.4f} segundos")
    print(f"   Tiempo predicción: {tiempo_prediccion:.4f} segundos")
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
    
    # Subplot 1: Tiempo de entrenamiento
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(modelos, resultados_df['Tiempo_Entrenamiento'], 
                    color='skyblue', alpha=0.7)
    plt.title('Tiempo de Entrenamiento por Modelo')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45)
    for bar, valor in zip(bars1, resultados_df['Tiempo_Entrenamiento']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{valor:.3f}s', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Tiempo de predicción
    plt.subplot(2, 2, 2)
    bars2 = plt.bar(modelos, resultados_df['Tiempo_Prediccion'], 
                    color='lightgreen', alpha=0.7)
    plt.title('Tiempo de Predicción por Modelo')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45)
    for bar, valor in zip(bars2, resultados_df['Tiempo_Prediccion']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{valor:.4f}s', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Uso de memoria
    plt.subplot(2, 2, 3)
    bars3 = plt.bar(modelos, resultados_df['Memoria_MB'], 
                    color='lightcoral', alpha=0.7)
    plt.title('Uso de Memoria por Modelo')
    plt.ylabel('Memoria (MB)')
    plt.xticks(rotation=45)
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
    plt.xticks(rotation=45)
    for bar, valor in zip(bars4, resultados_df['CPU_Percent']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{valor:.1f}%', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Costo Computacional - Comparativa de Modelos', fontsize=16, fontweight='bold')
    plt.tight_layout()
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
    mas_rapido_entrenamiento = resultados_df.loc[resultados_df['Tiempo_Entrenamiento'].idxmin()]
    print(f"Más rápido en entrenamiento: {mas_rapido_entrenamiento['Modelo']} ({mas_rapido_entrenamiento['Tiempo_Entrenamiento']:.4f}s)")
    
    # Modelo más rápido en predicción
    mas_rapido_prediccion = resultados_df.loc[resultados_df['Tiempo_Prediccion'].idxmin()]
    print(f"Más rápido en predicción: {mas_rapido_prediccion['Modelo']} ({mas_rapido_prediccion['Tiempo_Prediccion']:.4f}s)")
    
    # Modelo con menor uso de memoria
    menor_memoria = resultados_df.loc[resultados_df['Memoria_MB'].idxmin()]
    print(f"Menor uso de memoria: {menor_memoria['Modelo']} ({menor_memoria['Memoria_MB']:.2f}MB)")
    
    print("=" * 80)

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
