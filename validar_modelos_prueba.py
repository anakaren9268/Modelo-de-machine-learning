#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación final de modelos con el dataset de prueba
Autor: Asistente IA
Fecha: 2025

Este script entrena los cuatro modelos seleccionados previamente utilizando
los conjuntos de entrenamiento y validación combinados, y evalúa su
desempeño sobre el dataset de prueba. Para cada modelo se calculan las
métricas de rendimiento más relevantes y se genera una matriz de confusión
con el mismo estilo visual que en los análisis anteriores.
"""

import os
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# === CONFIGURACIÓN GLOBAL ===
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

RUTA_BASE = r"C:\\Users\\FRCemco04\\Documents\\Proyecto ML\\Porgramas Python"
RUTA_ENTRENAMIENTO = os.path.join(RUTA_BASE, "dataset_entrenamiento.csv")
RUTA_VALIDACION = os.path.join(RUTA_BASE, "dataset_validacion.csv")
RUTA_PRUEBA = os.path.join(RUTA_BASE, "dataset_prueba.csv")

FEATURES = [
    "HarshBrakingPer100Km_Norm",
    "HarshAccelerationPer100Km_Norm",
    "HarshTurningPer100Km_Norm",
    "IdlingRatePercentOfIgnitionTime_Norm",
    "SpeedOver95Per100Km_Norm",
    "FuelUnder50PercentPer100Km_Norm",
    "RPMOver1600Per100Km_Norm",
    "FueraDeRutaFRPer100Km_Norm",
]

TARGET = "Categoria_Conductor"

MODELOS = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (C=100, kernel=linear, gamma=0.001)": SVC(
        C=100.0, kernel="linear", gamma=0.001, probability=True, random_state=42
    ),
    "Naive Bayes (GaussianNB)": GaussianNB(var_smoothing=1e-05),
    "Árbol de Decisión": DecisionTreeClassifier(
        max_depth=None, min_samples_split=20, min_samples_leaf=5, random_state=42
    ),
}


def cargar_datos() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets de entrenamiento, validación y prueba."""

    print("=" * 80)
    print("VALIDACIÓN FINAL CON EL DATASET DE PRUEBA")
    print("=" * 80)

    print("\n[INFO] Cargando datasets...")
    try:
        df_train = pd.read_csv(RUTA_ENTRENAMIENTO)
        df_val = pd.read_csv(RUTA_VALIDACION)
        df_test = pd.read_csv(RUTA_PRUEBA)
    except Exception as exc:  # pragma: no cover - manejo de errores en ejecución
        raise FileNotFoundError(
            f"No fue posible cargar los archivos necesarios: {exc}"
        ) from exc

    faltantes = [col for col in FEATURES + [TARGET] if col not in df_train.columns]
    if faltantes:
        raise ValueError(
            "Las siguientes columnas faltan en los datasets de entrenamiento/validación: "
            + ", ".join(faltantes)
        )

    faltantes_test = [col for col in FEATURES + [TARGET] if col not in df_test.columns]
    if faltantes_test:
        raise ValueError(
            "Las siguientes columnas faltan en el dataset de prueba: "
            + ", ".join(faltantes_test)
        )

    print(f"[OK] Registros entrenamiento: {len(df_train):,}")
    print(f"[OK] Registros validación: {len(df_val):,}")
    print(f"[OK] Registros prueba: {len(df_test):,}")

    return df_train, df_val, df_test


def preparar_conjuntos(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Prepara matrices de características y etiquetas para cada conjunto."""

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_val = df_val[FEATURES]
    y_val = df_val[TARGET]
    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]

    # Combinar entrenamiento y validación para el modelo final
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    print("\n[INFO] Combinando entrenamiento + validación para el ajuste final...")
    print(f"[OK] Tamaño conjunto total de entrenamiento: {len(X_train_full):,} registros")

    return {
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_test": X_test,
        "y_test": y_test,
    }


def graficar_matriz_confusion(
    matriz: np.ndarray,
    etiquetas: np.ndarray,
    titulo: str,
    cmap: str = "Blues",
) -> None:
    """Genera una matriz de confusión con estilo consistente."""

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 12})

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
        xticklabels=etiquetas,
        yticklabels=etiquetas,
    )
    plt.title(titulo, fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Predicción", fontsize=14, fontweight="bold")
    plt.ylabel("Real", fontsize=14, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    plt.show()


def evaluar_modelos(conjuntos: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    """Entrena y evalúa todos los modelos sobre el dataset de prueba."""

    resultados = []
    X_train_full = conjuntos["X_train_full"]
    y_train_full = conjuntos["y_train_full"]
    X_test = conjuntos["X_test"]
    y_test = conjuntos["y_test"]

    clases = sorted(y_train_full.unique())
    print(f"\n[INFO] Clases detectadas: {clases}")

    for nombre_modelo, modelo in MODELOS.items():
        print("\n" + "-" * 80)
        print(f"[MODELO] {nombre_modelo}")
        print("-" * 80)

        # === ENTRENAMIENTO ===
        print("[INFO] Entrenando modelo con el conjunto total...")
        modelo.fit(X_train_full, y_train_full)
        print("[OK] Entrenamiento completado.")

        # === PREDICCIÓN ===
        print("[INFO] Generando predicciones sobre el dataset de prueba...")
        y_pred = modelo.predict(X_test)

        # === MÉTRICAS ===
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print("\n[MÉTRICAS - DATASET PRUEBA]")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Precision (macro): {prec:.4f}")
        print(f"  Recall (macro) : {rec:.4f}")
        print(f"  F1-Score (macro): {f1:.4f}")

        print("\n[REPORTE DE CLASIFICACIÓN]")
        print(classification_report(y_test, y_pred, digits=4))

        # === MATRIZ DE CONFUSIÓN ===
        matriz = confusion_matrix(y_test, y_pred, labels=clases)
        graficar_matriz_confusion(
            matriz,
            etiquetas=clases,
            titulo=f"Matriz de Confusión - {nombre_modelo}",
            cmap="coolwarm",
        )

        # Análisis adicional en consola
        aciertos = np.trace(matriz)
        total = matriz.sum()
        print("[ANÁLISIS] Detalles de la matriz de confusión:")
        print(f"  Total de observaciones: {total:,}")
        print(f"  Predicciones correctas: {aciertos:,} ({aciertos / total:.2%})")
        if total - aciertos:
            print(
                f"  Predicciones incorrectas: {total - aciertos:,} "
                f"({(total - aciertos) / total:.2%})"
            )

        resultados.append(
            {
                "Modelo": nombre_modelo,
                "Accuracy": acc,
                "Precision (macro)": prec,
                "Recall (macro)": rec,
                "F1-Score (macro)": f1,
            }
        )

    return pd.DataFrame(resultados)


def mostrar_resumen(df_resultados: pd.DataFrame) -> None:
    """Muestra en consola un resumen tabular y ordenado por Accuracy."""

    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO EN EL DATASET DE PRUEBA")
    print("=" * 80)

    df_ordenado = df_resultados.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    print(df_ordenado.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision (macro)": "{:.4f}".format,
        "Recall (macro)": "{:.4f}".format,
        "F1-Score (macro)": "{:.4f}".format,
    }))

    mejor_modelo = df_ordenado.iloc[0]
    print("\n[CONCLUSIÓN]")
    print(
        "El modelo con mejor desempeño en el dataset de prueba es "
        f"{mejor_modelo['Modelo']} con Accuracy={mejor_modelo['Accuracy']:.4f}, "
        f"Precision={mejor_modelo['Precision (macro)']:.4f}, "
        f"Recall={mejor_modelo['Recall (macro)']:.4f} y F1={mejor_modelo['F1-Score (macro)']:.4f}."
    )


def main() -> None:
    """Ejecuta el flujo completo de validación."""

    df_train, df_val, df_test = cargar_datos()
    conjuntos = preparar_conjuntos(df_train, df_val, df_test)
    df_resultados = evaluar_modelos(conjuntos)
    mostrar_resumen(df_resultados)


if __name__ == "__main__":
    main()


