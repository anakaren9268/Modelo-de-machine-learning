#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para generar heatmap de matriz de correlación
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import sys

def main():
    """
    Función principal que ejecuta todo el proceso de generación del heatmap
    """
    print("=" * 70)
    print("PROGRAMA PARA GENERAR HEATMAP DE MATRIZ DE CORRELACIÓN")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_clasificado.csv")
    ruta_archivo_salida = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\heatmap_correlacion.png")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo de salida: {ruta_archivo_salida}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_clasificado.csv no existe.")
        print(f"   Ruta buscada: {ruta_archivo_entrada}")
        print("   Asegúrate de que el archivo existe antes de ejecutar este programa.")
        return False
    
    print("[OK] Archivo de entrada encontrado")
    
    # Paso 2: Cargar el archivo CSV
    print("[INFO] Cargando archivo...")
    try:
        df = pd.read_csv(ruta_archivo_entrada)
        print("[OK] Archivo cargado correctamente")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        print(f"   Dimensiones: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Error al cargar el archivo: {str(e)}")
        return False
    
    print()
    
    # Paso 3: Detectar columnas numéricas
    print("[INFO] Detectando columnas numéricas...")
    
    # Obtener columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        print("[ERROR] No se encontraron columnas numéricas en el dataset.")
        return False
    
    print(f"[OK] Se encontraron {len(columnas_numericas)} columnas numéricas:")
    for i, col in enumerate(columnas_numericas, 1):
        print(f"   {i:2d}. {col}")
    
    # Verificar que score_final esté incluido
    if 'score_final' in columnas_numericas:
        print("   [OK] Columna 'score_final' incluida en el análisis")
    else:
        print("   [WARNING] Columna 'score_final' no encontrada")
    
    print()
    
    # Paso 4: Preparar datos para correlación
    print("[INFO] Preparando datos para análisis de correlación...")
    
    # Seleccionar solo las columnas numéricas
    df_numerico = df[columnas_numericas]
    
    # Eliminar filas con valores NaN para el cálculo de correlación
    df_limpio = df_numerico.dropna()
    
    registros_originales = len(df_numerico)
    registros_limpios = len(df_limpio)
    
    if registros_limpios < registros_originales:
        print(f"   [INFO] Se eliminaron {registros_originales - registros_limpios:,} registros con valores faltantes")
    
    print(f"[OK] Datos preparados: {registros_limpios:,} registros válidos")
    print()
    
    # Paso 5: Calcular matriz de correlación
    print("[INFO] Calculando matriz de correlación...")
    
    try:
        matriz_correlacion = df_limpio.corr()
        print("[OK] Correlación calculada correctamente")
        print(f"   Dimensiones de la matriz: {matriz_correlacion.shape}")
        
        # Mostrar estadísticas básicas de la correlación
        correlaciones_no_diagonal = matriz_correlacion.values[np.triu_indices_from(matriz_correlacion.values, k=1)]
        print(f"   Correlación mínima: {correlaciones_no_diagonal.min():.3f}")
        print(f"   Correlación máxima: {correlaciones_no_diagonal.max():.3f}")
        print(f"   Correlación promedio: {correlaciones_no_diagonal.mean():.3f}")
        
    except Exception as e:
        print(f"[ERROR] Error al calcular correlación: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Generar heatmap
    print("[INFO] Generando heatmap...")
    
    try:
        # Configurar el tamaño de la figura dinámicamente
        n_columnas = len(columnas_numericas)
        # Calcular tamaño basado en número de columnas y longitud de nombres
        longitud_max_nombre = max([len(col) for col in columnas_numericas])
        tamaño_base = max(12, n_columnas * 0.8)
        tamaño_ajustado = max(tamaño_base, longitud_max_nombre * 0.15)
        tamaño_figura = min(tamaño_ajustado, 20)  # Limitar tamaño máximo
        
        plt.figure(figsize=(tamaño_figura, tamaño_figura))
        
        # Configurar parámetros de fuente globales
        plt.rcParams.update({'font.size': 11})
        
        # Crear máscara para la diagonal superior (opcional)
        mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))
        
        # Calcular tamaño de fuente para anotaciones basado en número de columnas
        if n_columnas <= 10:
            annot_size = 12
            cbar_shrink = 0.8
        elif n_columnas <= 15:
            annot_size = 10
            cbar_shrink = 0.7
        else:
            annot_size = 8
            cbar_shrink = 0.6
        
        # Generar heatmap con seaborn con mejoras
        sns.heatmap(
            matriz_correlacion,
            mask=mascara,
            annot=True,  # Mostrar valores de correlación
            annot_kws={'size': annot_size, 'weight': 'bold'},  # Tamaño y peso de anotaciones
            cmap='RdYlBu_r',  # Colormap rojo-amarillo-azul invertido
            center=0,  # Centrar en 0
            square=True,  # Forma cuadrada
            fmt='.2f',  # Formato de 2 decimales
            cbar_kws={"shrink": cbar_shrink, "label": "Correlación"},  # Barra de color con label
            linewidths=0.8,  # Líneas entre celdas más visibles
            linecolor='white',  # Color de las líneas
            xticklabels=True,  # Mostrar labels en X
            yticklabels=True   # Mostrar labels en Y
        )
        
        # Configurar título y etiquetas con tamaño más grande
        plt.title('Matriz de Correlación - Variables Numéricas', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('Variables', fontsize=15, fontweight='bold')
        plt.ylabel('Variables', fontsize=15, fontweight='bold')
        
        # Rotar etiquetas para mejor legibilidad y evitar amontonamiento
        # Ajustar rotación según longitud de nombres
        if longitud_max_nombre > 20:
            rotacion_x = 90
            ha_x = 'center'
        elif longitud_max_nombre > 15:
            rotacion_x = 60
            ha_x = 'right'
        else:
            rotacion_x = 45
            ha_x = 'right'
        
        plt.xticks(rotation=rotacion_x, ha=ha_x, fontsize=11)
        plt.yticks(rotation=0, ha='right', fontsize=11)
        
        # Ajustar layout con padding adicional para evitar que se corten las etiquetas
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        
        print("[OK] Heatmap generado correctamente")
        
    except Exception as e:
        print(f"[ERROR] Error al generar heatmap: {str(e)}")
        return False
    
    print()
    
    # Paso 7: Guardar imagen (ANTES de mostrar)
    print("[INFO] Guardando imagen...")
    print(f"   Ubicación: {ruta_archivo_salida}")
    
    try:
        plt.savefig(ruta_archivo_salida, 
                   dpi=300,  # Alta resolución
                   bbox_inches='tight',  # Ajustar bordes
                   facecolor='white',  # Fondo blanco
                   edgecolor='none',  # Sin borde
                   pad_inches=0.5)  # Padding adicional para evitar cortes de labels
        
        # Verificar que se guardó correctamente
        if ruta_archivo_salida.exists():
            tamaño_archivo = ruta_archivo_salida.stat().st_size
            print("[OK] Imagen guardada correctamente")
            print(f"   Tamaño del archivo: {tamaño_archivo:,} bytes ({tamaño_archivo/1024/1024:.2f} MB)")
        else:
            print("[ERROR] El archivo no se guardó correctamente")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error al guardar imagen: {str(e)}")
        return False
    
    print()
    
    # Paso 8: Mostrar el gráfico
    print("[INFO] Mostrando gráfico en pantalla...")
    try:
        plt.show()
        print("[OK] Gráfico mostrado correctamente")
    except Exception as e:
        print(f"[WARNING] Error al mostrar gráfico: {str(e)}")
    
    # Cerrar la figura para liberar memoria
    plt.close()
    
    # Paso 9: Resumen final
    print()
    print("=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Imagen generada: {ruta_archivo_salida}")
    print(f"[INFO] Columnas numéricas analizadas: {len(columnas_numericas)}")
    print(f"[INFO] Registros utilizados: {registros_limpios:,}")
    print(f"[INFO] Dimensiones de matriz de correlación: {matriz_correlacion.shape}")
    
    # === ANÁLISIS DE CORRELACIONES ===
    print("\n[ANÁLISIS] Correlaciones más importantes:")
    print("-" * 60)
    
    # Obtener todas las correlaciones (sin diagonal)
    correlaciones_flat = []
    for i in range(len(matriz_correlacion.columns)):
        for j in range(i+1, len(matriz_correlacion.columns)):
            var1 = matriz_correlacion.columns[i]
            var2 = matriz_correlacion.columns[j]
            corr_val = matriz_correlacion.iloc[i, j]
            correlaciones_flat.append((var1, var2, corr_val))
    
    # Ordenar por valor absoluto de correlación
    correlaciones_flat.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop 10 correlaciones más fuertes:")
    for i, (var1, var2, corr) in enumerate(correlaciones_flat[:10], 1):
        tipo_corr = "Fuerte positiva" if corr > 0.7 else "Moderada positiva" if corr > 0.4 else "Débil positiva" if corr > 0 else "Débil negativa" if corr > -0.4 else "Moderada negativa" if corr > -0.7 else "Fuerte negativa"
        print(f"  {i:2d}. {var1} - {var2}: {corr:+.3f} ({tipo_corr})")
    
    # Mostrar correlaciones más altas con score_final si existe
    if 'score_final' in matriz_correlacion.columns:
        print("\nCorrelaciones con score_final:")
        print("-" * 60)
        correlaciones_score = matriz_correlacion['score_final'].abs().sort_values(ascending=False)
        for i, (variable, corr_abs) in enumerate(correlaciones_score.head(6).items(), 1):
            if variable != 'score_final':  # Excluir auto-correlación
                corr_val = matriz_correlacion.loc[variable, 'score_final']
                tipo = "Positiva" if corr_val > 0 else "Negativa"
                print(f"  {i}. {variable}: {corr_val:+.3f} ({tipo})")
    
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
