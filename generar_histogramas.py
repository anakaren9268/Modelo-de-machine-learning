#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para generar histogramas de columnas numéricas
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

def main():
    """
    Función principal que ejecuta todo el proceso de generación de histogramas
    """
    print("=" * 70)
    print("PROGRAMA PARA GENERAR HISTOGRAMAS DE COLUMNAS NUMÉRICAS")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_clasificado.csv")
    ruta_carpeta_histogramas = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\histogramas")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Carpeta de histogramas: {ruta_carpeta_histogramas}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_clasificado.csv no existe.")
        print(f"   Ruta buscada: {ruta_archivo_entrada}")
        print("   Asegúrate de que el archivo existe antes de ejecutar este programa.")
        return False
    
    print("[OK] Archivo de entrada encontrado")
    
    # Paso 2: Cargar el archivo CSV
    print("[INFO] Cargando dataset...")
    try:
        df = pd.read_csv(ruta_archivo_entrada)
        print("[OK] Dataset cargado exitosamente")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        print(f"   Dimensiones: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Error al cargar el archivo: {str(e)}")
        return False
    
    print()
    
    # Paso 3: Identificar columnas numéricas
    print("[INFO] Identificando columnas numéricas...")
    
    # Obtener columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        print("[ERROR] No se encontraron columnas numéricas en el dataset.")
        return False
    
    print(f"[OK] Se encontraron {len(columnas_numericas)} columnas numéricas:")
    for i, col in enumerate(columnas_numericas, 1):
        print(f"   {i:2d}. {col}")
    
    print()
    
    # Paso 4: Crear carpeta de histogramas
    print("[INFO] Creando carpeta de histogramas...")
    
    try:
        ruta_carpeta_histogramas.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Carpeta creada/verificada: {ruta_carpeta_histogramas}")
    except Exception as e:
        print(f"[ERROR] Error al crear carpeta: {str(e)}")
        return False
    
    print()
    
    # Paso 5: Generar histogramas
    print("[INFO] Generando histogramas...")
    
    histogramas_generados = 0
    histogramas_fallidos = 0
    
    # Configurar estilo de matplotlib
    plt.style.use('default')
    
    for i, columna in enumerate(columnas_numericas, 1):
        try:
            print(f"   [{i:2d}/{len(columnas_numericas)}] Procesando: {columna}")
            
            # Obtener datos de la columna (excluir valores NaN)
            datos = df[columna].dropna()
            
            if len(datos) == 0:
                print(f"       [WARNING] No hay datos válidos en {columna}")
                histogramas_fallidos += 1
                continue
            
            # === ANÁLISIS ESTADÍSTICO ===
            media = datos.mean()
            mediana = datos.median()
            desv_std = datos.std()
            min_val = datos.min()
            max_val = datos.max()
            q25 = datos.quantile(0.25)
            q75 = datos.quantile(0.75)
            iqr = q75 - q25
            
            print(f"       [ESTADÍSTICAS] Media: {media:.4f}, Mediana: {mediana:.4f}, Desv. Est.: {desv_std:.4f}")
            print(f"                     Rango: [{min_val:.4f}, {max_val:.4f}], IQR: {iqr:.4f}")
            
            # Crear figura y eje con tamaño más grande
            # Ajustar tamaño según longitud del nombre de columna
            longitud_nombre = len(columna)
            ancho_figura = max(12, min(16, longitud_nombre * 0.3))
            plt.figure(figsize=(ancho_figura, 8))
            
            # Configurar parámetros de fuente globales
            plt.rcParams.update({'font.size': 12})
            
            # Generar histograma
            plt.hist(datos, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
            
            # Las estadísticas ya fueron calculadas arriba
            
            # Configurar título con tamaño más grande y mejor padding
            plt.title(f'Distribución de {columna}', 
                     fontsize=16, fontweight='bold', pad=25)
            
            # Configurar etiquetas con tamaño más grande
            plt.xlabel('Valores', fontsize=14, fontweight='bold')
            plt.ylabel('Frecuencia', fontsize=14, fontweight='bold')
            
            # Agregar líneas de referencia
            plt.axvline(media, color='red', linestyle='--', linewidth=2.5, 
                       label=f'Media: {media:.2f}')
            plt.axvline(mediana, color='green', linestyle='--', linewidth=2.5, 
                       label=f'Mediana: {mediana:.2f}')
            
            # Agregar estadísticas en una caja de texto
            stats_text = f'Desv. Est.: {desv_std:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
            plt.text(0.98, 0.98, stats_text, 
                    transform=plt.gca().transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Configurar leyenda con tamaño más grande
            plt.legend(fontsize=12, framealpha=0.9, loc='best')
            
            # Configurar grid
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Configurar ticks con tamaño adecuado
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            
            # Ajustar layout con padding adicional para evitar cortes
            plt.tight_layout(rect=[0, 0.02, 1, 0.98])
            
            # Guardar histograma como PNG con mejor calidad y ajuste (ANTES de mostrar)
            nombre_archivo = f"{columna}.png"
            ruta_archivo = ruta_carpeta_histogramas / nombre_archivo
            
            plt.savefig(ruta_archivo, 
                       dpi=300, 
                       bbox_inches='tight', 
                       facecolor='white',
                       edgecolor='none',
                       pad_inches=0.3)  # Padding adicional para evitar cortes
            
            # Mostrar histograma
            plt.show()
            
            plt.close()  # Cerrar figura para liberar memoria
            
            # Verificar que se guardó correctamente
            if ruta_archivo.exists():
                tamaño_archivo = ruta_archivo.stat().st_size
                print(f"       [OK] Histograma guardado: {nombre_archivo} ({tamaño_archivo:,} bytes)")
                histogramas_generados += 1
            else:
                print(f"       [ERROR] No se pudo guardar {nombre_archivo}")
                histogramas_fallidos += 1
            
        except Exception as e:
            print(f"       [ERROR] Error al procesar {columna}: {str(e)}")
            histogramas_fallidos += 1
            plt.close()  # Cerrar figura en caso de error
            continue
    
    print()
    
    # Paso 6: Resumen final
    print("=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Carpeta de histogramas: {ruta_carpeta_histogramas}")
    print(f"[INFO] Columnas numéricas encontradas: {len(columnas_numericas)}")
    print(f"[INFO] Histogramas generados exitosamente: {histogramas_generados}")
    print(f"[INFO] Histogramas fallidos: {histogramas_fallidos}")
    
    if histogramas_generados > 0:
        print(f"[INFO] Archivos PNG guardados en: {ruta_carpeta_histogramas}")
        
        # Mostrar lista de archivos generados
        archivos_png = list(ruta_carpeta_histogramas.glob("*.png"))
        if archivos_png:
            print("\nArchivos generados:")
            for archivo in sorted(archivos_png):
                tamaño = archivo.stat().st_size
                print(f"  - {archivo.name} ({tamaño:,} bytes)")
    
    print("=" * 70)
    
    return histogramas_generados > 0

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
