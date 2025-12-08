#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para calcular score final de conductores
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
from pathlib import Path
import os
import sys

def main():
    """
    Función principal que ejecuta todo el proceso de evaluación de conductores
    """
    print("=" * 70)
    print("PROGRAMA PARA CALCULAR SCORE FINAL DE CONDUCTORES")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_normalizado.csv")
    ruta_archivo_salida = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_evaluado.csv")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo de salida: {ruta_archivo_salida}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_normalizado.csv no existe.")
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
    
    # Paso 3: Definir columnas requeridas y sus pesos
    print("[INFO] Configurando pesos para cálculo de score...")
    
    columnas_requeridas = {
        'HarshAccelerationPer100Km_Norm': 0.20,
        'HarshBrakingPer100Km_Norm': 0.18,
        'HarshTurningPer100Km_Norm': 0.18,
        'IdlingRatePercentOfIgnitionTime_Norm': 0.12,
        'FuelUnder50PercentPer100Km_Norm': 0.10,
        'SpeedOver95Per100Km_Norm': 0.08,
        'RPMOver1600Per100Km_Norm': 0.06,
        'FueraDeRutaFRPer100Km_Norm': 0.08
    }
    
    print(f"   Columnas configuradas: {len(columnas_requeridas)}")
    for columna, peso in columnas_requeridas.items():
        print(f"   - {columna}: peso={peso}")
    
    print()
    
    # Paso 4: Verificar que todas las columnas existen
    print("[INFO] Verificando columnas en el dataset...")
    columnas_faltantes = []
    columnas_encontradas = []
    
    for columna in columnas_requeridas.keys():
        if columna in df.columns:
            columnas_encontradas.append(columna)
            print(f"   [OK] {columna} encontrada")
        else:
            columnas_faltantes.append(columna)
            print(f"   [ERROR] {columna} NO encontrada")
    
    if columnas_faltantes:
        print(f"\n[ERROR] {len(columnas_faltantes)} columnas faltantes:")
        for col in columnas_faltantes:
            print(f"   - {col}")
        print("   No se puede continuar sin todas las columnas requeridas.")
        return False
    
    print(f"\n[OK] Todas las columnas requeridas están presentes")
    print()
    
    # Paso 5: Calcular score final
    print("[INFO] Calculando score...")
    
    try:
        # Aplicar la fórmula del score final
        df['score_final'] = (
            df['HarshAccelerationPer100Km_Norm'] * 0.20 +
            df['HarshBrakingPer100Km_Norm'] * 0.18 +
            df['HarshTurningPer100Km_Norm'] * 0.18 +
            df['IdlingRatePercentOfIgnitionTime_Norm'] * 0.12 +
            df['FuelUnder50PercentPer100Km_Norm'] * 0.10 +
            df['SpeedOver95Per100Km_Norm'] * 0.08 +
            df['RPMOver1600Per100Km_Norm'] * 0.06 +
            df['FueraDeRutaFRPer100Km_Norm'] * 0.08
        )
        
        print("[OK] Score calculado exitosamente")
        print(f"   Score mínimo: {df['score_final'].min():.6f}")
        print(f"   Score máximo: {df['score_final'].max():.6f}")
        print(f"   Score promedio: {df['score_final'].mean():.6f}")
        
    except Exception as e:
        print(f"[ERROR] Error al calcular score: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Ordenar por score_final (menor score = mejor conductor)
    print("[INFO] Ordenando conductores por score (menor = mejor)...")
    
    try:
        df_ordenado = df.sort_values('score_final', ascending=True).reset_index(drop=True)
        print("[OK] Dataset ordenado exitosamente")
    except Exception as e:
        print(f"[ERROR] Error al ordenar dataset: {str(e)}")
        return False
    
    print()
    
    # Paso 7: Análisis de percentiles
    print("ANÁLISIS DE PERCENTILES:")
    print("=" * 70)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nDistribución de scores por percentiles:")
    print("-" * 50)
    for p in percentiles:
        valor_percentil = df_ordenado['score_final'].quantile(p/100)
        print(f"Percentil {p:2d}%: {valor_percentil:.6f}")
    
    # Análisis de categorías de conductores
    print("\nCATEGORIZACIÓN DE CONDUCTORES:")
    print("-" * 50)
    
    # Definir categorías basadas en percentiles
    excelente = df_ordenado['score_final'].quantile(0.10)  # Top 10%
    bueno = df_ordenado['score_final'].quantile(0.25)      # Top 25%
    promedio = df_ordenado['score_final'].quantile(0.50)    # Mediana
    regular = df_ordenado['score_final'].quantile(0.75)     # Top 75%
    
    # Contar conductores por categoría
    conductores_excelentes = len(df_ordenado[df_ordenado['score_final'] <= excelente])
    conductores_buenos = len(df_ordenado[(df_ordenado['score_final'] > excelente) & (df_ordenado['score_final'] <= bueno)])
    conductores_promedio = len(df_ordenado[(df_ordenado['score_final'] > bueno) & (df_ordenado['score_final'] <= promedio)])
    conductores_regulares = len(df_ordenado[(df_ordenado['score_final'] > promedio) & (df_ordenado['score_final'] <= regular)])
    conductores_malos = len(df_ordenado[df_ordenado['score_final'] > regular])
    
    print(f"Excelentes (≤{excelente:.6f}): {conductores_excelentes:,} conductores ({conductores_excelentes/len(df_ordenado)*100:.1f}%)")
    print(f"Buenos ({excelente:.6f} - {bueno:.6f}): {conductores_buenos:,} conductores ({conductores_buenos/len(df_ordenado)*100:.1f}%)")
    print(f"Promedio ({bueno:.6f} - {promedio:.6f}): {conductores_promedio:,} conductores ({conductores_promedio/len(df_ordenado)*100:.1f}%)")
    print(f"Regulares ({promedio:.6f} - {regular:.6f}): {conductores_regulares:,} conductores ({conductores_regulares/len(df_ordenado)*100:.1f}%)")
    print(f"Malos (>{regular:.6f}): {conductores_malos:,} conductores ({conductores_malos/len(df_ordenado)*100:.1f}%)")
    
    print()
    
    # Paso 8: Mostrar mejores y peores conductores
    print("RANKING DE CONDUCTORES:")
    print("=" * 70)
    
    print("\nTOP 5 MEJORES CONDUCTORES (menor score):")
    print("-" * 50)
    mejores = df_ordenado.head(5)
    for i, (idx, row) in enumerate(mejores.iterrows(), 1):
        nombre_completo = f"{row['FirstName']} {row['LastName']}"
        print(f"{i:2d}. {nombre_completo:<30} | Score: {row['score_final']:.6f}")
    
    print("\nTOP 5 PEORES CONDUCTORES (mayor score):")
    print("-" * 50)
    peores = df_ordenado.tail(5)
    for i, (idx, row) in enumerate(peores.iterrows(), 1):
        nombre_completo = f"{row['FirstName']} {row['LastName']}"
        print(f"{i:2d}. {nombre_completo:<30} | Score: {row['score_final']:.6f}")
    
    print()
    
    # Paso 9: Guardar archivo evaluado
    print("[INFO] Guardando archivo final...")
    print(f"   Ubicación: {ruta_archivo_salida}")
    
    try:
        df_ordenado.to_csv(ruta_archivo_salida, index=False, encoding='utf-8')
        
        # Verificar que el archivo se guardó correctamente
        if ruta_archivo_salida.exists():
            tamaño_archivo = ruta_archivo_salida.stat().st_size
            print("[OK] Archivo guardado exitosamente")
            print(f"   Tamaño del archivo: {tamaño_archivo:,} bytes ({tamaño_archivo/1024/1024:.2f} MB)")
        else:
            print("[ERROR] El archivo no se guardó correctamente")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error al guardar archivo: {str(e)}")
        return False
    
    # Paso 10: Resumen final
    print()
    print("=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo generado: {ruta_archivo_salida}")
    print(f"[INFO] Total de conductores evaluados: {len(df_ordenado):,}")
    print(f"[INFO] Columnas en archivo final: {len(df_ordenado.columns)}")
    print(f"[INFO] Mejor conductor: {df_ordenado.iloc[0]['FirstName']} {df_ordenado.iloc[0]['LastName']} (Score: {df_ordenado.iloc[0]['score_final']:.6f})")
    print(f"[INFO] Peor conductor: {df_ordenado.iloc[-1]['FirstName']} {df_ordenado.iloc[-1]['LastName']} (Score: {df_ordenado.iloc[-1]['score_final']:.6f})")
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
