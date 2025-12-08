#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para normalizar datos usando Min-Max (escala 0-1)
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
from pathlib import Path
import os
import sys

def main():
    """
    Función principal que ejecuta todo el proceso de normalización Min-Max
    """
    print("=" * 70)
    print("PROGRAMA PARA NORMALIZAR DATOS CON MIN-MAX (ESCALA 0-1 CON CLIPPING)")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_unido.csv")
    ruta_archivo_salida = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_normalizado.csv")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo de salida: {ruta_archivo_salida}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_unido.csv no existe.")
        print(f"   Ruta buscada: {ruta_archivo_entrada}")
        print("   Asegúrate de que el archivo existe antes de ejecutar este programa.")
        return False
    
    print("[OK] Archivo de entrada encontrado")
    
    # Paso 2: Cargar el archivo CSV
    print("[INFO] Cargando archivo CSV...")
    try:
        df = pd.read_csv(ruta_archivo_entrada)
        print(f"[OK] Archivo cargado exitosamente")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        print(f"   Dimensiones: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Error al cargar el archivo: {str(e)}")
        return False
    
    print()
    
    # Paso 3: Definir parámetros de normalización
    print("[INFO] Configurando parámetros de normalización...")
    
    # Diccionario con los límites para cada parámetro (nombres reales de columnas)
    parametros_normalizacion = {
        'HarshBrakingPer100Km': {'min': 0, 'max': 15},
        'HarshAccelerationPer100Km': {'min': 0, 'max': 12},
        'HarshTurningPer100Km': {'min': 0, 'max': 8},
        'IdlingRatePercentOfIgnitionTime': {'min': 10, 'max': 30},
        'SpeedOver95Per100Km': {'min': 0, 'max': 8},
        'FuelUnder50PercentPer100Km': {'min': 0, 'max': 0.5},
        'RPMOver1600Per100Km': {'min': 0, 'max': 8},
        'FueraDeRutaFRPer100Km': {'min': 0, 'max': 5}
    }
    
    print(f"   Parámetros configurados: {len(parametros_normalizacion)}")
    for param, limites in parametros_normalizacion.items():
        print(f"   - {param}: min={limites['min']}, max={limites['max']}")
    
    print()
    
    # Paso 4: Verificar que las columnas existen en el dataset
    print("[INFO] Verificando columnas en el dataset...")
    columnas_faltantes = []
    columnas_encontradas = []
    
    for columna in parametros_normalizacion.keys():
        if columna in df.columns:
            columnas_encontradas.append(columna)
            print(f"   [OK] {columna} encontrada")
        else:
            columnas_faltantes.append(columna)
            print(f"   [WARNING] {columna} NO encontrada")
    
    if columnas_faltantes:
        print(f"\n[WARNING] {len(columnas_faltantes)} columnas no se encontraron en el dataset:")
        for col in columnas_faltantes:
            print(f"   - {col}")
        print("   Solo se normalizarán las columnas encontradas.")
    
    print(f"\n[INFO] Columnas a normalizar: {len(columnas_encontradas)}")
    print()
    
    # Paso 5: Aplicar normalización Min-Max manual
    print("[INFO] Aplicando normalización Min-Max...")
    columnas_normalizadas = []
    
    for columna in columnas_encontradas:
        try:
            # Obtener límites
            min_valor = parametros_normalizacion[columna]['min']
            max_valor = parametros_normalizacion[columna]['max']
            
            # Aplicar fórmula: valor_normalizado = (valor_actual - minimo) / (maximo - minimo)
            columna_normalizada = f"{columna}_Norm"
            df[columna_normalizada] = (df[columna] - min_valor) / (max_valor - min_valor)
            
            # Limitar valores normalizados al rango [0, 1] usando clipping
            valores_antes_clipping = df[columna_normalizada].copy()
            df[columna_normalizada] = df[columna_normalizada].clip(lower=0, upper=1)
            
            # Contar valores que fueron ajustados por el clipping
            valores_ajustados = (valores_antes_clipping != df[columna_normalizada]).sum()
            
            print(f"   [OK] {columna} -> {columna_normalizada}")
            print(f"       Valores originales: min={df[columna].min():.3f}, max={df[columna].max():.3f}")
            print(f"       Valores normalizados: min={df[columna_normalizada].min():.3f}, max={df[columna_normalizada].max():.3f}")
            if valores_ajustados > 0:
                print(f"       [INFO] {valores_ajustados} valores ajustados al rango [0,1] por clipping")
            else:
                print(f"       [OK] Todos los valores ya estaban en rango [0,1]")
            
            columnas_normalizadas.append(columna_normalizada)
            
        except Exception as e:
            print(f"   [ERROR] Error al normalizar {columna}: {str(e)}")
            continue
    
    print()
    print(f"[OK] Normalización completada: {len(columnas_normalizadas)} columnas procesadas")
    
    # Paso 6: Mostrar resumen y vista previa
    print()
    print("RESUMEN DE NORMALIZACIÓN:")
    print("-" * 50)
    print(f"[INFO] Columnas originales: {len(df.columns) - len(columnas_normalizadas)}")
    print(f"[INFO] Columnas normalizadas: {len(columnas_normalizadas)}")
    print(f"[INFO] Total de columnas: {len(df.columns)}")
    print(f"[INFO] Registros: {len(df):,}")
    print()
    
    print("VISTA PREVIA DE COLUMNAS NORMALIZADAS:")
    print("-" * 50)
    if columnas_normalizadas:
        # Mostrar las primeras 5 filas de las columnas normalizadas
        print(df[columnas_normalizadas].head())
    else:
        print("No se generaron columnas normalizadas.")
    
    print()
    
    # Paso 7: Guardar archivo normalizado
    print("[INFO] Guardando archivo normalizado...")
    print(f"   Ubicación: {ruta_archivo_salida}")
    
    try:
        df.to_csv(ruta_archivo_salida, index=False, encoding='utf-8')
        
        # Verificar que el archivo se guardó correctamente
        if ruta_archivo_salida.exists():
            tamaño_archivo = ruta_archivo_salida.stat().st_size
            print(f"[OK] Archivo guardado exitosamente")
            print(f"   Tamaño del archivo: {tamaño_archivo:,} bytes ({tamaño_archivo/1024/1024:.2f} MB)")
        else:
            print("[ERROR] El archivo no se guardó correctamente")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error al guardar archivo: {str(e)}")
        return False
    
    # Paso 8: Resumen final
    print()
    print("=" * 70)
    print("PROCESO DE NORMALIZACIÓN COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo generado: {ruta_archivo_salida}")
    print(f"[INFO] Columnas normalizadas: {len(columnas_normalizadas)}")
    print(f"[INFO] Registros procesados: {len(df):,}")
    print(f"[INFO] Total de columnas en archivo final: {len(df.columns)}")
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
