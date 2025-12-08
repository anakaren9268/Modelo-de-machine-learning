#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para dividir dataset en conjuntos de entrenamiento, validación y prueba
usando hash estable basado en ID del conductor
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
import hashlib
import numpy as np
from pathlib import Path
import os
import sys

def hash_conductor_id(conductor_id, salt="ml_project_2024"):
    """
    Genera un hash estable para el ID del conductor
    
    Args:
        conductor_id: ID del conductor
        salt: Salt para hacer el hash más estable
    
    Returns:
        Valor hash entre 0 y 1
    """
    # Convertir a string y normalizar
    id_str = str(conductor_id).strip().lower()
    
    # Crear hash usando SHA256
    hash_obj = hashlib.sha256()
    hash_obj.update(f"{salt}_{id_str}".encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Convertir a número entre 0 y 1
    hash_int = int(hash_hex[:8], 16)  # Usar primeros 8 caracteres
    return hash_int / (16**8)  # Normalizar a [0, 1]

def asignar_conjunto(hash_value):
    """
    Asigna un conjunto basado en el valor hash
    
    Args:
        hash_value: Valor hash entre 0 y 1
    
    Returns:
        'entrenamiento', 'validacion', o 'prueba'
    """
    if hash_value < 0.7:
        return 'entrenamiento'
    elif hash_value < 0.85:  # 0.7 + 0.15
        return 'validacion'
    else:  # >= 0.85
        return 'prueba'

def main():
    """
    Función principal que ejecuta todo el proceso de división del dataset
    """
    print("=" * 70)
    print("PROGRAMA PARA DIVIDIR DATASET EN CONJUNTOS DE ML")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_clasificado.csv")
    ruta_entrenamiento = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_entrenamiento.csv")
    ruta_validacion = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_validacion.csv")
    ruta_prueba = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_prueba.csv")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo entrenamiento: {ruta_entrenamiento}")
    print(f"[INFO] Archivo validación: {ruta_validacion}")
    print(f"[INFO] Archivo prueba: {ruta_prueba}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_clasificado.csv no existe.")
        print(f"   Ruta buscada: {ruta_archivo_entrada}")
        print("   Asegúrate de que el archivo existe antes de ejecutar este programa.")
        return False
    
    print("[OK] Archivo de entrada encontrado")
    
    # Paso 2: Cargar el archivo CSV
    print("[INFO] Cargando dataset clasificado...")
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
    
    # Paso 3: Verificar que existe la columna DriverId
    print("[INFO] Verificando columna DriverId...")
    
    if 'DriverId' not in df.columns:
        print("[ERROR] La columna 'DriverId' no existe en el dataset.")
        print("   Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        return False
    
    print("[OK] Columna DriverId encontrada")
    
    # Mostrar estadísticas de conductores únicos
    conductores_unicos = df['DriverId'].nunique()
    registros_totales = len(df)
    print(f"   Conductores únicos: {conductores_unicos:,}")
    print(f"   Registros totales: {registros_totales:,}")
    print(f"   Promedio registros por conductor: {registros_totales/conductores_unicos:.1f}")
    
    print()
    
    # Paso 4: Aplicar hash estable a cada conductor
    print("[INFO] Aplicando hash estable a conductores...")
    
    try:
        # Generar hash para cada conductor único
        conductores_hash = {}
        conductores_unicos_lista = df['DriverId'].unique()
        
        print(f"   Procesando {len(conductores_unicos_lista):,} conductores únicos...")
        
        for conductor_id in conductores_unicos_lista:
            hash_value = hash_conductor_id(conductor_id)
            conjunto = asignar_conjunto(hash_value)
            conductores_hash[conductor_id] = conjunto
        
        print("[OK] Hash aplicado a todos los conductores")
        
        # Mostrar distribución de conductores por conjunto
        distribucion_conductores = {}
        for conjunto in ['entrenamiento', 'validacion', 'prueba']:
            count = sum(1 for c in conductores_hash.values() if c == conjunto)
            distribucion_conductores[conjunto] = count
        
        print("\nDistribución de conductores por conjunto:")
        for conjunto, count in distribucion_conductores.items():
            porcentaje = (count / conductores_unicos) * 100
            print(f"   {conjunto.capitalize()}: {count:,} conductores ({porcentaje:.1f}%)")
        
    except Exception as e:
        print(f"[ERROR] Error al aplicar hash: {str(e)}")
        return False
    
    print()
    
    # Paso 5: Asignar cada registro a su conjunto
    print("[INFO] Asignando registros a conjuntos...")
    
    try:
        # Crear columna de conjunto para cada registro
        df['conjunto'] = df['DriverId'].map(conductores_hash)
        
        print("[OK] Registros asignados a conjuntos")
        
        # Mostrar distribución de registros por conjunto
        distribucion_registros = df['conjunto'].value_counts()
        
        print("\nDistribución de registros por conjunto:")
        for conjunto in ['entrenamiento', 'validacion', 'prueba']:
            if conjunto in distribucion_registros:
                count = distribucion_registros[conjunto]
                porcentaje = (count / registros_totales) * 100
                print(f"   {conjunto.capitalize()}: {count:,} registros ({porcentaje:.1f}%)")
        
    except Exception as e:
        print(f"[ERROR] Error al asignar registros: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Verificar que no hay duplicación de conductores
    print("[INFO] Verificando que no hay duplicación de conductores...")
    
    try:
        # Verificar que cada conductor aparece solo en un conjunto
        verificacion = df.groupby('DriverId')['conjunto'].nunique()
        conductores_duplicados = verificacion[verificacion > 1]
        
        if len(conductores_duplicados) > 0:
            print(f"[ERROR] Se encontraron {len(conductores_duplicados)} conductores en múltiples conjuntos:")
            for conductor in conductores_duplicados.head(5).index:
                conjuntos = df[df['DriverId'] == conductor]['conjunto'].unique()
                print(f"   {conductor}: {conjuntos}")
            return False
        else:
            print("[OK] No hay duplicación de conductores entre conjuntos")
        
    except Exception as e:
        print(f"[ERROR] Error en verificación: {str(e)}")
        return False
    
    print()
    
    # Paso 7: Dividir y guardar datasets
    print("[INFO] Dividiendo y guardando datasets...")
    
    try:
        # Dividir dataset
        df_entrenamiento = df[df['conjunto'] == 'entrenamiento'].drop('conjunto', axis=1)
        df_validacion = df[df['conjunto'] == 'validacion'].drop('conjunto', axis=1)
        df_prueba = df[df['conjunto'] == 'prueba'].drop('conjunto', axis=1)
        
        # Guardar archivos
        archivos_guardados = []
        
        # Entrenamiento
        df_entrenamiento.to_csv(ruta_entrenamiento, index=False, encoding='utf-8')
        if ruta_entrenamiento.exists():
            tamaño = ruta_entrenamiento.stat().st_size
            archivos_guardados.append(('Entrenamiento', len(df_entrenamiento), tamaño))
            print(f"[OK] Dataset entrenamiento guardado: {len(df_entrenamiento):,} registros")
        
        # Validación
        df_validacion.to_csv(ruta_validacion, index=False, encoding='utf-8')
        if ruta_validacion.exists():
            tamaño = ruta_validacion.stat().st_size
            archivos_guardados.append(('Validación', len(df_validacion), tamaño))
            print(f"[OK] Dataset validación guardado: {len(df_validacion):,} registros")
        
        # Prueba
        df_prueba.to_csv(ruta_prueba, index=False, encoding='utf-8')
        if ruta_prueba.exists():
            tamaño = ruta_prueba.stat().st_size
            archivos_guardados.append(('Prueba', len(df_prueba), tamaño))
            print(f"[OK] Dataset prueba guardado: {len(df_prueba):,} registros")
        
    except Exception as e:
        print(f"[ERROR] Error al guardar archivos: {str(e)}")
        return False
    
    print()
    
    # Paso 8: Resumen final
    print("=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Conductores únicos procesados: {conductores_unicos:,}")
    print(f"[INFO] Registros totales procesados: {registros_totales:,}")
    
    print("\nRESUMEN DE DIVISIÓN:")
    print("-" * 50)
    
    total_registros = sum(count for _, count, _ in archivos_guardados)
    
    for nombre, count, tamaño in archivos_guardados:
        porcentaje = (count / total_registros) * 100
        tamaño_mb = tamaño / (1024 * 1024)
        print(f"{nombre:<12}: {count:>6,} registros ({porcentaje:>5.1f}%) - {tamaño_mb:>5.1f} MB")
    
    print(f"{'Total':<12}: {total_registros:>6,} registros (100.0%)")
    
    # Verificar porcentajes objetivo
    print("\nVERIFICACIÓN DE PORCENTAJES OBJETIVO:")
    print("-" * 50)
    objetivos = {'Entrenamiento': 70.0, 'Validación': 15.0, 'Prueba': 15.0}
    
    for nombre, count, _ in archivos_guardados:
        porcentaje_real = (count / total_registros) * 100
        objetivo = objetivos[nombre]
        diferencia = abs(porcentaje_real - objetivo)
        estado = "[OK]" if diferencia < 2.0 else "[WARNING]"
        print(f"{nombre:<12}: {porcentaje_real:>5.1f}% (objetivo: {objetivo:>4.1f}%) {estado}")
    
    print("\nArchivos generados:")
    for nombre, _, tamaño in archivos_guardados:
        tamaño_mb = tamaño / (1024 * 1024)
        print(f"  - dataset_{nombre.lower()}.csv ({tamaño_mb:.1f} MB)")
    
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
