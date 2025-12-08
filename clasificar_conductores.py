#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para clasificar conductores por categorías según score
Autor: Asistente IA
Fecha: 2024
"""

import pandas as pd
from pathlib import Path
import os
import sys

def main():
    """
    Función principal que ejecuta todo el proceso de clasificación de conductores
    """
    print("=" * 70)
    print("PROGRAMA PARA CLASIFICAR CONDUCTORES POR CATEGORÍAS")
    print("=" * 70)
    
    # Definir rutas
    ruta_archivo_entrada = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_evaluado.csv")
    ruta_archivo_salida = Path(r"C:\Users\FRCemco04\Documents\Proyecto ML\Porgramas Python\dataset_clasificado.csv")
    
    print(f"[INFO] Archivo de entrada: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo de salida: {ruta_archivo_salida}")
    print()
    
    # Paso 1: Validar que existe el archivo de entrada
    if not ruta_archivo_entrada.exists():
        print("[ERROR] El archivo dataset_evaluado.csv no existe.")
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
    
    # Paso 3: Verificar que existe la columna score_final
    print("[INFO] Verificando columna score_final...")
    
    if 'score_final' not in df.columns:
        print("[ERROR] La columna 'score_final' no existe en el dataset.")
        print("   Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        return False
    
    print("[OK] Columna score_final encontrada")
    
    # Mostrar estadísticas básicas del score
    print(f"   Score mínimo: {df['score_final'].min():.6f}")
    print(f"   Score máximo: {df['score_final'].max():.6f}")
    print(f"   Score promedio: {df['score_final'].mean():.6f}")
    
    print()
    
    # Paso 4: Definir rangos de clasificación
    print("[INFO] Configurando rangos de clasificación...")
    
    rangos_clasificacion = {
        'Excelente': {'min': 0, 'max': 0.126000, 'descripcion': 'Score_Final <= 0.126000'},
        'Bueno': {'min': 0.126000, 'max': 0.140000, 'descripcion': 'Score_Final > 0.126000 y <= 0.140000'},
        'Regular': {'min': 0.140000, 'max': 0.189560, 'descripcion': 'Score_Final > 0.140000 y <= 0.189560'},
        'Malo': {'min': 0.189560, 'max': float('inf'), 'descripcion': 'Score_Final > 0.189560'}
    }
    
    print("   Rangos configurados:")
    for categoria, rango in rangos_clasificacion.items():
        print(f"   - {categoria}: {rango['descripcion']}")
    
    print()
    
    # Paso 5: Clasificar conductores
    print("[INFO] Clasificando conductores...")
    
    try:
        # Crear función para clasificar
        def clasificar_conductor(score):
            if score <= 0.126000:
                return 'Excelente'
            elif score <= 0.140000:
                return 'Bueno'
            elif score <= 0.189560:
                return 'Regular'
            else:
                return 'Malo'
        
        # Aplicar clasificación
        df['Categoria_Conductor'] = df['score_final'].apply(clasificar_conductor)
        
        print("[OK] Clasificación completada exitosamente")
        
    except Exception as e:
        print(f"[ERROR] Error al clasificar conductores: {str(e)}")
        return False
    
    print()
    
    # Paso 6: Contar conductores por categoría
    print("[INFO] Contando conductores por categoría...")
    
    conteo_categorias = df['Categoria_Conductor'].value_counts().sort_index()
    total_conductores = len(df)
    
    print("[OK] Conteo completado")
    print()
    
    # Paso 7: Mostrar resumen de clasificación
    print("RESUMEN DE CLASIFICACIÓN:")
    print("=" * 70)
    
    # Ordenar categorías según el orden deseado
    orden_categorias = ['Excelente', 'Bueno', 'Regular', 'Malo']
    
    print("\nDistribución de conductores por categoría:")
    print("-" * 50)
    
    for categoria in orden_categorias:
        if categoria in conteo_categorias:
            cantidad = conteo_categorias[categoria]
            porcentaje = (cantidad / total_conductores) * 100
            print(f"{categoria:<10}: {cantidad:>6,} conductores ({porcentaje:>5.1f}%)")
        else:
            print(f"{categoria:<10}: {0:>6,} conductores ({0:>5.1f}%)")
    
    print(f"\nTotal: {total_conductores:,} conductores")
    
    # Mostrar estadísticas adicionales
    print("\nEstadísticas adicionales:")
    print("-" * 50)
    print(f"Score mínimo: {df['score_final'].min():.6f}")
    print(f"Score máximo: {df['score_final'].max():.6f}")
    print(f"Score promedio: {df['score_final'].mean():.6f}")
    print(f"Score mediana: {df['score_final'].median():.6f}")
    
    print()
    
    # Paso 8: Guardar archivo clasificado
    print("[INFO] Guardando archivo clasificado...")
    print(f"   Ubicación: {ruta_archivo_salida}")
    
    try:
        df.to_csv(ruta_archivo_salida, index=False, encoding='utf-8')
        
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
    
    # Paso 9: Resumen final
    print()
    print("=" * 70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"[INFO] Archivo procesado: {ruta_archivo_entrada}")
    print(f"[INFO] Archivo generado: {ruta_archivo_salida}")
    print(f"[INFO] Total de conductores clasificados: {total_conductores:,}")
    print(f"[INFO] Columnas en archivo final: {len(df.columns)}")
    print(f"[INFO] Nueva columna agregada: Categoria_Conductor")
    
    # Mostrar distribución final
    print("\nDistribución final:")
    for categoria in orden_categorias:
        if categoria in conteo_categorias:
            cantidad = conteo_categorias[categoria]
            porcentaje = (cantidad / total_conductores) * 100
            print(f"  {categoria}: {cantidad:,} ({porcentaje:.1f}%)")
    
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
