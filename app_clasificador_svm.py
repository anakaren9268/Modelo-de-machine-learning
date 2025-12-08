#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación web interactiva para clasificar conductores con un modelo SVM ganador.

Esta app permite:
1. Ingresar manualmente valores normalizados mediante controles deslizantes.
2. Cargar archivos CSV con datos sin normalizar para evaluar el modelo.
3. Visualizar probabilidades de predicción con gráficas dinámicas.
4. Explorar fronteras de decisión usando PCA en dos dimensiones.
5. Simular datos en tiempo real para observar el comportamiento del modelo.
6. Consultar instrucciones de ejecución local y despliegue en Streamlit Cloud.

Requisitos (pip install):
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- seaborn
- joblib

Ejecutar localmente:
> streamlit run app_clasificador_svm.py
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# === CONFIGURACIÓN GENERAL DE LA APP ===
st.set_page_config(
    page_title="Clasificador SVM de Conductores",
    layout="wide",
    page_icon="🚗",
)

# Ajustar estilo global de gráficos estáticos
sns.set(style="whitegrid")


# === CONSTANTES DEL PROYECTO ===
RUTA_BASE = r"C:\Users\anaka\Downloads\Proyecto ML (2)\Proyecto ML\Porgramas Python"
RUTA_MODELO = os.path.join(RUTA_BASE, "svm_modelo_ganador.pkl")
# Dataset de referencia para visualizar fronteras y contexto histórico
RUTA_DATASET_REFERENCIA = os.path.join(RUTA_BASE, "dataset_entrenamiento.csv")

COLUMNAS_ORIGINALES = {
    "HarshAccelerationPer100Km": {"min": 0, "max": 12},
    "HarshBrakingPer100Km": {"min": 0, "max": 15},
    "HarshTurningPer100Km": {"min": 0, "max": 8},
    "IdlingRatePercentOfIgnitionTime": {"min": 10, "max": 30},
    "FuelUnder50PercentPer100Km": {"min": 0, "max": 0.5},
    "SpeedOver95Per100Km": {"min": 0, "max": 8},
    "RPMOver1600Per100Km": {"min": 0, "max": 8},
    "FueraDeRutaFRPer100Km": {"min": 0, "max": 5},
}

COLUMNAS_NORMALIZADAS = [
    "HarshBrakingPer100Km_Norm",
    "HarshAccelerationPer100Km_Norm",
    "HarshTurningPer100Km_Norm",
    "IdlingRatePercentOfIgnitionTime_Norm",
    "SpeedOver95Per100Km_Norm",
    "FuelUnder50PercentPer100Km_Norm",
    "RPMOver1600Per100Km_Norm",
    "FueraDeRutaFRPer100Km_Norm",
]

CLASES_ORDENADAS = ["Excelente", "Bueno", "Regular", "Malo"]


# === FUNCIONES AUXILIARES ===
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta_modelo: str):
    """Carga el modelo SVM previamente entrenado."""

    if not os.path.exists(ruta_modelo):
        st.error(
            f"No se encontró el archivo del modelo en: {ruta_modelo}. "
            "Verifica la ruta o entrena nuevamente el modelo."
        )
        st.stop()

    try:
        modelo = joblib.load(ruta_modelo)
    except Exception as exc:  # pragma: no cover - manejo en despliegue
        st.error(f"Ocurrió un error al cargar el modelo: {exc}")
        st.stop()

    # Si el archivo contiene un diccionario con el modelo, extraerlo
    if isinstance(modelo, dict):
        if "model" in modelo:
            return modelo["model"]
        st.error(
            "El archivo del modelo no tiene el formato esperado. Vuelve a entrenarlo o reemplázalo por uno válido."
        )
        st.stop()

    return modelo


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la normalización min-max con clipping a las columnas originales."""

    df_norm = df.copy()

    for columna, limites in COLUMNAS_ORIGINALES.items():
        if columna not in df_norm.columns:
            raise ValueError(
                f"El dataset no contiene la columna requerida: '{columna}'."
            )

        min_val = limites["min"]
        max_val = limites["max"]
        rango = max_val - min_val
        if rango == 0:
            df_norm[columna + "_Norm"] = 0.0
        else:
            df_norm[columna + "_Norm"] = ((df_norm[columna] - min_val) / rango).clip(0, 1)

    return df_norm


def preparar_features(df_normalizado: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y devuelve únicamente las columnas normalizadas requeridas."""

    faltantes = [col for col in COLUMNAS_NORMALIZADAS if col not in df_normalizado.columns]
    if faltantes:
        raise ValueError(
            "El dataset normalizado no contiene todas las columnas requeridas: "
            + ", ".join(faltantes)
        )

    return df_normalizado[COLUMNAS_NORMALIZADAS].copy()


def predecir_clase(modelo, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Realiza la predicción de clase y devuelve probabilidades."""

    # Detener el flujo si existen valores faltantes
    if features.isnull().values.any():
        raise ValueError(
            "Se detectaron valores vacíos en las características proporcionadas. "
            "Completa o elimina dichos registros antes de continuar."
        )

    # Alinear columnas con el orden utilizado durante el entrenamiento
    if hasattr(modelo, "feature_names_in_"):
        expected_columns = list(modelo.feature_names_in_)
        faltantes = [col for col in expected_columns if col not in features.columns]
        if faltantes:
            raise ValueError(
                "Faltan columnas en los datos proporcionados: " + ", ".join(faltantes)
            )
        features = features.reindex(columns=expected_columns)

    clases_predichas = modelo.predict(features)
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(features)
    else:
        # Fallback para modelos sin predict_proba (no es el caso del SVC actual)
        probabilidades = np.zeros((len(features), len(CLASES_ORDENADAS)))

    return clases_predichas, probabilidades


@st.cache_data(show_spinner=False)
def cargar_dataset_referencia(ruta: str) -> pd.DataFrame:
    """Carga un dataset de referencia para visualizar las fronteras de decisión."""

    if not os.path.exists(ruta):
        st.warning(
            "No se encontró el dataset de referencia para graficar las fronteras de decisión."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(ruta)
    except Exception as exc:  # pragma: no cover
        st.warning(f"No se pudo cargar el dataset de referencia: {exc}")
        return pd.DataFrame()

    if all(col in df.columns for col in COLUMNAS_NORMALIZADAS + ["Categoria_Conductor"]):
        return df

    try:
        df_norm = normalizar_dataframe(df)
    except Exception:
        return pd.DataFrame()

    if "Categoria_Conductor" in df.columns:
        df_norm["Categoria_Conductor"] = df["Categoria_Conductor"]
    else:
        df_norm["Categoria_Conductor"] = "Sin etiqueta"

    return df_norm


def graficar_probabilidades(probabilidades: np.ndarray, clases: List[str]) -> None:
    """Genera una gráfica de barras interactiva con Plotly."""

    df_plot = pd.DataFrame({"Clase": clases, "Probabilidad": probabilidades[0]})
    fig = px.bar(
        df_plot,
        x="Clase",
        y="Probabilidad",
        text="Probabilidad",
        color="Clase",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Distribución de Probabilidades por Clase",
    )
    fig.update_traces(texttemplate="%{y:.2%}", textposition="outside")
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        xaxis_title="Clase",
        yaxis_title="Probabilidad",
        margin=dict(l=40, r=40, t=60, b=40),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


def graficar_frontera_decision(modelo, df_referencia: pd.DataFrame) -> None:
    """Grafica fronteras de decisión usando PCA para reducir a dos dimensiones."""

    if df_referencia.empty:
        st.info(
            "No se pudo generar la visualización de fronteras de decisión por falta de datos de referencia."
        )
        return

    X = df_referencia[COLUMNAS_NORMALIZADAS].values
    y = df_referencia.get("Categoria_Conductor", "Sin etiqueta").values

    # Reducir a 2 componentes principales
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Entrenar un modelo auxiliar sobre espacio PCA para graficar fronteras
    modelo_aux = modelo.__class__(**modelo.get_params())
    modelo_aux.fit(X_pca, y)

    # Crear malla de puntos en el plano PCA
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    Z = modelo_aux.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    mapa_clases = {clase: idx for idx, clase in enumerate(CLASES_ORDENADAS)}
    Z_numeric = np.vectorize(lambda etiqueta: mapa_clases.get(etiqueta, np.nan))(Z).astype(float)
    niveles = np.arange(len(CLASES_ORDENADAS) + 1) - 0.5
    norma = BoundaryNorm(niveles, len(CLASES_ORDENADAS))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_fondo = ListedColormap(
        [
            "#d4f1f4",
            "#fef9c3",
            "#fde2e4",
            "#d1fae5",
        ]
    )
    cmap_puntos = ListedColormap(
        [
            "#0d47a1",
            "#f39c12",
            "#c0392b",
            "#16a085",
        ]
    )
    cmap_fondo.set_bad(color="#f5f5f5")
    cmap_puntos.set_bad(color="#9e9e9e")

    ax.contourf(xx, yy, Z_numeric, alpha=0.25, cmap=cmap_fondo, levels=niveles, norm=norma)
    valores_clase = np.array(
        [mapa_clases.get(valor, np.nan) for valor in y], dtype=float
    )
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=valores_clase,
        cmap=cmap_puntos,
        norm=norma,
        s=60,
        edgecolor="k",
        alpha=0.85,
    )
    ax.set_xlabel("Componente Principal 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("Componente Principal 2", fontsize=13, fontweight="bold")
    ax.set_title("Fronteras de decisión en espacio PCA", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.3)

    handles, _ = scatter.legend_elements()
    ax.legend(handles, CLASES_ORDENADAS, title="Clases", fontsize=11)
    st.pyplot(fig, clear_figure=True)


def generar_interpretacion(clase: str, features_dict: Dict[str, float]) -> str:
    """Genera un comentario automático que explique la clasificación obtenida."""

    comentarios = []

    aceleracion = features_dict.get("HarshAccelerationPer100Km_Norm", 0)
    frenado = features_dict.get("HarshBrakingPer100Km_Norm", 0)
    giros = features_dict.get("HarshTurningPer100Km_Norm", 0)
    ralenti = features_dict.get("IdlingRatePercentOfIgnitionTime_Norm", 0)
    combustible = features_dict.get("FuelUnder50PercentPer100Km_Norm", 0)
    velocidad = features_dict.get("SpeedOver95Per100Km_Norm", 0)
    rpm = features_dict.get("RPMOver1600Per100Km_Norm", 0)
    fuera_ruta = features_dict.get("FueraDeRutaFRPer100Km_Norm", 0)

    altos = []
    bajos = []

    for nombre, valor in [
        ("aceleraciones bruscas", aceleracion),
        ("frenadas bruscas", frenado),
        ("giros bruscos", giros),
        ("tiempo en ralentí", ralenti),
        ("incidencias de combustible", combustible),
        ("excesos de velocidad", velocidad),
        ("incidencias de RPM", rpm),
        ("salidas de ruta", fuera_ruta),
    ]:
        if valor >= 0.7:
            altos.append(nombre)
        elif valor <= 0.3:
            bajos.append(nombre)

    if altos:
        comentarios.append(
            "Se detectan valores elevados en: " + ", ".join(altos) + "."
        )
    if bajos:
        comentarios.append(
            "Los indicadores positivos incluyen niveles bajos en: "
            + ", ".join(bajos)
            + "."
        )

    resumen_clase = {
        "Excelente": "El perfil es sobresaliente, con comportamientos muy seguros y eficientes.",
        "Bueno": "El desempeño es favorable con algunos aspectos puntuales a mejorar.",
        "Regular": "Se observan comportamientos mixtos; conviene reforzar la conducción preventiva.",
        "Malo": "Se requiere intervención inmediata para corregir hábitos de conducción de riesgo.",
    }

    interpretacion_base = resumen_clase.get(
        clase, "La clasificación requiere análisis adicional."
    )

    if not comentarios:
        comentarios.append(
            "Los indicadores se mantienen en niveles intermedios, sin desviaciones extremas."
        )

    return interpretacion_base + " " + " ".join(comentarios)


def generar_datos_simulados() -> Dict[str, float]:
    """Genera un diccionario con valores normalizados aleatorios controlados."""

    datos = {}
    for columna in COLUMNAS_NORMALIZADAS:
        # Simular valores con distribución beta para dar más realismo (sesgo hacia 0.3-0.4)
        valor = np.random.beta(2, 2)
        datos[columna] = float(np.clip(valor, 0, 1))
    return datos


# === VISTAS DE LA APLICACIÓN ===
def vista_entrada_manual(modelo) -> None:
    """Sección para ingresar valores manualmente mediante sliders."""

    st.subheader("Ingreso manual de indicadores normalizados")
    st.markdown(
        "**Ajusta los controles deslizantes (0-100) para simular un nuevo conductor.**\n"
        "Los valores representan el porcentaje normalizado (0 equivale al mínimo histórico y 100 al máximo)."
    )

    columnas_display = [
        ("HarshAccelerationPer100Km_Norm", "Aceleraciones bruscas / 100 km"),
        ("HarshBrakingPer100Km_Norm", "Frenadas bruscas / 100 km"),
        ("HarshTurningPer100Km_Norm", "Giros bruscos / 100 km"),
        ("IdlingRatePercentOfIgnitionTime_Norm", "Tasa de ralentí (%)"),
        ("FuelUnder50PercentPer100Km_Norm", "Incidencias crítico de combustible / 100 km"),
        ("SpeedOver95Per100Km_Norm", "Excesos de velocidad / 100 km"),
        ("RPMOver1600Per100Km_Norm", "Incidencias límite de RPM / 100 km"),
        ("FueraDeRutaFRPer100Km_Norm", "Fuera de ruta / 100 km"),
    ]

    valores_slider = {}
    columnas = st.columns(2)
    for idx, (col_norm, etiqueta) in enumerate(columnas_display):
        with columnas[idx % 2]:
            valor = st.slider(
                etiqueta,
                min_value=0,
                max_value=100,
                value=30,
                step=1,
                help="Arrastra para ajustar el porcentaje normalizado (0 a 100).",
            )
            valores_slider[col_norm] = valor / 100

    st.markdown("---")
    if st.button("Clasificar conductor manual", use_container_width=True):
        features_df = pd.DataFrame([valores_slider])
        clases, proba = predecir_clase(modelo, features_df)
        clase = clases[0]

        st.success(f"La clasificación estimada es: **{clase}**")
        st.info(generar_interpretacion(clase, valores_slider))

        graficar_probabilidades(proba, modelo.classes_)


def vista_carga_csv(modelo) -> None:
    """Sección para cargar archivos CSV y aplicar la normalización automáticamente."""

    st.subheader("Clasificación mediante carga de archivo CSV")
    st.markdown(
        "Carga un archivo con las mismas columnas crudas que `dataset_unido`."
        "El sistema realizará la normalización automáticamente antes de predecir."
    )

    archivo = st.file_uploader(
        "Selecciona un archivo CSV", type=["csv"], help="Las columnas deben coincidir con el dataset original."
    )

    if archivo is not None:
        try:
            df_raw = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo. Verifica el formato. Detalles: {exc}")
            return

        patrones_vacios = ["", " ", "none", "None", "NONE", "null", "Null", "NULL", "nan", "NaN", "NAN"]
        df_raw = df_raw.replace(patrones_vacios, np.nan)

        st.write("Vista previa de los primeros registros:")
        st.dataframe(df_raw.head())

        try:
            df_norm = normalizar_dataframe(df_raw)
            features_df = preparar_features(df_norm)
        except Exception as exc:
            st.error(str(exc))
            return
        columnas_base = list(COLUMNAS_ORIGINALES.keys())

        def _es_vacio(valor) -> bool:
            if pd.isna(valor):
                return True
            if isinstance(valor, str):
                texto = valor.strip().lower()
                return texto in {"", "none", "nan", "null"}
            return False

        mask_inactivos_raw = df_raw[columnas_base].apply(
            lambda fila: all(_es_vacio(valor) for valor in fila), axis=1
        )
        mask_inactivos_features = features_df.isnull().all(axis=1)
        mask_incompletos_features = features_df.isnull().any(axis=1)
        mask_inactivos = mask_inactivos_raw | mask_inactivos_features | mask_incompletos_features

        if mask_inactivos.any():
            st.warning(
                f"Se detectaron {mask_inactivos.sum()} registros con datos vacíos o incompletos. "
                "Se marcarán como 'Inactivo' y no se clasificarán."
            )

        features_validos = features_df[~mask_inactivos].copy()

        if st.button("Clasificar archivo cargado", use_container_width=True):
            resultados = df_raw.copy()
            resultados["Categoria_Predicha"] = "Inactivo"

            if not features_validos.empty:
                clases, probabilidades = predecir_clase(modelo, features_validos)
                resultados.loc[~mask_inactivos, "Categoria_Predicha"] = clases

                st.success("Clasificación completada. Resultados (primeros 20 registros):")
                st.dataframe(resultados.head(20))

                st.download_button(
                    "Descargar resultados completos",
                    data=resultados.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_svm.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.markdown("### Probabilidades promedio por clase")
                promedios = probabilidades.mean(axis=0)
                graficar_probabilidades(np.array([promedios]), modelo.classes_)
            else:
                st.info(
                    "Todos los registros fueron marcados como 'Inactivo'. No se ejecutó la clasificación."
                )
                st.dataframe(resultados.head(20))
                st.download_button(
                    "Descargar registros marcados como inactivos",
                    data=resultados.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_svm_inactivos.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def vista_analisis_longitudinal(modelo) -> None:
    """Carga CSV longitudinal (un conductor, múltiples horas) y grafica clasificación vs tiempo."""

    st.subheader("Análisis longitudinal (un conductor en distintas horas)")
    st.markdown(
        "Carga un archivo CSV de un solo conductor con múltiples intervalos de tiempo. "
        "Se normaliza, se clasifica cada fila y se grafica la clase predicha a lo largo del tiempo "
        "usando la columna 'interval start' como eje X."
    )

    archivo = st.file_uploader(
        "Selecciona un archivo CSV", type=["csv"], help="Debe contener la columna 'interval start'."
    )

    if archivo is not None:
        try:
            df_raw = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo. Verifica el formato. Detalles: {exc}")
            return

        patrones_vacios = ["", " ", "none", "None", "NONE", "null", "Null", "NULL", "nan", "NaN", "NAN"]
        df_raw = df_raw.replace(patrones_vacios, np.nan)

        st.write("Vista previa de los primeros registros:")
        st.dataframe(df_raw.head())

        # Detectar columna de tiempo 'interval start' con variantes comunes
        columnas_norm = {c.strip().lower().replace("_", " "): c for c in df_raw.columns}
        posibles_nombres = ["interval start", "interval-start", "intervalstart"]
        col_interval = None
        for nombre in posibles_nombres:
            if nombre in columnas_norm:
                col_interval = columnas_norm[nombre]
                break
        if col_interval is None:
            # Intento adicional por coincidencia parcial
            for clave_norm, original in columnas_norm.items():
                if "interval" in clave_norm and "start" in clave_norm:
                    col_interval = original
                    break
        if col_interval is None:
            st.error(
                "No se encontró la columna de tiempo requerida 'interval start' en el archivo. "
                "Asegúrate de incluirla (por ejemplo: 'interval start', 'Interval Start', 'interval_start')."
            )
            return

        try:
            df_norm = normalizar_dataframe(df_raw)
            features_df = preparar_features(df_norm)
        except Exception as exc:
            st.error(str(exc))
            return
        columnas_base = list(COLUMNAS_ORIGINALES.keys())

        def _es_vacio(valor) -> bool:
            if pd.isna(valor):
                return True
            if isinstance(valor, str):
                texto = valor.strip().lower()
                return texto in {"", "none", "nan", "null"}
            return False

        mask_inactivos_raw = df_raw[columnas_base].apply(
            lambda fila: all(_es_vacio(valor) for valor in fila), axis=1
        )
        mask_inactivos_features = features_df.isnull().all(axis=1)
        mask_incompletos_features = features_df.isnull().any(axis=1)
        mask_inactivos = mask_inactivos_raw | mask_inactivos_features | mask_incompletos_features

        if mask_inactivos.any():
            st.warning(
                f"Se detectaron {mask_inactivos.sum()} registros con datos vacíos o incompletos. "
                "Se marcarán como 'Inactivo' y no se clasificarán."
            )

        features_validos = features_df[~mask_inactivos].copy()

        if st.button("Clasificar archivo longitudinal", use_container_width=True):
            resultados = df_raw.copy()
            resultados["Categoria_Predicha"] = "Inactivo"

            if not features_validos.empty:
                clases, _probabilidades = predecir_clase(modelo, features_validos)
                resultados.loc[~mask_inactivos, "Categoria_Predicha"] = clases

            # Preparar eje temporal
            resultados["_Interval"] = pd.to_datetime(resultados[col_interval], errors="coerce")
            resultados_plot = resultados[["_Interval", "Categoria_Predicha"]].dropna(subset=["_Interval"]).copy()
            resultados_plot = resultados_plot.sort_values("_Interval")

            if resultados_plot.empty:
                st.info("No hay registros con tiempos válidos para graficar.")
            else:
                orden_categorias = ["Inactivo"] + CLASES_ORDENADAS
                fig = px.line(
                    resultados_plot,
                    x="_Interval",
                    y="Categoria_Predicha",
                    markers=True,
                    title="Clasificación a lo largo del tiempo (análisis longitudinal)",
                )
                # Eje Y categórico con orden definido y una sola línea, pantalla completa y fuentes grandes
                fig.update_traces(line=dict(width=3), marker=dict(size=9))
                fig.update_layout(
                    xaxis_title="<b>Interval start</b>",
                    yaxis_title="<b>Clasificación</b>",
                    margin=dict(l=10, r=10, t=60, b=40),
                    height=780,
                    autosize=True,
                    font=dict(size=18),
                    title=dict(font=dict(size=24)),
                    xaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=22, color="black")),
                    yaxis=dict(
                        type="category",
                        categoryorder="array",
                        categoryarray=orden_categorias,
                        title=dict(font=dict(size=24)),
                        tickfont=dict(size=22, color="black"),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=16)),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.success("Proceso finalizado. Puedes descargar los resultados completos.")
            st.dataframe(resultados.head(20))
            st.download_button(
                "Descargar resultados completos",
                data=resultados.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_longitudinal.csv",
                mime="text/csv",
                use_container_width=True,
            )

def vista_comparativa_conductores(modelo) -> None:
    """Carga CSV combinado (varios conductores) y grafica clasificación por conductor vs tiempo."""

    st.subheader("Comparativa de conductores (archivo combinado)")
    st.markdown(
        "Carga el archivo generado por el script de unión (por ejemplo, 'conductores_unidos.csv'). "
        "Se normaliza, se clasifica cada fila y se grafica la clase predicha a lo largo del tiempo "
        "por cada conductor, cada uno con una línea de distinto color."
    )

    archivo = st.file_uploader(
        "Selecciona el archivo combinado CSV", type=["csv"], help="Debe contener la columna 'interval start' y un identificador de conductor."
    )

    if archivo is not None:
        try:
            df_raw = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo. Verifica el formato. Detalles: {exc}")
            return

        patrones_vacios = ["", " ", "none", "None", "NONE", "null", "Null", "NULL", "nan", "NaN", "NAN"]
        df_raw = df_raw.replace(patrones_vacios, np.nan)

        st.write("Vista previa de los primeros registros:")
        st.dataframe(df_raw.head())

        # Detección de columna temporal
        columnas_norm = {c.strip().lower().replace("_", " "): c for c in df_raw.columns}
        posibles_nombres_tiempo = ["interval start", "interval-start", "intervalstart"]
        col_interval = None
        for nombre in posibles_nombres_tiempo:
            if nombre in columnas_norm:
                col_interval = columnas_norm[nombre]
                break
        if col_interval is None:
            for clave_norm, original in columnas_norm.items():
                if "interval" in clave_norm and "start" in clave_norm:
                    col_interval = original
                    break
        if col_interval is None:
            st.error("No se encontró la columna de tiempo 'interval start'.")
            return

        # Detección de columna de conductor: priorizar firstName + LastName
        posibles_first = {"firstname", "first name", "first_name"}
        posibles_last = {"lastname", "last name", "last_name"}
        col_first = next((columnas_norm[n] for n in posibles_first if n in columnas_norm), None)
        col_last = next((columnas_norm[n] for n in posibles_last if n in columnas_norm), None)

        col_conductor = None
        if col_first or col_last:
            nombre = (df_raw[col_first].astype(str) if col_first else pd.Series([""] * len(df_raw)))
            apellido = (df_raw[col_last].astype(str) if col_last else pd.Series([""] * len(df_raw)))
            df_raw["_ConductorNombre"] = (nombre.fillna("").str.strip() + " " + apellido.fillna("").str.strip()).str.strip()
            df_raw["_ConductorNombre"].replace({"": "Desconocido"}, inplace=True)
            col_conductor = "_ConductorNombre"
        else:
            # Fallback: columnas comunes de identificación
            posibles_conductor = ["source file", "source_file", "conductor", "driver", "id_conductor", "id driver"]
            for nombre in posibles_conductor:
                if nombre in columnas_norm:
                    col_conductor = columnas_norm[nombre]
                    break
            if col_conductor is None:
                for clave_norm, original in columnas_norm.items():
                    if "conductor" in clave_norm or "driver" in clave_norm:
                        col_conductor = original
                        break
            if col_conductor is None:
                col_conductor = "_Conductor"
                df_raw[col_conductor] = "Desconocido"
            else:
                # Si el identificador es un nombre de archivo, derivar un nombre legible
                nombre_normalizado = next((k for k, v in columnas_norm.items() if v == col_conductor), "").strip()
                if nombre_normalizado in {"source file", "sourcefile", "archivo fuente"} or col_conductor.lower() in {"source_file"}:
                    df_raw["_ConductorNombre"] = df_raw[col_conductor].astype(str).apply(
                        lambda s: os.path.splitext(os.path.basename(s))[0]
                    )
                    col_conductor = "_ConductorNombre"

        try:
            df_norm = normalizar_dataframe(df_raw)
            features_df = preparar_features(df_norm)
        except Exception as exc:
            st.error(str(exc))
            return
        columnas_base = list(COLUMNAS_ORIGINALES.keys())

        def _es_vacio(valor) -> bool:
            if pd.isna(valor):
                return True
            if isinstance(valor, str):
                texto = valor.strip().lower()
                return texto in {"", "none", "nan", "null"}
            return False

        mask_inactivos_raw = df_raw[columnas_base].apply(
            lambda fila: all(_es_vacio(valor) for valor in fila), axis=1
        )
        mask_inactivos_features = features_df.isnull().all(axis=1)
        mask_incompletos_features = features_df.isnull().any(axis=1)
        mask_inactivos = mask_inactivos_raw | mask_inactivos_features | mask_incompletos_features

        if mask_inactivos.any():
            st.warning(
                f"Se detectaron {mask_inactivos.sum()} registros con datos vacíos o incompletos. "
                "Se marcarán como 'Inactivo' y no se clasificarán."
            )

        features_validos = features_df[~mask_inactivos].copy()

        if st.button("Clasificar y graficar comparativa", use_container_width=True):
            resultados = df_raw.copy()
            resultados["Categoria_Predicha"] = "Inactivo"

            if not features_validos.empty:
                clases, _ = predecir_clase(modelo, features_validos)
                resultados.loc[~mask_inactivos, "Categoria_Predicha"] = clases

            resultados["_Interval"] = pd.to_datetime(resultados[col_interval], errors="coerce")
            resultados_plot = resultados[["_Interval", "Categoria_Predicha", col_conductor]].dropna(subset=["_Interval"]).copy()
            resultados_plot = resultados_plot.sort_values(["_Interval", col_conductor])

            # Título fuera de la figura, justo debajo del botón
            st.subheader("Comparativa de conductores a lo largo del tiempo")

            if resultados_plot.empty:
                st.info("No hay registros con tiempos válidos para graficar.")
            else:
                orden_categorias = ["Inactivo"] + CLASES_ORDENADAS
                fig = px.line(
                    resultados_plot,
                    x="_Interval",
                    y="Categoria_Predicha",
                    color=col_conductor,
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                fig.update_layout(
                    xaxis_title="<b>Interval start</b>",
                    yaxis_title="<b>Clasificación</b>",
                    margin=dict(l=10, r=10, t=80, b=40),
                    height=780,
                    autosize=True,
                    font=dict(size=18),
                    xaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=22, color="black")),
                    yaxis=dict(
                        type="category",
                        categoryorder="array",
                        categoryarray=orden_categorias,
                        title=dict(font=dict(size=24)),
                        tickfont=dict(size=22, color="black"),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=16)),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.success("Proceso finalizado. Puedes descargar los resultados completos.")
            st.dataframe(resultados.head(20))
            st.download_button(
                "Descargar resultados completos",
                data=resultados.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_comparativa.csv",
                mime="text/csv",
                use_container_width=True,
            )
def vista_analisis_visual(modelo, df_referencia: pd.DataFrame) -> None:
    """Visualizaciones complementarias: fronteras y sliders de exploración."""

    st.subheader("Exploración visual del modelo")

    with st.expander("Fronteras de decisión (PCA)", expanded=True):
        graficar_frontera_decision(modelo, df_referencia)

    st.markdown("---")
    st.subheader("Explorador de predicciones en dos dimensiones")

    col1, col2 = st.columns(2)
    with col1:
        slider_x = st.slider(
            "Componente PCA 1 (escala simulada)",
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            value=0.0,
        )
    with col2:
        slider_y = st.slider(
            "Componente PCA 2 (escala simulada)",
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            value=0.0,
        )

    if not df_referencia.empty:
        pca = PCA(n_components=2, random_state=42)
        X = df_referencia[COLUMNAS_NORMALIZADAS].values
        pca.fit(X)

        # Reconstruir un punto aproximado en el espacio original
        punto_pca = np.array([[slider_x, slider_y]])
        punto_original = pca.inverse_transform(punto_pca)

        punto_df = pd.DataFrame(punto_original, columns=COLUMNAS_NORMALIZADAS)
        clases, proba = predecir_clase(modelo, punto_df)
        st.write(
            f"Predicción aproximada para la ubicación seleccionada en PCA: **{clases[0]}**"
        )
        graficar_probabilidades(proba, modelo.classes_)
    else:
        st.info(
            "Se requiere un dataset de referencia válido para habilitar esta sección."
        )


def vista_simulacion_tiempo_real(modelo) -> None:
    """Modo opcional de simulación con generación de datos cada pocos segundos."""

    st.subheader("Simulación en tiempo real")
    st.markdown(
        "Activa el modo de simulación para observar cómo se comporta el modelo con datos aleatorios."
    )

    if "simulacion_activa" not in st.session_state:
        st.session_state.simulacion_activa = False

    col_inicio, col_detener = st.columns(2)
    with col_inicio:
        if st.button("Iniciar simulación", type="primary"):
            st.session_state.simulacion_activa = True
    with col_detener:
        if st.button("Detener simulación", type="secondary"):
            st.session_state.simulacion_activa = False

    placeholder = st.empty()
    iteraciones = st.slider(
        "Número de observaciones simuladas",
        min_value=1,
        max_value=50,
        value=10,
        help="Cantidad de muestras a generar cuando la simulación esté activa.",
    )
    intervalo = st.slider(
        "Intervalo entre observaciones (segundos)",
        min_value=1,
        max_value=10,
        value=2,
    )

    if st.session_state.simulacion_activa:
        for i in range(iteraciones):
            datos = generar_datos_simulados()
            df_datos = pd.DataFrame([datos])
            clases, proba = predecir_clase(modelo, df_datos)
            clase = clases[0]

            with placeholder.container():
                st.info(
                    f"Iteración {i + 1}/{iteraciones} | Predicción: **{clase}**"
                )
                graficar_probabilidades(proba, modelo.classes_)
                st.caption(generar_interpretacion(clase, datos))

            time.sleep(intervalo)

        st.session_state.simulacion_activa = False
        st.success("Simulación completada.")
    else:
        placeholder.info("La simulación está detenida. Usa los botones para iniciarla o detenerla.")


def vista_instrucciones_finales() -> None:
    """Sección con instrucciones para ejecutar y desplegar la aplicación."""

    st.markdown("---")
    st.header("Instrucciones de ejecución y despliegue")

    st.subheader("Ejecución local")
    st.markdown(
        """1. Asegúrate de tener Python 3.10+ instalado.\n
2. Instala dependencias:
```
pip install streamlit joblib pandas scikit-learn plotly matplotlib seaborn
```
3. Ejecuta la aplicación:
```
streamlit run app_clasificador_svm.py
```
"""
    )

    st.subheader("Publicación en Streamlit Cloud")
    st.markdown(
        "1. Sube el proyecto (incluyendo `app_clasificador_svm.py` y `svm_modelo_ganador.pkl`) a un repositorio de GitHub.\n"
        "2. Entra a [https://share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub.\n"
        "3. Selecciona el repositorio, rama y archivo principal (`app_clasificador_svm.py`).\n"
        "4. Define las variables de entorno necesarias (si aplica) y despliega.\n"
        "5. Verifica que los archivos de datos o rutas externas sean accesibles en la nube (usa `st.secrets` o almacenamiento público si es necesario)."
    )


# === FLUJO PRINCIPAL ===
def main() -> None:
    st.title("🚗 Clasificador SVM de Conductores")
    st.write(
        "Explora el desempeño del modelo SVM ganador para clasificar conductores en las categorías "
        "**Excelente**, **Bueno**, **Regular** y **Malo**."
    )

    modelo = cargar_modelo(RUTA_MODELO)
    df_referencia = cargar_dataset_referencia(RUTA_DATASET_REFERENCIA)

    tabs = st.tabs(
        [
            "Carga CSV",
            "Análisis longitudinal",
            "Comparativa de conductores",
        ]
    )

    with tabs[0]:
        vista_carga_csv(modelo)
    with tabs[1]:
        vista_analisis_longitudinal(modelo)
    with tabs[2]:
        vista_comparativa_conductores(modelo)


if __name__ == "__main__":
    main()


