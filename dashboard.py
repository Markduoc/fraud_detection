import os
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ─── DESACOPLAMIENTO DE RUTAS ───
MODELS_DIR = os.getenv("FRAUD_MODELS_DIR", "models")
PROCESSED_DIR = os.getenv("FRAUD_PROCESSED_DIR", "data/processed")
REPORTS_DIR = os.getenv("FRAUD_REPORTS_DIR", "data/reports")

st.info(f"🔧 Entorno cargado desde variables de entorno (o defaults)")

# ── TÍTULO ───────────────────────────────────────────────────────────
st.title("🔍 Dashboard — Detección de Fraude en Tarjetas de Crédito")
st.markdown("Pipeline ITY1101 · Gestión de Datos para IA · DuocUC")
st.divider()

# ── CARGAR REPORTE DEL MODELO ────────────────────────────────────────
reporte_path = f"{REPORTS_DIR}/reporte_modelo.csv"

if not os.path.exists(reporte_path):
    st.error("⚠️ No se encontró reporte_modelo.csv — ejecuta modelado.py primero.")
    st.stop()

reporte = pd.read_csv(reporte_path).iloc[-1]  # última ejecución

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — KPIs PRINCIPALES
# ══════════════════════════════════════════════════════════════════════
st.subheader("📊 Indicadores Clave del Modelo (KPIs)")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("ROC-AUC",     f"{reporte['roc_auc']:.4f}",  help="Capacidad general de discriminación del modelo")
col2.metric("PR-AUC",      f"{reporte['pr_auc']:.4f}",   help="Precisión-Recall bajo desbalance de clases")
col3.metric("Registros Train", f"{int(reporte['n_train']):,}")
col4.metric("Registros Test",  f"{int(reporte['n_test']):,}")
col5.metric("Fraudes en Test", f"{int(reporte['fraudes_test']):,}")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — RENDIMIENTO DE RECURSOS
# ══════════════════════════════════════════════════════════════════════
st.subheader("⚙️ Rendimiento de Recursos (Nube/Local)")

col1, col2, col3, col4 = st.columns(4)

if "tiempo_entrenamiento_s" in reporte:
    col1.metric("Tiempo Entrenamiento", f"{reporte['tiempo_entrenamiento_s']} s")
    col2.metric("Tiempo Predicción",    f"{reporte['tiempo_prediccion_s']} s")
    col3.metric("RAM Consumida",        f"{reporte['ram_consumida_mb']} MB")
    col4.metric("CPUs Disponibles",     f"{int(reporte['n_cpus_disponibles'])}")

    # Gráfico de RAM antes vs después
    fig, ax = plt.subplots(figsize=(5, 3))
    ram_vals = [reporte["ram_antes_mb"], reporte["ram_despues_mb"]]
    bars = ax.bar(["Antes del\nentrenamiento", "Después del\nentrenamiento"],
                  ram_vals, color=["#4C72B0", "#DD8452"], width=0.5)
    ax.bar_label(bars, fmt="%.0f MB", padding=3)
    ax.set_ylabel("RAM (MB)")
    ax.set_title("Uso de RAM durante el entrenamiento")
    ax.set_ylim(0, max(ram_vals) * 1.2)
    st.pyplot(fig)
    plt.close()
else:
    st.info("Métricas de recursos no disponibles — regenera el modelo con modelado.py actualizado.")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — GRÁFICOS DEL MODELO (requiere datos procesados + modelo)
# ══════════════════════════════════════════════════════════════════════
st.subheader("📈 Análisis del Modelo")

# Cargar modelo
modelos_pkl = sorted(glob.glob(f"{MODELS_DIR}/fraud_model_*.pkl"), reverse=True)
datos_path  = f"{PROCESSED_DIR}/fraud_clean.csv"
if not modelos_pkl:
    st.warning("No se encontró modelo .pkl — ejecuta modelado.py.")
elif not os.path.exists(datos_path):
    st.warning("No se encontró fraud_clean.csv — ejecuta limpieza.py.")
else:
    with open(modelos_pkl[0], "rb") as f:
        pipeline = pickle.load(f)

    modelo        = pipeline["modelo"]
    preprocesador = pipeline["preprocesador"]

    # Reproducir el mismo split del entrenamiento
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(datos_path)
    X  = df.drop(columns=["is_fraud"])
    y  = df["is_fraud"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_proc = preprocesador.transform(X_test)
    y_prob = modelo.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    col1, col2 = st.columns(2)

    # ── Matriz de Confusión ──────────────────────────────────────────
    with col1:
        st.markdown("**Matriz de Confusión**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legítima", "Fraude"],
                    yticklabels=["Legítima", "Fraude"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)
        plt.close()

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        | | Valor |
        |---|---|
        | Verdaderos Positivos (fraudes detectados) | **{tp:,}** |
        | Falsos Negativos (fraudes no detectados)  | **{fn:,}** |
        | Falsos Positivos (falsas alarmas)         | **{fp:,}** |
        | Verdaderos Negativos                      | **{tn:,}** |
        """)

    # ── Curva ROC ───────────────────────────────────────────────────
    with col2:
        st.markdown("**Curva ROC**")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
        ax.set_xlabel("Tasa de Falsos Positivos")
        ax.set_ylabel("Tasa de Verdaderos Positivos")
        ax.set_title("Curva ROC")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # ── Curva Precisión-Recall ───────────────────────────────────────
    with col3:
        st.markdown("**Curva Precisión-Recall**")
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = average_precision_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(recall, precision, color="#DD8452", lw=2, label=f"PR-AUC = {pr_auc_val:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisión")
        ax.set_title("Curva Precisión-Recall")
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close()

    # ── Importancia de Variables ─────────────────────────────────────
    with col4:
        st.markdown("**Top 10 Variables más Importantes**")
        try:
            # Nombres de columnas tras el preprocesador
            num_names = preprocesador.transformers_[0][2]
            cat_encoder = preprocesador.transformers_[1][1]
            cat_names = cat_encoder.get_feature_names_out(
                preprocesador.transformers_[1][2]
            ).tolist()
            all_names = list(num_names) + cat_names

            importancias = pd.Series(modelo.feature_importances_, index=all_names)
            top10 = importancias.nlargest(10).sort_values()

            fig, ax = plt.subplots(figsize=(4, 3))
            top10.plot(kind="barh", ax=ax, color="#4C72B0")
            ax.set_title("Importancia de Variables (Top 10)")
            ax.set_xlabel("Importancia")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"No se pudo calcular importancia de variables: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — DISTRIBUCIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════
if os.path.exists(datos_path):
    st.subheader("🗂️ Análisis de Datos Procesados")
    df = pd.read_csv(datos_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución de clases (Balance del dataset)**")
        conteo = df["is_fraud"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Legítima (0)", "Fraude (1)"], conteo.values,
               color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Cantidad de transacciones")
        ax.set_title("             Balance de Clases")
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 100, f"{v:,}\n({v/len(df)*100:.2f}%)",
                    ha="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Distribución del monto de transacciones**")
        fig, ax = plt.subplots(figsize=(4, 3))
        df[df["is_fraud"] == 0]["amt"].clip(upper=500).hist(
            bins=50, ax=ax, alpha=0.6, color="#4C72B0", label="Legítima")
        df[df["is_fraud"] == 1]["amt"].clip(upper=500).hist(
            bins=50, ax=ax, alpha=0.6, color="#DD8452", label="Fraude")
        ax.set_xlabel("Monto ($)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Montos (clip $500)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

st.divider()
st.caption(f"ITY1101 · Gestión de Datos para IA · DuocUC 2025")