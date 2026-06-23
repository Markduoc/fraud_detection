import os
import time
import pickle
import logging
import sys
import numpy as np
import pandas as pd
import psutil
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ─── DESACOPLAMIENTO DE RUTAS ───
PROCESSED_DIR = os.getenv("FRAUD_PROCESSED_DIR", "data/processed")
MODELS_DIR = os.getenv("FRAUD_MODELS_DIR", "models")
REPORTS_DIR = os.getenv("FRAUD_REPORTS_DIR", "data/reports")

log.info(f"Configuración cargada desde entorno:")
log.info(f"  PROCESSED_DIR: {PROCESSED_DIR}")
log.info(f"  MODELS_DIR: {MODELS_DIR}")
log.info(f"  REPORTS_DIR: {REPORTS_DIR}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    """Función principal con manejo de excepciones controlado."""
    try:
        # ── 1. CARGAR DATOS ─────────────────────────────────────────────────
        ruta = f"{PROCESSED_DIR}/fraud_clean.csv"
        if not os.path.exists(ruta):
            log.error(f"No se encontró fraud_clean.csv en {PROCESSED_DIR}")
            raise FileNotFoundError(ruta)

        df = pd.read_csv(ruta)
        log.info(f"Datos cargados: {len(df):,} filas, {df.shape[1]} columnas")

        # ── 2. SEPARAR FEATURES Y TARGET ────────────────────────────────────
        TARGET = "is_fraud"
        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        log.info(f"Fraudes en dataset: {y.sum():,} ({y.mean()*100:.2f}%)")

        # ── 3. SPLIT TRAIN / TEST ESTRATIFICADO ─────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        log.info(f"Train: {len(X_train):,} filas | Test: {len(X_test):,} filas")

        # Iniciar monitoreo de recursos desde el preprocesamiento
        proceso = psutil.Process(os.getpid())

        # ── 4. PREPROCESAMIENTO ─────────────────────────────────────────────
        numericas = [
            "amt", "zip", "lat", "long", "city_pop",
            "merch_lat", "merch_long", "tx_hour", "tx_dow",
            "tx_is_weekend", "tx_is_night", "tx_month",
            "holder_age", "distance_km", "log_amt"
        ]
        categoricas = ["merchant", "category", "gender", "city", "state", "job"]

        preprocesador = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categoricas)
        ])

        log.info("Preprocesando features...")
        X_train_proc = preprocesador.fit_transform(X_train).toarray()
        X_test_proc = preprocesador.transform(X_test).toarray()
        log.info(f"RAM tras preprocesamiento: {proceso.memory_info().rss / 1024 / 1024:.1f} MB")

        # ── 5. SMOTE — BALANCEAR EL DATASET DE ENTRENAMIENTO ───────────────
        log.info(f"Distribución antes de SMOTE: "
                 f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

        smote = SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=5)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

        log.info(f"Distribución después de SMOTE: "
                 f"{dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

        # ── 6. ENTRENAR RANDOM FOREST ───────────────────────────────────────
        log.info("Entrenando Random Forest — esto puede tomar unos minutos...")

        ram_antes_mb = proceso.memory_info().rss / 1024 / 1024
        cpu_antes = psutil.cpu_percent(interval=1)
        tiempo_inicio = time.time()

        n_jobs_optimo = max(1, psutil.cpu_count(logical=False) // 2)
        log.info(f"CPUs físicos disponibles: {psutil.cpu_count(logical=False)} — usando {n_jobs_optimo}")

        modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs_optimo
        )
        modelo.fit(X_train_bal, y_train_bal)

        tiempo_fin = time.time()
        ram_despues_mb = proceso.memory_info().rss / 1024 / 1024
        cpu_despues = psutil.cpu_percent(interval=1)
        tiempo_entreno_s = round(tiempo_fin - tiempo_inicio, 2)
        ram_usada_mb = round(ram_despues_mb - ram_antes_mb, 2)

        log.info("Entrenamiento completado")
        log.info(f"  Tiempo de entrenamiento : {tiempo_entreno_s} segundos")
        log.info(f"  RAM antes               : {ram_antes_mb:.1f} MB")
        log.info(f"  RAM después             : {ram_despues_mb:.1f} MB")
        log.info(f"  RAM consumida           : {ram_usada_mb} MB")
        log.info(f"  CPU al terminar         : {cpu_despues}%")

        # ── 7. EVALUACIÓN ───────────────────────────────────────────────────
        log.info("Evaluando sobre el conjunto de test...")

        t_pred_inicio = time.time()
        y_prob = modelo.predict_proba(X_test_proc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        tiempo_prediccion_s = round(time.time() - t_pred_inicio, 4)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        log.info(f"\n{classification_report(y_test, y_pred, target_names=['Legítima','Fraude'])}")
        log.info(f"ROC-AUC : {roc_auc:.4f}")
        log.info(f"PR-AUC  : {pr_auc:.4f}")

        # ── 8. GUARDAR MODELO ───────────────────────────────────────────────
        fecha = datetime.now().strftime("%Y-%m-%d")
        ruta_modelo = f"{MODELS_DIR}/fraud_model_{fecha}.pkl"

        pipeline_completo = {
            "modelo": modelo,
            "preprocesador": preprocesador
        }
        with open(ruta_modelo, "wb") as f:
            pickle.dump(pipeline_completo, f)
        log.info(f"Modelo + preprocesador guardados en: {ruta_modelo}")

        # ── 9. REPORTE DE MÉTRICAS — MODO APPEND (HISTÓRICO) ────────────────
        reporte_path = f"{REPORTS_DIR}/reporte_modelo.csv"
        nuevo_reporte = pd.DataFrame([{
            "fecha": fecha,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "fraudes_test": int(y_test.sum()),
            "fraudes_detectados": int((y_pred == 1).sum()),
            "tiempo_entrenamiento_s": tiempo_entreno_s,
            "tiempo_prediccion_s": tiempo_prediccion_s,
            "ram_antes_mb": round(ram_antes_mb, 1),
            "ram_despues_mb": round(ram_despues_mb, 1),
            "ram_consumida_mb": ram_usada_mb,
            "cpu_pct_al_terminar": cpu_despues,
            "n_cpus_disponibles": psutil.cpu_count(),
        }])

        # Si el archivo ya existe, agregar sin duplicar encabezado
        if os.path.exists(reporte_path):
            reporte_existente = pd.read_csv(reporte_path)
            reporte_final = pd.concat([reporte_existente, nuevo_reporte], ignore_index=True)
        else:
            reporte_final = nuevo_reporte

        reporte_final.to_csv(reporte_path, index=False)
        log.info(f"✓ Reporte guardado en {reporte_path} (modo append — histórico acumulado)")
        log.info(f"  Total de ejecuciones registradas: {len(reporte_final)}")

        return 0

    except FileNotFoundError as e:
        log.error(f"Archivo no encontrado: {str(e)}")
        return 1
    except pd.errors.ParserError:
        log.error("Error al parsear CSV: el archivo puede estar corrupto")
        return 1
    except MemoryError:
        log.error("Memoria insuficiente para entrenar el modelo")
        return 1
    except Exception as e:
        log.error(f"Error durante el modelado: {type(e).__name__}")
        log.debug(f"Detalles técnicos: {str(e)}", exc_info=False)
        return 1

if __name__ == "__main__":
    sys.exit(main())