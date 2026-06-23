import os
import time
import pickle
import logging
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
os.makedirs("models", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

# ── 1. CARGAR DATOS ─────────────────────────────────────────────────
ruta = "data/processed/fraud_clean.csv"
if not os.path.exists(ruta):
    log.error("No se encontró fraud_clean.csv — ejecuta limpieza.py primero")
    raise FileNotFoundError(ruta)

df = pd.read_csv(ruta)
log.info(f"Datos cargados: {len(df):,} filas, {df.shape[1]} columnas")

# ── 2. SEPARAR FEATURES Y TARGET ────────────────────────────────────
TARGET = "is_fraud"
X = df.drop(columns=[TARGET])
y = df[TARGET]

log.info(f"Fraudes en dataset: {y.sum():,} ({y.mean()*100:.2f}%)")

# ── 3. SPLIT TRAIN / TEST ESTRATIFICADO ─────────────────────────────
# Estratificado = mantiene el mismo % de fraude en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% entrena, 20% evalúa
    stratify=y,          # mantiene proporción de fraudes
    random_state=42
)
log.info(f"Train: {len(X_train):,} filas | Test: {len(X_test):,} filas")

# Iniciar monitoreo de recursos desde el preprocesamiento
proceso = psutil.Process(os.getpid())

# ── 4. PREPROCESAMIENTO ─────────────────────────────────────────────
# Columnas numéricas → StandardScaler (lleva todo a la misma escala)
# Columnas categóricas → OneHotEncoder (convierte texto en números)
numericas = [
    "amt", "zip", "lat", "long", "city_pop",
    "merch_lat", "merch_long", "tx_hour", "tx_dow",
    "tx_is_weekend", "tx_is_night", "tx_month",
    "holder_age", "distance_km", "log_amt"
]
categoricas = ["merchant", "category", "gender", "city", "state", "job"]

# sparse_output=True → guarda solo los valores no-cero (ahorra RAM significativamente)
# Se convierte a denso (.toarray()) solo justo antes de SMOTE y del modelo
preprocesador = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categoricas)
])

log.info("Preprocesando features...")
X_train_proc = preprocesador.fit_transform(X_train).toarray()
X_test_proc  = preprocesador.transform(X_test).toarray()  # solo transform, nunca fit
log.info(f"RAM tras preprocesamiento: {proceso.memory_info().rss / 1024 / 1024:.1f} MB")

# ── 5. SMOTE — BALANCEAR EL DATASET DE ENTRENAMIENTO ───────────────
# IMPORTANTE: SMOTE solo se aplica al train, NUNCA al test
# Si lo aplicaras al test estarías evaluando con datos falsos
log.info(f"Distribución antes de SMOTE: "
         f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

smote = SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

log.info(f"Distribución después de SMOTE: "
         f"{dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

# ── 6. ENTRENAR RANDOM FOREST ───────────────────────────────────────
log.info("Entrenando Random Forest — esto puede tomar unos minutos...")

# Capturar recursos ANTES del entrenamiento
ram_antes_mb   = proceso.memory_info().rss / 1024 / 1024
cpu_antes      = psutil.cpu_percent(interval=1)
tiempo_inicio  = time.time()

# Usar solo la mitad de los núcleos disponibles para no saturar el sistema
n_jobs_optimo = max(1, psutil.cpu_count(logical=False) // 2)
log.info(f"CPUs físicos disponibles: {psutil.cpu_count(logical=False)} — usando {n_jobs_optimo}")

modelo = RandomForestClassifier(
    n_estimators=200,           # 200 árboles votando
    max_depth=15,               # profundidad máxima de cada árbol
    min_samples_leaf=4,         # mínimo de registros en cada hoja
    class_weight="balanced",    # doble protección junto a SMOTE
    random_state=42,
    n_jobs=n_jobs_optimo        # solo mitad de núcleos — optimización CPU
)
modelo.fit(X_train_bal, y_train_bal)

# Capturar recursos DESPUÉS del entrenamiento
tiempo_fin       = time.time()
ram_despues_mb   = proceso.memory_info().rss / 1024 / 1024
cpu_despues      = psutil.cpu_percent(interval=1)
tiempo_entreno_s = round(tiempo_fin - tiempo_inicio, 2)
ram_usada_mb     = round(ram_despues_mb - ram_antes_mb, 2)

log.info("Entrenamiento completado")
log.info(f"  Tiempo de entrenamiento : {tiempo_entreno_s} segundos")
log.info(f"  RAM antes               : {ram_antes_mb:.1f} MB")
log.info(f"  RAM después             : {ram_despues_mb:.1f} MB")
log.info(f"  RAM consumida           : {ram_usada_mb} MB")
log.info(f"  CPU al terminar         : {cpu_despues}%")

# ── 7. EVALUACIÓN ───────────────────────────────────────────────────
log.info("Evaluando sobre el conjunto de test...")

# Medir tiempo de predicción (cuánto tarda en evaluar el set de test)
t_pred_inicio = time.time()
y_prob = modelo.predict_proba(X_test_proc)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
tiempo_prediccion_s = round(time.time() - t_pred_inicio, 4)

# Métricas principales
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

log.info(f"\n{classification_report(y_test, y_pred, target_names=['Legítima','Fraude'])}")
log.info(f"ROC-AUC : {roc_auc:.4f}")
log.info(f"PR-AUC  : {pr_auc:.4f}")

# ── 8. GUARDAR MODELO Y REPORTE ─────────────────────────────────────
fecha = datetime.now().strftime("%Y-%m-%d")
ruta_modelo = f"models/fraud_model_{fecha}.pkl"

pipeline_completo = {
    "modelo"        : modelo,
    "preprocesador" : preprocesador
}
with open(ruta_modelo, "wb") as f:
    pickle.dump(pipeline_completo, f)
log.info(f"Modelo + preprocesador guardados en: {ruta_modelo}")

# Reporte de métricas en CSV
reporte = pd.DataFrame([{
    # Métricas del modelo
    "fecha"                  : fecha,
    "roc_auc"                : round(roc_auc, 4),
    "pr_auc"                 : round(pr_auc, 4),
    "n_train"                : len(X_train),
    "n_test"                 : len(X_test),
    "fraudes_test"           : int(y_test.sum()),
    "fraudes_detectados"     : int((y_pred == 1).sum()),
    # Métricas de rendimiento (recursos)
    "tiempo_entrenamiento_s" : tiempo_entreno_s,
    "tiempo_prediccion_s"    : tiempo_prediccion_s,
    "ram_antes_mb"           : round(ram_antes_mb, 1),
    "ram_despues_mb"         : round(ram_despues_mb, 1),
    "ram_consumida_mb"       : ram_usada_mb,
    "cpu_pct_al_terminar"    : cpu_despues,
    "n_cpus_disponibles"     : psutil.cpu_count(),
}])
reporte.to_csv("data/reports/reporte_modelo.csv", index=False)
log.info("✓ Reporte guardado en data/reports/reporte_modelo.csv")