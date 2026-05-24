import os
import pickle
import logging
import numpy as np
import pandas as pd
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

preprocesador = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)
])

log.info("Preprocesando features...")
X_train_proc = preprocesador.fit_transform(X_train)
X_test_proc  = preprocesador.transform(X_test)  # solo transform, nunca fit

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

modelo = RandomForestClassifier(
    n_estimators=200,       # 200 árboles votando
    max_depth=15,           # profundidad máxima de cada árbol
    min_samples_leaf=4,     # mínimo de registros en cada hoja
    class_weight="balanced",# doble protección junto a SMOTE
    random_state=42,
    n_jobs=-1               # usa todos los núcleos del procesador
)
modelo.fit(X_train_bal, y_train_bal)
log.info("Entrenamiento completado")

# ── 7. EVALUACIÓN ───────────────────────────────────────────────────
log.info("Evaluando sobre el conjunto de test...")

y_prob = modelo.predict_proba(X_test_proc)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Métricas principales
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

log.info(f"\n{classification_report(y_test, y_pred, target_names=['Legítima','Fraude'])}")
log.info(f"ROC-AUC : {roc_auc:.4f}")
log.info(f"PR-AUC  : {pr_auc:.4f}")

# ── 8. GUARDAR MODELO Y REPORTE ─────────────────────────────────────
fecha = datetime.now().strftime("%Y-%m-%d")
ruta_modelo = f"models/fraud_model_{fecha}.pkl"

with open(ruta_modelo, "wb") as f:
    pickle.dump(modelo, f)
log.info(f"Modelo guardado en: {ruta_modelo}")

# Reporte de métricas en CSV
reporte = pd.DataFrame([{
    "fecha"          : fecha,
    "roc_auc"        : round(roc_auc, 4),
    "pr_auc"         : round(pr_auc, 4),
    "n_train"        : len(X_train),
    "n_test"         : len(X_test),
    "fraudes_test"   : int(y_test.sum()),
    "fraudes_detectados": int((y_pred == 1).sum())
}])
reporte.to_csv("data/reports/reporte_modelo.csv", index=False)
log.info("✓ Reporte guardado en data/reports/reporte_modelo.csv")