import os
import glob
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── 1. CARGA EL MODELO MÁS RECIENTE ─
archivos = sorted(glob.glob("models/fraud_model_*.pkl"), reverse=True)

if not archivos:
    log.error("No se encontró ningún modelo — ejecuta modelado.py primero")
    raise FileNotFoundError("No hay modelos en models/")

ruta_modelo = archivos[0]
log.info(f"Cargando modelo: {ruta_modelo}")

with open(ruta_modelo, "rb") as f:
    pipeline = pickle.load(f)

modelo        = pipeline["modelo"]
preprocesador = pipeline["preprocesador"]
log.info("Modelo y preprocesador cargados exitosamente")

# ── 2. FUNCIÓN DE PREPROCESAMIENTO ─
# Replica los mismos pasos que hace limpieza.py
# para que los datos nuevos sean comparables con los de entrenamiento

def hashear(valor: str) -> str:
    return hashlib.sha256(str(valor).encode()).hexdigest()[:16]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def preparar_transaccion(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas crudas (igual que fraudTest.csv)
    y devuelve el DataFrame listo para el modelo.
    """
    df = df_raw.copy()

    # Anonimizar cc_num
    df["cc_num"] = df["cc_num"].apply(hashear)

    # Eliminar columnas de identidad en los datos
    df.drop(columns=["first", "last", "street"], errors="ignore", inplace=True)

    # Features temporales
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["tx_hour"]       = dt.dt.hour
    df["tx_dow"]        = dt.dt.dayofweek
    df["tx_is_weekend"] = (df["tx_dow"] >= 5).astype(int)
    df["tx_is_night"]   = ((df["tx_hour"] >= 22) | (df["tx_hour"] <= 5)).astype(int)
    df["tx_month"]      = dt.dt.month

    # Edad del titular
    dob = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")
    df["holder_age"] = (dt - dob).dt.days // 365

    # Distancia titular
    df["distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Log del monto
    df["log_amt"] = np.log1p(df["amt"])

    # Eliminar columnas que no usa el modelo
    df.drop(columns=["trans_date_trans_time", "dob", "trans_num",
                      "unix_time", "Unnamed: 0"], errors="ignore", inplace=True)

    # Eliminar is_fraud si es que viene
    df.drop(columns=["is_fraud"], errors="ignore", inplace=True)

    return df

# ── 3. TRANSACCIONES DE PRUEBA ───────────────────────────────────────
# Estas son transacciones inventadas para testear que el modelo funciona.
transacciones_prueba = pd.DataFrame([
    {   # Transacción 1: perfil de bajo riesgo
        "trans_date_trans_time": "2019-06-21 12:30:00",
        "cc_num": "1234567890123456",
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "grocery_pos",
        "amt": 15.50,
        "gender": "F",
        "city": "Mataichi",
        "state": "TX",
        "zip": 75001,
        "lat": 33.0, "long": -97.0,
        "city_pop": 50000,
        "job": "Teacher",
        "dob": "1985-03-15",
        "trans_num": "test_001",
        "unix_time": 1561112400,
        "merch_lat": 33.1, "merch_long": -97.1,
    },
    {   # Transacción 2: perfil de alto riesgo (monto alto, madrugada, lejos)
        "trans_date_trans_time": "2019-06-21 02:47:00",
        "cc_num": "9999888877776666",
        "merchant": "fraud_Swaniawski, Nitzsche and Welch",
        "category": "shopping_net",
        "amt": 1200.00,
        "gender": "M",
        "city": "Houston",
        "state": "TX",
        "zip": 77001,
        "lat": 29.7, "long": -95.3,
        "city_pop": 2300000,
        "job": "Student",
        "dob": "2000-11-20",
        "trans_num": "test_002",
        "unix_time": 1561081620,
        "merch_lat": 36.1, "merch_long": -86.8,
    },
])

# ── 4. PREPROCESAR Y PREDECIR ─
log.info("Preparando transacciones de prueba...")
df_listo = preparar_transaccion(transacciones_prueba)

log.info("Aplicando preprocesador del entrenamiento...")
X_proc = preprocesador.transform(df_listo)

log.info("Generando predicciones...")
predicciones  = modelo.predict(X_proc)
probabilidades = modelo.predict_proba(X_proc)[:, 1]

# ── 5. MOSTRAR RESULTADOS ─
print("\n" + "="*55)
print("         RESULTADO DE PREDICCIONES")
print("="*55)

for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
    etiqueta = "🚨 FRAUDE" if pred == 1 else "✅ LEGÍTIMA"
    monto    = transacciones_prueba.iloc[i]["amt"]
    hora     = transacciones_prueba.iloc[i]["trans_date_trans_time"]
    print(f"\nTransacción #{i+1}")
    print(f"  Hora:        {hora}")
    print(f"  Monto:       ${monto:,.2f}")
    print(f"  Resultado:   {etiqueta}")
    print(f"  Probabilidad de fraude: {prob*100:.1f}%")

print("\n" + "="*55)
log.info("✓ Predicción completada")