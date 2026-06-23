import os
import glob
import hashlib
import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ─── DESACOPLAMIENTO DE RUTAS ───
RAW_DIR = os.getenv("FRAUD_RAW_DIR", "data/raw")
PROCESSED_DIR = os.getenv("FRAUD_PROCESSED_DIR", "data/processed")

log.info(f"Configuración cargada desde entorno:")
log.info(f"  RAW_DIR: {RAW_DIR}")
log.info(f"  PROCESSED_DIR: {PROCESSED_DIR}")

def hashear(valor: str) -> str:
    """Hashea un valor usando SHA256."""
    return hashlib.sha256(str(valor).encode()).hexdigest()[:16]

def haversine_km_vectorized(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia de Haversine de forma VECTORIZADA usando numpy.
    Esto es órdenes de magnitud más rápido que la versión row-wise de pandas.
    
    Args:
        lat1, lon1: Arrays de latitud/longitud del titular
        lat2, lon2: Arrays de latitud/longitud del comercio
    
    Returns:
        Array de distancias en km
    """
    R = 6371.0  # Radio de la Tierra en km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    # Fórmula de Haversine vectorizada
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def main():
    """Función principal con manejo de excepciones controlado."""
    try:
        # SE CARGA EL ARCHIVO MAS RECIENTE DE RAW
        archivos = sorted(glob.glob(f"{RAW_DIR}/fraud_raw_*.csv"), reverse=True)

        if not archivos:
            log.error(f"No se encontró ningún archivo en {RAW_DIR}/")
            raise FileNotFoundError(f"No hay archivos en {RAW_DIR}/")

        ruta_entrada = archivos[0]
        log.info(f"Procesando: {ruta_entrada}")

        df = pd.read_csv(ruta_entrada)

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        log.info(f"── Estado ANTES de la limpieza ──")
        log.info(f"  Filas:      {len(df):,}")
        log.info(f"  Columnas:   {df.shape[1]}")
        log.info(f"  Nulos:      {df.isnull().sum().sum()}")
        log.info(f"  Duplicados: {df.duplicated(subset=['trans_num']).sum()}")
        log.info(f"  Fraudes:    {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

        # LIMPIEZA

        # 1. Eliminar duplicados por trans_num (ID único de transacción)
        antes = len(df)
        df = df.drop_duplicates(subset=["trans_num"])
        log.info(f"[Limpieza] Duplicados eliminados: {antes - len(df)}")

        # 2. Eliminar filas con nulos en columnas críticas
        criticas = ["amt", "category", "merchant", "lat", "long",
                    "merch_lat", "merch_long", "is_fraud"]
        antes = len(df)
        df = df.dropna(subset=criticas)
        log.info(f"[Limpieza] Filas con nulos en columnas críticas: {antes - len(df)}")

        # 3. Filtrar montos inválidos (negativos o cero)
        antes = len(df)
        df = df[df["amt"] > 0]
        log.info(f"[Limpieza] Filas con monto inválido eliminadas: {antes - len(df)}")

        # ANONIMIZAR

        # cc_num se hashea — conserva identidad de tarjeta sin exponer el número
        log.info("[Anonimización] Hasheando cc_num...")
        df["cc_num"] = df["cc_num"].apply(hashear)

        # Datos personales que no aportan al modelo se eliminan
        df.drop(columns=["first", "last", "street"], inplace=True)
        log.info("[Anonimización] Columnas personales eliminadas: first, last, street")

        # FEATURE ENGINEERING (transformar datos sin procesar (raw data) en variables)

        log.info("[Features] Generando columnas derivadas...")

        # 1. Features temporales
        dt = pd.to_datetime(df["trans_date_trans_time"])
        df["tx_hour"]       = dt.dt.hour
        df["tx_dow"]        = dt.dt.dayofweek        # 0=Lunes … 6=Domingo
        df["tx_is_weekend"] = (df["tx_dow"] >= 5).astype(int)
        df["tx_is_night"]   = ((df["tx_hour"] >= 22) | (df["tx_hour"] <= 5)).astype(int)
        df["tx_month"]      = dt.dt.month

        # 2. Edad del titular calculada desde dob
        dob = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")
        df["holder_age"] = (dt - dob).dt.days // 365

        # 3. Distancia titular → comercio usando VECTORIZACIÓN NUMPY (optimizado)
        log.info("[Features] Calculando distancias con Haversine vectorizado...")
        df["distance_km"] = haversine_km_vectorized(
            df["lat"].values, df["long"].values,
            df["merch_lat"].values, df["merch_long"].values
        )

        # 4. Log del monto — estabiliza la escala para el modelo
        df["log_amt"] = np.log1p(df["amt"])

        # ELIMINAR COLUMNAS QUE YA NO SE NECESITAN
        df.drop(columns=["trans_date_trans_time", "dob", "trans_num",
                          "unix_time"], inplace=True)
        log.info("[Limpieza] Columnas de tiempo/id originales eliminadas")

        # GUARDAR EN processed/
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        ruta_salida = f"{PROCESSED_DIR}/fraud_clean.csv"
        df.to_csv(ruta_salida, index=False)

        log.info(f"── Estado DESPUÉS de la limpieza ──")
        log.info(f"  Filas:    {len(df):,}")
        log.info(f"  Columnas: {df.shape[1]}")
        log.info(f"  Fraudes:  {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
        log.info(f"  Guardado en: {ruta_salida}")
        log.info("✓ Limpieza completada exitosamente.")
        return 0

    except FileNotFoundError as e:
        log.error(f"Archivo no encontrado: {str(e)}")
        return 1
    except pd.errors.ParserError:
        log.error("Error al parsear CSV: el archivo puede estar corrupto")
        return 1
    except KeyError as e:
        log.error(f"Columna faltante en los datos: {str(e)}")
        return 1
    except Exception as e:
        log.error(f"Error durante la limpieza: {type(e).__name__}")
        log.debug(f"Detalles técnicos: {str(e)}", exc_info=False)
        return 1

if __name__ == "__main__":
    sys.exit(main())