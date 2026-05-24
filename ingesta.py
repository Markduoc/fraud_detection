import os
import shutil
import glob
import logging
from datetime import datetime

# Se configura el LOG para dejar registro
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# Rutas del pipeline
LANDING_DIR = "data/landing"
RAW_DIR     = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)

# Se busca el CSV en landing/
archivos = glob.glob(f"{LANDING_DIR}/*.csv")

if not archivos:
    log.error(f"No se encontró ningún CSV en {LANDING_DIR}/")
    raise FileNotFoundError(f"No hay archivos CSV en {LANDING_DIR}/")

origen  = archivos[0]
fecha   = datetime.now().strftime("%Y-%m-%d")
destino = f"{RAW_DIR}/fraud_raw_{fecha}.csv"

log.info(f"Archivo encontrado: {origen}")
shutil.move(origen, destino)
log.info(f"Movido exitosamente a: {destino}")