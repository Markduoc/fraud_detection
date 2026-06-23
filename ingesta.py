import os
import shutil
import glob
import logging
import sys
from datetime import datetime

# Se configura el LOG para dejar registro
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── DESACOPLAMIENTO DE RUTAS ───
# Se lee del entorno o se usan valores por defecto
LANDING_DIR = os.getenv("FRAUD_LANDING_DIR", "data/landing")
RAW_DIR = os.getenv("FRAUD_RAW_DIR", "data/raw")

log.info(f"Configuración cargada desde entorno:")
log.info(f"  LANDING_DIR: {LANDING_DIR}")
log.info(f"  RAW_DIR: {RAW_DIR}")

def main():
    """Función principal con manejo de excepciones controlado."""
    try:
        os.makedirs(RAW_DIR, exist_ok=True)

        # Se busca el CSV en landing/
        archivos = glob.glob(f"{LANDING_DIR}/*.csv")

        if not archivos:
            log.error(f"No se encontró ningún CSV en {LANDING_DIR}/")
            raise FileNotFoundError(f"No hay archivos CSV en {LANDING_DIR}/")

        origen = archivos[0]
        fecha = datetime.now().strftime("%Y-%m-%d")
        destino = f"{RAW_DIR}/fraud_raw_{fecha}.csv"

        log.info(f"Archivo encontrado: {origen}")
        shutil.move(origen, destino)
        log.info(f"Movido exitosamente a: {destino}")
        return 0

    except FileNotFoundError as e:
        log.error(f"Archivo no encontrado: {str(e)}")
        return 1
    except PermissionError:
        log.error("Permiso denegado: no hay derechos suficientes para mover archivos")
        return 1
    except Exception as e:
        log.error(f"Error durante la ingestión de datos: {type(e).__name__}")
        log.debug(f"Detalles técnicos: {str(e)}", exc_info=False)
        return 1

if __name__ == "__main__":
    sys.exit(main())