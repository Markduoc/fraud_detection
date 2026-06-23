import os
import logging
import sys
import pandas as pd

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
REPORTS_DIR = os.getenv("FRAUD_REPORTS_DIR", "data/reports")

log.info(f"Configuración cargada desde entorno:")
log.info(f"  PROCESSED_DIR: {PROCESSED_DIR}")
log.info(f"  REPORTS_DIR: {REPORTS_DIR}")

def main():
    """Función principal con manejo de excepciones controlado."""
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # CARGAR DATOS LIMPIOS
        ruta = f"{PROCESSED_DIR}/fraud_clean.csv"
        if not os.path.exists(ruta):
            log.error(f"No se encontró fraud_clean.csv en {PROCESSED_DIR}")
            raise FileNotFoundError(ruta)

        df = pd.read_csv(ruta)
        log.info(f"Archivo cargado: {len(df):,} filas, {df.shape[1]} columnas")

        errores = []

        # VALIDACIONES ESTRUCTURALES

        # 1. Sin nulos tras la limpieza
        if df.isnull().values.any():
            cols_con_nulos = df.columns[df.isnull().any()].tolist()
            errores.append({
                "Capa": "Estructural",
                "Validación": "Nulos post-limpieza",
                "Detalle": f"Columnas afectadas: {cols_con_nulos}"
            })
            log.warning(f"  Nulos encontrados en: {cols_con_nulos}")
        else:
            log.info("  [OK] Sin nulos")

        # 2. is_fraud solo contiene 0 y 1
        valores_fraud = set(df["is_fraud"].unique())
        if not valores_fraud.issubset({0, 1}):
            errores.append({
                "Capa": "Estructural",
                "Validación": "Valores de is_fraud",
                "Detalle": f"Valores inesperados: {valores_fraud - {0,1}}"
            })
            log.warning(f"  is_fraud contiene valores inesperados: {valores_fraud}")
        else:
            log.info("  [OK] is_fraud solo contiene 0 y 1")

        # 3. Montos positivos
        negativos = (df["amt"] <= 0).sum()
        if negativos > 0:
            errores.append({
                "Capa": "Estructural",
                "Validación": "Montos positivos",
                "Detalle": f"{negativos} filas con amt <= 0"
            })
            log.warning(f"  {negativos} montos inválidos")
        else:
            log.info("  [OK] Todos los montos son positivos")

        # 4. Distancia no negativa
        neg_dist = (df["distance_km"] < 0).sum()
        if neg_dist > 0:
            errores.append({
                "Capa": "Estructural",
                "Validación": "Distancia no negativa",
                "Detalle": f"{neg_dist} filas con distance_km < 0"
            })
        else:
            log.info("  [OK] Distancias válidas")

        # ─── VALIDACIONES SEMÁNTICAS ───

        # 5. Coherencia tx_is_night vs tx_hour
        noche_incoherente = df[
            (df["tx_is_night"] == 1) &
            (df["tx_hour"] > 5) &
            (df["tx_hour"] < 22)
        ]
        if len(noche_incoherente) > 0:
            errores.append({
                "Capa": "Semántica",
                "Validación": "Coherencia tx_is_night",
                "Detalle": f"{len(noche_incoherente)} filas con tx_is_night=1 pero hora fuera de rango nocturno"
            })
            log.warning(f"  tx_is_night incoherente: {len(noche_incoherente)} filas")
        else:
            log.info("  [OK] tx_is_night coherente con tx_hour")

        # 6. Coherencia tx_is_weekend vs tx_dow
        finde_incoherente = df[
            (df["tx_is_weekend"] == 1) &
            (~df["tx_dow"].isin([5, 6]))
        ]
        if len(finde_incoherente) > 0:
            errores.append({
                "Capa": "Semántica",
                "Validación": "Coherencia tx_is_weekend",
                "Detalle": f"{len(finde_incoherente)} filas con tx_is_weekend=1 pero dow no es 5 ni 6"
            })
            log.warning(f"  tx_is_weekend incoherente: {len(finde_incoherente)} filas")
        else:
            log.info("  [OK] tx_is_weekend coherente con tx_dow")

        # 7. Edad razonable del titular (entre 15 y 100 años)
        edad_invalida = df[~df["holder_age"].between(15, 100)]
        if len(edad_invalida) > 0:
            errores.append({
                "Capa": "Semántica",
                "Validación": "Edad del titular",
                "Detalle": f"{len(edad_invalida)} titulares con edad fuera de rango (15-100)"
            })
            log.warning(f"  Edades fuera de rango: {len(edad_invalida)} filas")
        else:
            log.info("  [OK] Edades del titular dentro de rango")

        # 8. Coordenadas dentro de rangos geográficos válidos
        coord_invalidas = df[
            ~df["lat"].between(-90, 90) |
            ~df["long"].between(-180, 180) |
            ~df["merch_lat"].between(-90, 90) |
            ~df["merch_long"].between(-180, 180)
        ]
        if len(coord_invalidas) > 0:
            errores.append({
                "Capa": "Semántica",
                "Validación": "Coordenadas geográficas",
                "Detalle": f"{len(coord_invalidas)} filas con coordenadas fuera de rango"
            })
            log.warning(f"  Coordenadas inválidas: {len(coord_invalidas)} filas")
        else:
            log.info("  [OK] Coordenadas geográficas válidas")

        # ─── REPORTE FINAL ───
        reporte = pd.DataFrame(errores)
        ruta_reporte = f"{REPORTS_DIR}/reporte_validacion.csv"
        reporte.to_csv(ruta_reporte, index=False)

        if len(errores) == 0:
            log.info("✓ Validación completada — sin errores detectados")
        else:
            log.warning(f"✗ Validación completada — {len(errores)} problema(s) encontrado(s)")
            log.warning(f"  Revisa el reporte en: {ruta_reporte}")
        
        return 0

    except FileNotFoundError as e:
        log.error(f"Archivo no encontrado: {str(e)}")
        return 1
    except pd.errors.ParserError:
        log.error("Error al parsear CSV: el archivo puede estar corrupto")
        return 1
    except Exception as e:
        log.error(f"Error durante la validación: {type(e).__name__}")
        log.debug(f"Detalles técnicos: {str(e)}", exc_info=False)
        return 1

if __name__ == "__main__":
    sys.exit(main())

