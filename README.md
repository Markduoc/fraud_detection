# Credit Card Fraud Detection Pipeline

Pipeline modular de detección de fraude en transacciones con tarjeta de crédito. Procesa datos crudos, los limpia, valida su calidad y entrena un modelo de clasificación capaz de identificar transacciones fraudulentas.

---

## Estructura del proyecto

```
fraud_detection/
├── data/
│   ├── landing/          # Depósito inicial del CSV crudo
│   ├── raw/              # CSV tras la ingesta (generado automáticamente)
│   ├── processed/        # CSV limpio y con features (generado automáticamente)
│   └── reports/          # Reportes de validación y métricas del modelo
├── models/               # Modelo entrenado (.pkl)
├── ingesta.py            # Etapa 1 — mueve el CSV de landing/ a raw/
├── limpieza.py           # Etapa 2 — limpia, anonimiza y genera features
├── validacion.py         # Etapa 3 — valida la calidad del dato procesado
├── modelado.py           # Etapa 4 — entrena y evalúa el modelo
└── requirements.txt      # Dependencias del proyecto
```

---

## Requisitos

- Python 3.8 o superior
- Las siguientes librerías:

```
pandas
numpy
scikit-learn
imbalanced-learn
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/fraud-detection.git
cd fraud-detection

# 2. Instalar dependencias
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## Uso

El pipeline se ejecuta en orden, un script a la vez.

**Paso 1** — Coloca el archivo `fraudTest.csv` dentro de `data/landing/`.

**Paso 2** — Ejecuta los scripts en secuencia:

```bash
python ingesta.py
python limpieza.py
python validacion.py
python modelado.py
```

Cada script genera un log en consola con el estado de la ejecución. Si alguna etapa falla, las siguientes no deben ejecutarse hasta resolver el error.

---

## Descripción de cada etapa

### `ingesta.py`
Busca el archivo CSV en `data/landing/` y lo mueve a `data/raw/` con la fecha de ejecución en el nombre. Si no encuentra ningún archivo, lanza un error.

### `limpieza.py`
Toma el archivo más reciente de `data/raw/` y realiza:
- Eliminación de duplicados por `trans_num`
- Eliminación de filas con nulos en columnas críticas
- Filtrado de montos inválidos
- Anonimización de `cc_num` mediante hash SHA-256
- Eliminación de columnas de identidad personal (`first`, `last`, `street`)
- Generación de features derivadas:

| Feature | Descripción |
|---|---|
| `tx_hour` | Hora de la transacción |
| `tx_dow` | Día de la semana (0=Lunes, 6=Domingo) |
| `tx_is_weekend` | 1 si es sábado o domingo |
| `tx_is_night` | 1 si ocurre entre las 22:00 y las 05:59 |
| `tx_month` | Mes del año |
| `holder_age` | Edad del titular calculada desde fecha de nacimiento |
| `distance_km` | Distancia en km entre titular y comercio (fórmula de Haversine) |
| `log_amt` | Logaritmo del monto para estabilizar la escala |

El resultado se guarda en `data/processed/fraud_clean.csv`.

### `validacion.py`
Verifica la calidad del dato procesado mediante 8 validaciones:

- **Estructurales:** ausencia de nulos, valores válidos en `is_fraud`, montos positivos, distancias no negativas
- **Semánticas:** coherencia entre `tx_is_night` y `tx_hour`, entre `tx_is_weekend` y `tx_dow`, rango de edad del titular, coordenadas geográficas dentro de rangos válidos

El reporte de errores se guarda en `data/reports/reporte_validacion.csv`.

### `modelado.py`
Entrena un modelo de clasificación **Random Forest** para detectar fraudes. El proceso incluye:

- Split estratificado 80/20 (preserva la proporción de fraudes en train y test)
- Preprocesamiento con `StandardScaler` para variables numéricas y `OneHotEncoder` para categóricas
- Balanceo del dataset de entrenamiento con **SMOTE** (`sampling_strategy=0.15`)
- Entrenamiento con `class_weight="balanced"` como protección adicional contra el desbalance
- Evaluación con métricas orientadas a clases desbalanceadas: ROC-AUC, PR-AUC, Precision, Recall y F1

El modelo se guarda en `models/` y las métricas en `data/reports/reporte_modelo.csv`.

---

## Dataset

El dataset contiene transacciones con tarjeta de crédito con las siguientes características principales:

| Campo | Descripción |
|---|---|
| `trans_date_trans_time` | Fecha y hora de la transacción |
| `amt` | Monto en dólares |
| `category` | Categoría del comercio |
| `lat`, `long` | Coordenadas geográficas del titular |
| `merch_lat`, `merch_long` | Coordenadas geográficas del comercio |
| `is_fraud` | Variable objetivo — 1 = Fraude, 0 = Legítima |

Distribución: aproximadamente 0.39% de transacciones fraudulentas sobre un total de 555,719 registros.

---

## Resultados obtenidos

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.9720 |
| PR-AUC | 0.6407 |
| Recall (fraude) | 0.75 |
| Precision (fraude) | 0.20 |

El modelo detecta 3 de cada 4 fraudes reales. En detección de fraude bancario se prioriza el recall sobre la precisión, ya que es preferible investigar una falsa alarma que dejar pasar una transacción fraudulenta.