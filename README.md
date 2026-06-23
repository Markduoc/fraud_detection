# Credit Card Fraud Detection Pipeline

Pipeline modular de detección de fraude en transacciones con tarjeta de crédito. Procesa datos crudos, los limpia, valida su calidad, entrena un modelo de clasificación y expone los resultados en un dashboard interactivo.

---

## Estructura del proyecto

```
fraud_detection/
├── data/
│   ├── landing/          # Deposito inicial del CSV crudo
│   ├── raw/              # CSV tras la ingesta (generado automaticamente)
│   ├── processed/        # CSV limpio y con features (generado automaticamente)
│   └── reports/          # Reportes de validacion, metricas del modelo y recursos
├── models/               # Modelo entrenado (.pkl) + preprocesador
├── ingesta.py            # Etapa 1 — mueve el CSV de landing/ a raw/
├── limpieza.py           # Etapa 2 — limpia, anonimiza y genera features
├── validacion.py         # Etapa 3 — valida la calidad del dato procesado
├── modelado.py           # Etapa 4 — entrena, evalua y mide rendimiento del modelo
├── prediccion.py         # Etapa 5 — carga el modelo y predice transacciones nuevas
├── dashboard.py          # Dashboard interactivo con Streamlit
└── requirements.txt      # Dependencias del proyecto
```

---

## Requisitos

- Python 3.8 o superior
- Las siguientes librerias:

```
pandas
numpy
scikit-learn
imbalanced-learn
psutil
streamlit
matplotlib
seaborn
```

---

## Instalacion

```bash
# 1. Clonar el repositorio
git clone https://github.com/Markduoc/fraud_detection.git
cd fraud_detection

# 2. Instalar dependencias
pip install pandas numpy scikit-learn imbalanced-learn psutil streamlit matplotlib seaborn
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
python prediccion.py
```

**Paso 3** — Levanta el dashboard:

```bash
python -m streamlit run dashboard.py
```

El dashboard queda disponible en `http://localhost:8501`.

Cada script genera un log en consola con el estado de la ejecucion. Si alguna etapa falla, las siguientes no deben ejecutarse hasta resolver el error.

---

## Descripcion de cada etapa

### `ingesta.py`
Busca el archivo CSV en `data/landing/` y lo mueve a `data/raw/` con la fecha de ejecucion en el nombre. Si no encuentra ningun archivo, lanza un error.

### `limpieza.py`
Toma el archivo mas reciente de `data/raw/` y realiza:
- Eliminacion de duplicados por `trans_num`
- Eliminacion de filas con nulos en columnas criticas
- Filtrado de montos invalidos
- Anonimizacion de `cc_num` mediante hash SHA-256
- Eliminacion de columnas de identidad personal (`first`, `last`, `street`)
- Generacion de features derivadas:

| Feature | Descripcion |
|---|---|
| `tx_hour` | Hora de la transaccion |
| `tx_dow` | Dia de la semana (0=Lunes, 6=Domingo) |
| `tx_is_weekend` | 1 si es sabado o domingo |
| `tx_is_night` | 1 si ocurre entre las 22:00 y las 05:59 |
| `tx_month` | Mes del año |
| `holder_age` | Edad del titular calculada desde fecha de nacimiento |
| `distance_km` | Distancia en km entre titular y comercio (formula de Haversine) |
| `log_amt` | Logaritmo del monto para estabilizar la escala |

El resultado se guarda en `data/processed/fraud_clean.csv`.

### `validacion.py`
Verifica la calidad del dato procesado mediante 8 validaciones:

- **Estructurales:** ausencia de nulos, valores validos en `is_fraud`, montos positivos, distancias no negativas
- **Semanticas:** coherencia entre `tx_is_night` y `tx_hour`, entre `tx_is_weekend` y `tx_dow`, rango de edad del titular, coordenadas geograficas dentro de rangos validos

El reporte de errores se guarda en `data/reports/reporte_validacion.csv`.

### `modelado.py`
Entrena un modelo de clasificacion **Random Forest** para detectar fraudes. El proceso incluye:

- Split estratificado 80/20 (preserva la proporcion de fraudes en train y test)
- Preprocesamiento con `StandardScaler` para variables numericas y `OneHotEncoder` en formato sparse para categoricas, reduciendo el consumo de RAM entre un 60-80% respecto a una matriz densa
- Balanceo del dataset de entrenamiento con **SMOTE** (`sampling_strategy=0.15`)
- Entrenamiento con `class_weight="balanced"` como proteccion adicional contra el desbalance
- Uso controlado de CPU: el modelo utiliza la mitad de los nucleos fisicos disponibles (`n_jobs = cpu_count // 2`) para evitar saturacion del sistema
- Medicion de recursos durante el entrenamiento: tiempo, RAM antes y despues, y uso de CPU
- Evaluacion con metricas orientadas a clases desbalanceadas: ROC-AUC, PR-AUC, Precision, Recall y F1

El modelo y el preprocesador se guardan juntos en `models/` como un unico archivo `.pkl`. Las metricas del modelo y las metricas de recursos se guardan en `data/reports/reporte_modelo.csv`.

### `prediccion.py`
Carga el modelo y el preprocesador mas recientes desde `models/` y los aplica sobre transacciones nuevas. Replica exactamente el mismo preprocesamiento de `limpieza.py` (anonimizacion, features temporales, distancia Haversine, log del monto) antes de pasar los datos al modelo. Muestra para cada transaccion el resultado (legitima o fraude) y la probabilidad asociada.

### `dashboard.py`
Dashboard interactivo desarrollado con Streamlit. Lee los archivos generados por el pipeline y presenta:

- **KPIs principales:** ROC-AUC, PR-AUC, tamaño de train/test y cantidad de fraudes detectados
- **Rendimiento de recursos:** tiempo de entrenamiento, tiempo de prediccion, RAM consumida y nucleos disponibles, con grafico comparativo de RAM antes y despues del entrenamiento
- **Analisis del modelo:** matriz de confusion, curva ROC, curva Precision-Recall y top 10 variables mas importantes
- **Analisis de datos:** distribucion de clases y distribucion de montos separando transacciones legitimas y fraudulentas

---

## Dataset

El dataset contiene transacciones con tarjeta de credito con las siguientes caracteristicas principales:

| Campo | Descripcion |
|---|---|
| `trans_date_trans_time` | Fecha y hora de la transaccion |
| `amt` | Monto en dolares |
| `category` | Categoria del comercio |
| `lat`, `long` | Coordenadas geograficas del titular |
| `merch_lat`, `merch_long` | Coordenadas geograficas del comercio |
| `is_fraud` | Variable objetivo — 1 = Fraude, 0 = Legitima |

Distribucion: aproximadamente 0.39% de transacciones fraudulentas sobre un total de 555,719 registros.

---

## Resultados obtenidos

| Metrica | Valor |
|---|---|
| ROC-AUC | 0.9720 |
| PR-AUC | 0.6407 |
| Recall (fraude) | 0.75 |
| Precision (fraude) | 0.20 |

El modelo detecta 3 de cada 4 fraudes reales. En deteccion de fraude bancario se prioriza el recall sobre la precision, ya que es preferible investigar una falsa alarma que dejar pasar una transaccion fraudulenta.

---

## Optimizaciones implementadas

**Uso de CPU**
El parametro `n_jobs` del RandomForestClassifier se calcula dinamicamente como la mitad de los nucleos fisicos disponibles en la maquina. Esto evita que el entrenamiento sature el procesador y permite que el sistema operativo mantenga recursos disponibles para otros procesos durante la ejecucion.

**Uso de RAM**
El `OneHotEncoder` se configura con `sparse_output=True`, lo que genera una matriz dispersa que almacena unicamente los valores distintos de cero. La conversion a matriz densa (`.toarray()`) se realiza solo en el momento en que el modelo la necesita. Esto reduce significativamente el pico de memoria durante el preprocesamiento de variables categoricas.

**Preprocesador persistente**
El preprocesador entrenado se guarda junto al modelo en el mismo archivo `.pkl`. Esto garantiza que cualquier prediccion futura aplique exactamente la misma transformacion (mismos promedios, desviaciones y categorias conocidas) que se uso durante el entrenamiento.