# Arquitectura y decisiones de diseño

> Documenta **cómo** funciona el sistema y, sobre todo, **por qué** se tomó cada
> decisión. Es la base de las secciones de metodología y modelización de la memoria.

---

## 1. Visión general del pipeline

```
vídeo .mp4
   │  muestreo de N fotogramas equiespaciados        (src/data/sampling.py)
   ▼
fotogramas
   │  detección y recorte del rostro (MTCNN, por lotes)  (src/data/face_extraction.py)
   ▼
rostros 224×224
   │  CNN preentrenada CONGELADA → vector              (src/features/embeddings.py)
   ▼
embeddings [n_frames, D]  ──► .npy cacheado en data/processed/
   │  modelado temporal (LSTM/GRU)                     (src/models/hybrid.py)
   ▼
probabilidad de manipulación
   │  umbral por coste de negocio                      (src/evaluation/metrics.py)
   ▼
veredicto + decisión operativa                          (app/app.py)
```

En paralelo, **Grad-CAM** (`src/explainability/gradcam.py`) explica la componente
espacial sobre los fotogramas.

## 2. Base teórica

### Redes Neuronales Convolucionales (CNN) y *transfer learning*

Una CNN aprende a "ver" aplicando filtros que recorren la imagen detectando
patrones locales. Sus dos ideas fundacionales son la **conectividad local** (cada
neurona observa una pequeña región) y la **compartición de pesos** (el mismo filtro
se aplica en toda la imagen), lo que permite reconocer un patrón esté donde esté
con muchísimos menos parámetros que una red densa. Al apilar capas se construye una
**jerarquía de características**: bordes y texturas en las primeras, partes del
rostro en las intermedias, estructuras complejas en las profundas.

Aquí no se entrena una CNN desde cero: se aplica ***transfer learning***,
reutilizando una red preentrenada en ImageNet como **extractor de características
congelado** que convierte cada rostro en un *embedding*.

Los dos *backbones* comparados encarnan filosofías distintas:

| Backbone | Idea central | Dimensión del embedding |
|---|---|---|
| **EfficientNet-B0** | Escalado compuesto y equilibrado de profundidad, anchura y resolución: más precisión con menos parámetros | 1280 |
| **ResNet-50** | Conexiones residuales (atajos que suman la entrada a la salida del bloque), que resuelven el desvanecimiento del gradiente y permiten gran profundidad | 2048 |

Su papel es **espacial**: detectar en cada fotograma los artefactos del deepfake
(bordes de mezcla en el contorno facial, texturas de piel incoherentes, anomalías
en ojos y dientes).

### Redes recurrentes LSTM

Un fotograma aislado no cuenta toda la historia: los deepfakes suelen fallar en la
**coherencia temporal** (parpadeos anómalos, micromovimientos rígidos, artefactos
que "parpadean" entre fotogramas). Capturarlo exige procesar *secuencias*.

Las redes recurrentes (RNN) mantienen un **estado oculto** que se actualiza paso a
paso, actuando como memoria. Su problema clásico es el **desvanecimiento del
gradiente**: al retropropagar el error en el tiempo la señal se diluye y se olvidan
las dependencias largas. La **LSTM** lo resuelve con una **celda de memoria**
regulada por tres puertas aprendibles:

- **puerta de olvido** — qué descartar de la memoria;
- **puerta de entrada** — qué información nueva incorporar;
- **puerta de salida** — qué exponer al siguiente paso.

En esta arquitectura, la LSTM consume la secuencia de *embeddings* y su estado
final resume la dinámica temporal del clip.

### Por qué el híbrido

Cada técnica responde una pregunta distinta y complementaria:

- la **CNN** → *¿qué aspecto tiene cada fotograma?*
- la **LSTM** → *¿cómo evoluciona en el tiempo?*

Esa combinación espacio-temporal es justamente lo que un enfoque tabular o un
AutoML no puede modelar, y es el argumento central del TFM.

### Grad-CAM

Genera un mapa de calor a partir de los **gradientes** de la puntuación de "fake"
respecto a las activaciones de la última capa convolucional: los canales cuyo
gradiente medio es mayor pesan más, y su combinación ponderada indica qué regiones
sustentaron la decisión.

## 3. Decisiones de diseño (y su justificación)

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| **Backbone congelado** (*feature extraction*) | *Fine-tuning* extremo a extremo | Permite cachear los embeddings y entrenar en segundos; el fine-tuning multiplicaría el coste. Queda implementado en `EndToEndModel` como línea futura |
| **Pipeline fusionado** vídeo→embedding | Dos pasos con fotogramas en disco | Escribir ~80.000 imágenes sueltas en Google Drive era el mayor cuello de botella; ahora solo se escribe un `.npy` por vídeo |
| **N fotogramas equiespaciados** (16) | Procesar el vídeo completo | El coste crece linealmente con los fotogramas y la señal se satura pronto; 16 capturan la dinámica a coste asumible |
| **Recorte facial (MTCNN)** | Fotograma completo | Elimina ruido de fondo y concentra al modelo donde están los artefactos |
| **Compresión c23** | `raw` (sin pérdida) | c23 es el *benchmark* estándar de FF++ y reduce el volumen en un orden de magnitud |
| **Umbral por coste** | Umbral fijo 0.5 / máximo F1 | En KYC un falso negativo cuesta mucho más que un falso positivo; el punto de operación debe reflejarlo |
| **Split oficial con respaldo estratificado** | Solo split oficial | Con subconjuntos pequeños el split oficial puede dejar conjuntos vacíos; el respaldo garantiza train/val/test con ambas clases |
| **Grad-CAM sobre la cabeza del baseline** | Grad-CAM sobre la LSTM | La LSTM opera sobre vectores, no sobre imágenes; se explica la componente espacial con el clasificador por fotograma y la temporal con la curva de probabilidad |

## 4. Optimizaciones de rendimiento

Pensadas para que el pipeline sea viable sobre Google Drive, donde cada operación
de fichero es una petición de red:

- **Sin fotogramas intermedios en disco:** un `.npy` por vídeo en lugar de 16 JPG.
- **Detección facial por lotes:** MTCNN procesa los 16 fotogramas de una vez.
- **Pasada compartida entre *backbones*:** al comparar EfficientNet y ResNet, la
  detección facial (la parte cara) se hace **una sola vez** y alimenta a ambos.
- **Escaneo único de directorios:** una lectura por carpeta en vez de miles de
  comprobaciones individuales (esto reducía de horas a segundos la detección de
  vídeos pendientes).
- **Reutilización del manifiesto:** los metadatos de vídeos ya procesados se leen
  del CSV, sin reabrir los `.npy`.
- **Caché en RAM de embeddings:** cada `.npy` se lee de Drive una única vez por
  sesión; el resto de épocas y experimentos trabajan en memoria.
- **Carga perezosa de modelos:** si todo está cacheado, no se cargan las redes ni
  se toca la GPU.
- **Idempotencia total:** cada etapa detecta lo ya hecho, de modo que una sesión
  interrumpida se retoma sin repetir trabajo.

## 5. Diseño experimental

| Experimento | Qué responde | Salida |
|---|---|---|
| Baseline vs Híbrido vs AutoML | ¿Aporta valor modelar el tiempo? | `tabla_comparativa.csv` |
| Cross-manipulation | ¿Generaliza a una manipulación no vista? | métricas sobre el método *holdout* |
| Curva de aprendizaje | ¿Cuántos datos hacen falta? | `curva_aprendizaje.png/.csv` |
| EfficientNet vs ResNet | ¿Qué extractor es mejor aquí? | `comparativa_backbones.csv` |
| Métricas por método | ¿Qué manipulaciones son más difíciles? | `metricas_por_metodo.csv`, `auc_por_metodo.png` |

En todos los casos el **conjunto de test se mantiene fijo**, para que las
comparaciones sean justas.

## 6. Reproducibilidad

- Semilla global fija (42) aplicada a `random`, `numpy` y PyTorch (incluido
  `cudnn.deterministic`), en `src/utils/seeds.py`.
- Toda la parametrización en `config/config.yaml`, sin números mágicos dispersos.
- Batería de pruebas con datos sintéticos en `tests/` que valida cada fase sin
  necesidad del dataset.
- `pip freeze > requirements-lock.txt` antes de la entrega.
