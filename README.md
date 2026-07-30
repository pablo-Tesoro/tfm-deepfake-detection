# TFM — Detección de Deepfakes en Vídeo (perfil Data Scientist)

> Detección de manipulaciones sintéticas (*deepfakes*) en secuencias de vídeo
> mediante una arquitectura híbrida espacio-temporal (CNN + LSTM), explicabilidad
> (Grad-CAM) y una aplicación web de verificación forense.

Máster en Big Data, Ciencia de Datos e Inteligencia Artificial — UCM.

---

## Idea en una frase

Un sistema que, dado un vídeo, predice si el rostro ha sido manipulado y **muestra
visualmente por qué** lo cree, anclado en un caso de negocio concreto:
la **videoidentificación en el *onboarding* bancario** (ver `docs/00_alcance_caso_negocio.md`).

## Enfoque técnico

El cuello de botella de este problema es el coste computacional del vídeo. La
estrategia evita entrenar de extremo a extremo sobre vídeo crudo y cachea los
resultados intermedios caros:

1. **Vídeo → embeddings (fusionado)** — por cada vídeo se muestrean N fotogramas,
   se aísla el rostro (MTCNN, detección **por lotes**) y una CNN preentrenada y
   *congelada* (EfficientNet-B0) lo convierte en un vector. Todo ocurre **en
   memoria**: solo se escribe un `.npy` por vídeo en `data/processed/`. Evita
   decenas de miles de imágenes sueltas, que en Google Drive son el gran cuello
   de botella.
2. **Modelado temporal** — una LSTM/GRU consume las secuencias de embeddings y
   evalúa la coherencia temporal del clip (parpadeos, micromovimientos).
3. **Comparativa** — *baseline* a nivel de frame vs. híbrido vs. AutoML, más un
   experimento **cross-manipulation** (entrenar con 3 métodos, evaluar en el 4º).
4. **Experimentos avanzados** — curva de aprendizaje (AUC vs nº de vídeos),
   EfficientNet-B0 vs ResNet-50, y métricas desglosadas por método de manipulación.
5. **Explicabilidad** — Grad-CAM (qué regiones del rostro) + curva de probabilidad
   por fotograma (en qué momentos del clip).
6. **Negocio** — el umbral de decisión no maximiza el F1: minimiza el **coste
   esperado**, penalizando más los falsos negativos (dejar pasar un deepfake).
7. **Productivización** — **VERIFAKE**, app Gradio: vídeo → veredicto + confianza +
   mapas de calor + decisión operativa (aprobar / revisar / rechazar).

## Estructura del repositorio

```
TFM_Deepfake_Detection/
├── README.md
├── requirements.txt          # dependencias (stack PyTorch)
├── .gitignore                # excluye datos y modelos pesados
├── run_all.py                # ORQUESTADOR de extremo a extremo
├── config/
│   └── config.yaml           # parámetros centrales (semillas, rutas, modelo)
├── data/                     # (vacío en git; los datos NO se versionan)
│   ├── raw/                  # vídeos FF++ (c23) + splits oficiales
│   ├── interim/              # rostros recortados (solo ruta clásica de 2 pasos)
│   └── processed/            # embeddings .npy + manifiesto CSV
│       └── resnet50/         # embeddings del backbone alternativo
├── notebooks/
│   ├── RUN_ALL.ipynb         # ▶ ejecutar TODO y lanzar la app (Colab/Kaggle)
│   ├── 00_setup_colab.ipynb  # preparación paso a paso
│   ├── 01_eda.ipynb          # análisis exploratorio
│   ├── 02_modeling.ipynb     # modelización y evaluación
│   ├── 03_explainability.ipynb  # Grad-CAM y explicabilidad temporal
│   ├── 04_app.ipynb          # lanzar VERIFAKE
│   └── 05_experimentos.ipynb # curva de aprendizaje, backbones, por método
├── src/
│   ├── data/                 # inventario FF++, muestreo, extracción facial, dataset
│   ├── features/             # embeddings (pipeline fusionado multi-backbone)
│   ├── models/               # baseline (media+MLP) e híbrido CNN+LSTM
│   ├── training/             # bucle de entrenamiento + early stopping
│   ├── evaluation/           # métricas técnicas y de coste de negocio
│   ├── explainability/       # Grad-CAM
│   ├── experiments/          # curva de aprendizaje, backbones, por método
│   └── utils/                # semillas, config, rutas (local/Drive)
├── app/
│   └── app.py                # VERIFAKE (interfaz Gradio)
├── tests/                    # pruebas de cada fase (datos sintéticos)
├── reports/
│   ├── memoria/              # la memoria de 20 caras
│   └── figures/              # figuras y tablas CSV generadas
└── docs/
    ├── 00_alcance_caso_negocio.md   # alcance, caso de negocio y objetivos
    ├── 01_descarga_datos.md         # cómo descargar FaceForensics++
    ├── 02_arquitectura.md           # arquitectura, base teórica y decisiones
    ├── 03_etica_privacidad.md       # licencia, privacidad y uso dual
    └── 04_guia_memoria.md           # mapa guía UCM → memoria → figuras
```

## Ejecución

### Todo de una vez (recomendado)

Abre `notebooks/RUN_ALL.ipynb` en **Colab** (o Kaggle), activa la GPU, edita
`REPO_URL` y ejecuta todo. Encadena: splits → (descarga opcional) →
vídeo→embeddings → entrenamiento → evaluación → experimentos → app con enlace
público `*.gradio.live`.

```python
from run_all import run_pipeline

demo = run_pipeline(
    download=False,               # True para descargar FF++ (script oficial en el repo)
    n_videos=150,                 # None = todos los vídeos
    retrain=True,                 # reentrenar (False reutiliza los checkpoints)
    make_figs=True,               # figuras básicas para la memoria
    experiments=True,             # curva de aprendizaje + backbones + por método
    compare_backbone="resnet50",  # None omite la comparativa (ahorra una pasada)
    share=True,                   # enlace público de la app
)
```

**Todo es idempotente:** los embeddings ya calculados, los modelos entrenados y los
vídeos descargados no se repiten. Si la sesión se corta, relanza y retoma.

### Paso a paso

`00_setup_colab.ipynb` para preparar el entorno, y después los notebooks 01 (EDA),
02 (modelado), 03 (explicabilidad), 04 (app) y 05 (experimentos).

### En local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "from src.utils.seeds import set_seed, load_config; set_seed(); print(load_config())"
python run_all.py
```

En Colab/Kaggle **no reinstales** `torch`/`torchvision` (ya vienen).

### Pruebas

```bash
for t in tests/test_*.py; do python "$t"; done
```

## Rutas y workspace

Las salidas (datos, figuras, modelos) se resuelven en `src/utils/paths.py`:

- **Local:** `workspace_root: null` en `config.yaml` → todo vive en el proyecto.
- **Colab:** `TFM_WORKSPACE` (lo fija el notebook) apunta a Google Drive.
- **Kaggle:** trabaja en `/kaggle/working` y sincroniza con Drive vía rclone.

El código no cambia entre entornos.

## Datos

- **Dataset:** FaceForensics++ (compresión c23): 1000 vídeos reales y 4000
  manipulados con 4 métodos (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **Acceso oficial:** https://github.com/ondyari/FaceForensics — formulario de
  Google; una vez aceptado, envían el script de descarga (ver `docs/01_descarga_datos.md`).
- **Licencia:** uso de investigación/académico. **No redistribuir** los vídeos.
  Es obligatorio **citar** a Rössler et al. (2019) en la bibliografía.

## Resultados generados

En `reports/figures/` (los produce `run_all.py`):

| Archivo | Contenido |
|---|---|
| `01_distribucion_clases.png` | Reparto Real / Fake del dataset |
| `tabla_comparativa.csv` | Baseline vs Híbrido CNN+LSTM en test |
| `matriz_confusion.png` | Matriz de confusión del híbrido |
| `curva_aprendizaje.png/.csv` | AUC y F1 según el nº de vídeos de entrenamiento |
| `comparativa_backbones.csv` | EfficientNet-B0 vs ResNet-50 |
| `metricas_por_metodo.csv`, `auc_por_metodo.png` | Rendimiento por manipulación |

## Documentación

| Documento | Contenido |
|---|---|
| `docs/00_alcance_caso_negocio.md` | Problema, caso de negocio ancla, objetivos, alcance y riesgos |
| `docs/01_descarga_datos.md` | Obtención y descarga de FaceForensics++ |
| `docs/02_arquitectura.md` | Pipeline, base teórica (CNN, LSTM, Grad-CAM), decisiones de diseño y optimizaciones |
| `docs/03_etica_privacidad.md` | Licencia del dataset, privacidad, uso dual y limitaciones |
| `docs/04_guia_memoria.md` | Mapa entre los requisitos de la guía UCM, las secciones de la memoria y las figuras |

## Estado del proyecto

- [x] **Fase 0** — Preparación: repo, entorno, alcance y caso de negocio.
- [x] **Fase 1** — Datos y EDA: descarga, inventario, extracción facial, EDA.
- [x] **Fase 2** — Modelización: embeddings, baseline, híbrido, cross-manipulation.
- [x] **Fase 3** — Explicabilidad: Grad-CAM, curva temporal, umbral por coste.
- [x] **Fase 4** — Productivización: app VERIFAKE con enlace público.
- [x] **Experimentos avanzados** — curva de aprendizaje, backbones, por método.
- [x] **Orquestación** — `run_all.py` + `RUN_ALL.ipynb` (Colab y Kaggle).
- [ ] **Fase 5** — Memoria (20 caras), anexos, vídeo (5 min) y checklist final.

## Entregables del TFM

1. **Memoria** (máx. 20 caras, orientada a negocio) en PDF/HTML/DOCX.
2. **Vídeo** MP4 de máx. 5 min (< 50 MB), con voz en off descriptiva.
3. **Anexos**: código (este repo) y estudios detallados de EDA/modelos.

## Reproducibilidad

Semilla global fija (42) en `config.yaml`, aplicada a `random`, `numpy` y PyTorch.
Antes de la entrega, congela las versiones exactas:

```bash
pip freeze > requirements-lock.txt
```

## Autor

Pablo Tesoro García
