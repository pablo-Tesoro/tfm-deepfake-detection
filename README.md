# TFM — Detección de Deepfakes en Vídeo (perfil Data Scientist)

> Detección de manipulaciones sintéticas (*deepfakes*) en secuencias de vídeo
> mediante una arquitectura híbrida espacio-temporal (CNN + LSTM) y técnicas de
> explicabilidad (Grad-CAM), con una herramienta web de demostración.

Máster en Ciencia de Datos e Inteligencia Artificial — UCM.

---

## Idea en una frase

Un sistema que, dado un vídeo, predice si el rostro ha sido manipulado y **muestra
visualmente por qué** lo cree, pensado para un caso de negocio concreto:
la **videoidentificación en el *onboarding* bancario** (ver `docs/00_alcance_caso_negocio.md`).

## Enfoque técnico (resumen)

El cuello de botella de este problema es el coste computacional del vídeo. La
estrategia evita entrenar de extremo a extremo sobre vídeo crudo y desacopla el
*pipeline* en etapas cacheables:

1. **Extracción facial** — se muestrean N frames por vídeo y se recorta el rostro
   (MTCNN). Los recortes se guardan en `data/interim/`.
2. **Embeddings espaciales** — una CNN preentrenada y *congelada* (EfficientNet)
   convierte cada rostro en un vector. Se calculan **una sola vez** y se cachean
   en `data/processed/`. → barato y rápido de iterar.
3. **Modelado temporal** — una LSTM/GRU consume las secuencias de embeddings y
   evalúa la coherencia temporal del clip (parpadeos, micromovimientos).
4. **Comparativa** — *baseline* a nivel de frame vs. híbrido vs. AutoML, más un
   experimento **cross-manipulation** (entrenar con 3 métodos, evaluar en el 4º).
5. **Explicabilidad** — Grad-CAM genera mapas de calor sobre el rostro.
6. **Productivización** — app en Gradio: subir vídeo → predicción + mapa de calor.

## Estructura del repositorio

```
TFM_Deepfake_Detection/
├── README.md
├── requirements.txt          # dependencias (stack PyTorch)
├── .gitignore                # excluye datos/modelos pesados
├── config/
│   └── config.yaml           # parámetros centrales (semillas, rutas, modelo)
├── data/                     # (vacío en git; los datos NO se versionan)
│   ├── raw/                  # vídeos FF++ descargados (c23)
│   ├── interim/              # rostros recortados por frame
│   └── processed/            # embeddings cacheados
├── notebooks/                # EDA, extracción, modelado, resultados
├── src/
│   ├── data/                 # descarga, extracción facial, muestreo
│   ├── features/             # cálculo de embeddings
│   ├── models/               # baseline e híbrido
│   ├── evaluation/           # métricas y análisis de coste
│   └── utils/                # semillas, config, device
├── app/                      # interfaz Gradio (productivización)
├── reports/
│   ├── memoria/              # la memoria de 20 caras
│   └── figures/              # gráficas generadas
└── docs/
    └── 00_alcance_caso_negocio.md   # alcance y caso de negocio (Fase 0)
```

## Puesta en marcha

```bash
# 1. Entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Dependencias
pip install -r requirements.txt

# 3. Comprobar configuración y semillas
python -c "from src.utils.seeds import set_seed, load_config; set_seed(); print(load_config())"
```

En **Google Colab**, no reinstales `torch`/`torchvision` (ya vienen). Monta tu
Google Drive para persistir `data/processed/` entre sesiones.

## Datos

- **Dataset:** FaceForensics++ (compresión c23).
- **Acceso oficial:** https://github.com/ondyari/FaceForensics — requiere rellenar
  un formulario de Google; una vez aceptado, envían el script de descarga.
- **Licencia:** uso de investigación/académico. **No redistribuir** los vídeos.
  Es obligatorio **citar** a Rössler et al. (2019) en la bibliografía.

## Hoja de ruta por fases

- [~] **Fase 0** — Preparación: repo, entorno, alcance y caso de negocio. *(en curso)*
- [ ] **Fase 1** — Datos y EDA: descarga, extracción facial, análisis descriptivo.
- [ ] **Fase 2** — Modelización: embeddings, baseline, híbrido, cross-manipulation.
- [ ] **Fase 3** — Explicabilidad y negocio: Grad-CAM, métricas de coste, umbral.
- [ ] **Fase 4** — Productivización: app Gradio end-to-end.
- [ ] **Fase 5** — Memoria (20 caras), anexos, vídeo (5 min) y checklist final.

## Entregables del TFM

1. **Memoria** (máx. 20 caras, orientada a negocio) en PDF/HTML/DOCX.
2. **Vídeo** MP4 de máx. 5 min (< 50 MB), con voz en off descriptiva.
3. **Anexos**: código (este repo) y estudios detallados de EDA/modelos.

## Autor

[Tu Nombre y dos apellidos] — el ZIP de entrega se nombrará
`Nombre_Apellido1_Apellido2_TFM_Deepfakes.zip` (formato pedido por la guía).
