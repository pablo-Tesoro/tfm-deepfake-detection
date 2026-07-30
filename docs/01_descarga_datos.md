# Descarga de datos — FaceForensics++

> Guía operativa para descargar un subconjunto **manejable** del dataset una vez
> recibido el acceso. No reimplementamos la descarga: FF++ proporciona su propio
> script oficial. Aquí se documenta cómo usarlo para nuestro caso.

## 1. Obtener el acceso

1. Rellena el formulario de Google enlazado en https://github.com/ondyari/FaceForensics
2. Una vez aceptado, recibirás por correo el script `download-FaceForensics.py`
   (o un enlace a él) junto con el enlace de descarga.

## 2. Descarga automática (recomendado)

El orquestador se encarga de todo: descarga **solo las cinco categorías** que usa
el proyecto y acepta por ti el aviso de términos de uso. Solo tienes que colocar
`download-FaceForensics.py` en la raíz del repositorio.

```python
from run_all import run_pipeline

run_pipeline(download=True, n_videos=150)   # 150 vídeos por categoría
run_pipeline(download=True, n_videos=None)  # TODO el dataset
```

Es incremental: los vídeos ya descargados se omiten, así que puedes ampliar el
subconjunto en cualquier momento sin volver a bajar lo que ya tienes.

## 3. Descarga manual (alternativa)

Importante: **no uses `-d all`**, porque baja 8 categorías y 3 de ellas no las usa
este proyecto (`DeepFakeDetection_original`/actors, `DeepFakeDetection` y
`FaceShifter`). Descarga solo las cinco necesarias llamando al script una vez por
categoría:

```bash
# original (reales) + los 4 métodos. -n = nº de vídeos por categoría.
# Omite -n por completo para descargar el dataset entero.
for D in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  python download-FaceForensics.py ./data/raw -d $D -c c23 -t videos -n 100 --server EU2
done
```

> El script pide aceptar los términos de uso (pulsar una tecla); si lo lanzas a
> mano, pulsa Intro. Verifica los parámetros con `--help`.

## 4. Splits oficiales

Definen la partición train/val/test (720/140/140 vídeos) y hacen tus resultados
comparables con la literatura. **`run_all.py` los descarga automáticamente** si
faltan; no tienes que hacer nada.

Si los quieres bajar a mano, están en el repositorio público de FaceForensics
(`dataset/splits/`) y van a `data/raw/splits/`:

```python
import urllib.request
from pathlib import Path

d = Path("data/raw/splits"); d.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
for n in ["train.json", "val.json", "test.json"]:
    urllib.request.urlretrieve(f"{base}/{n}", d / n)
```

> Con subconjuntos pequeños el split oficial puede dejar algún conjunto vacío. En
> ese caso el código lo detecta y aplica un reparto **estratificado** equivalente,
> avisando por pantalla de cuál ha usado.

## 5. Estructura resultante esperada

```
data/raw/
├── original_sequences/youtube/c23/videos/*.mp4
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/*.mp4
│   ├── Face2Face/c23/videos/*.mp4
│   ├── FaceSwap/c23/videos/*.mp4
│   └── NeuralTextures/c23/videos/*.mp4
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

Cuando tengas esto, el inventario se construye con:

```python
from src.data.dataset import enumerate_videos, load_official_splits, assign_splits
df = enumerate_videos("data/raw", compression="c23")
splits = load_official_splits("data/raw/splits")
df["split"] = assign_splits(df, splits)
```

## 6. Consejos prácticos

- **En Colab (recomendado):** usa `notebooks/RUN_ALL.ipynb`, que monta Drive, fija
  `TFM_WORKSPACE` y deja los datos en tu Drive, persistentes entre sesiones.
- **Empieza pequeño:** valida el pipeline con `n_videos=20` antes de bajar el
  dataset completo.
- **Espacio en disco:** el dataset completo en c23 ocupa decenas de GB; la
  compresión `raw` es inviable para un TFM.
- **No subas los vídeos al repositorio** (ya está cubierto por el `.gitignore`, y
  además la licencia de FF++ no permite redistribuirlos).
