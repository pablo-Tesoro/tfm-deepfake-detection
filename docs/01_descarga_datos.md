# Descarga de datos — FaceForensics++

> Guía operativa para descargar un subconjunto **manejable** del dataset una vez
> recibido el acceso. No reimplementamos la descarga: FF++ proporciona su propio
> script oficial. Aquí se documenta cómo usarlo para nuestro caso.

## 1. Obtener el acceso

1. Rellena el formulario de Google enlazado en https://github.com/ondyari/FaceForensics
2. Una vez aceptado, recibirás por correo el script `download-FaceForensics.py`
   (o un enlace a él) junto con el enlace de descarga.

## 2. Descargar un subconjunto en c23 (recomendado)

El dataset completo es enorme. Para un TFM con recursos limitados, descarga un
**subconjunto** de los vídeos en compresión **c23** (calidad alta pero manejable).

Importante: **no uses `-d all`**, porque baja 8 categorías y 3 de ellas no las usa
este proyecto (`DeepFakeDetection_original`/actors, `DeepFakeDetection` y
`FaceShifter`). Descarga solo las cinco que usamos llamando al script una vez por
categoría:

```bash
# original (reales) + los 4 métodos. -n = nº de vídeos por categoría (subconjunto).
for D in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  python download-FaceForensics.py ./data/raw -d $D -c c23 -t videos -n 100 --server EU2
done
```

> El script pide aceptar los términos de uso (pulsar una tecla). En el orquestador
> (`run_all.py`) esto se resuelve automáticamente; si lo lanzas a mano, pulsa Intro.
> Verifica los parámetros con `python download-FaceForensics.py --help`.

## 3. Descargar los splits oficiales

Para que tus resultados sean comparables con la literatura, usa la partición
oficial train/val/test. Los ficheros `train.json`, `val.json` y `test.json`
están en el repositorio oficial, en `dataset/splits/`. Descárgalos y colócalos
en una carpeta accesible (p. ej. `data/raw/splits/`).

## 4. Estructura resultante esperada

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

## 5. Consejos prácticos

- **En Colab (recomendado):** usa `notebooks/00_setup_colab.ipynb`, que monta Drive,
  fija `TFM_WORKSPACE` y descarga directamente a tu Drive. Así los datos persisten
  entre sesiones aunque Colab reinicie.
- **Empieza pequeño**: valida todo el pipeline con `-n 20` antes de bajar más vídeos.
- **Espacio en disco**: c23 con ~100 vídeos por categoría es asumible; raw no lo es.
- **No subas los vídeos al repositorio** (ya está cubierto por el `.gitignore`).
