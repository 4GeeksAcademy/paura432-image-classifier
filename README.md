# Clasificador de imágenes: perros y gatos

Proyecto del módulo «Modelos de clasificación» de 4Geeks (*Proyecto Tutorial de Clasificador de Imágenes*), creado a partir del template [`machine-learning-python-template`](https://github.com/4GeeksAcademy/machine-learning-python-template).

Una red EfficientNet-B0 clasifica como perro o gato las 25.000 fotos del dataset Asirra. El análisis, las decisiones y los resultados están en el notebook; este README explica qué hay en el repositorio y cómo ejecutarlo.

## Resultados

| Modelo | Accuracy en test (3.750 fotos) |
| --- | --- |
| EfficientNet-B0 desde cero, arquitectura del enunciado (paso 3) | 72,8 % |
| EfficientNet-B0 con pesos de ImageNet, `ModelCheckpoint` + `EarlyStopping` (paso 4) | 99,2 % |

## Contenido

| Ruta | Qué es |
| --- | --- |
| `src/explore.ipynb` | Notebook ejecutado con los pasos 1 a 5 del enunciado: carga, visualización, red base, optimización con callbacks, evaluación en test y guardado |
| `src/app.py` | Clasifica fotos sueltas con el modelo guardado |
| `models/efficientnet_b0_imagenet.keras` | Mejor modelo, guardado por `ModelCheckpoint` (formato nativo de Keras 3, 18 MB) |
| `requirements.txt` | Dependencias del template más `tensorflow`, `keras` y `pillow`, todas con versión fija |

El resto (`.devcontainer/`, `src/utils.py`, `.env.example`, `data/processed/`) viene del template y este proyecto no lo usa.

## Datos

Las fotos no están en el repositorio: el zip ocupa 580 MB. Se descargan del enlace del enunciado y se descomprimen en `data/raw/`:

```sh
curl -L -o /tmp/dogs-vs-cats.zip https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip
unzip -q /tmp/dogs-vs-cats.zip -d data/raw/ -x "__MACOSX/*"
```

Queda `data/raw/dogs-vs-cats/train/` con 25.000 `.jpg` (12.500 `cat.*` y 12.500 `dog.*`). La etiqueta sale del nombre del archivo.

La primera ejecución del notebook guarda en `data/interim/dogs_vs_cats_200.npz` las fotos ya redimensionadas a 200x200 junto con sus etiquetas (unos 3 GB). Las ejecuciones siguientes las leen de ahí. Ese archivo tampoco se sube al repositorio.

## Ejecución

En Codespaces el entorno instala `requirements.txt` al crearse. En local hace falta Python 3.12 o 3.13:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para clasificar fotos con el modelo guardado:

```sh
python src/app.py data/raw/dogs-vs-cats/train/cat.1.jpg data/raw/dogs-vs-cats/train/dog.1.jpg
```

Para volver a ejecutar el notebook entero, con sus salidas:

```sh
cd src
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 explore.ipynb
```

El notebook carga las 25.000 fotos en memoria, siguiendo la regla del enunciado para equipos con más de 12 GB de RAM, y comprueba la RAM disponible antes de hacerlo. En una CPU de 16 hilos sin GPU tarda unos 40 minutos: 16 para el modelo desde cero, 17 para el optimizado y el resto en carga y evaluación. La primera vez descarga de Keras los pesos de ImageNet de EfficientNet-B0 (16 MB).

## Diferencias con el enunciado

- **Tamaño de entrada.** El paso 2 pide imágenes de 200x200 y la arquitectura del paso 3 declara `input_shape = (224, 224, 3)`. Usamos 200x200 en los dos sitios; EfficientNet-B0 sin cabeza de clasificación acepta ese tamaño.
- **`ImageDataGenerator` y `fit_generator`.** Ya no existen en Keras 3.15. Como la máquina tiene más de 12 GB de RAM, las imágenes se cargan con `keras.utils.load_img`; `trdata` y `tsdata` son `tf.data.Dataset` y los callbacks se pasan a `fit`.
- **Conjunto de test.** El zip solo trae etiquetas para `train`, así que el test (15 %) y la validación (15 %) salen de esa carpeta con una división estratificada y semilla fija.
- **Optimización.** Además de los callbacks del paso 4, el modelo optimizado arranca con los pesos de ImageNet en lugar de `weights = None` y lleva un aumento de datos ligero (volteo horizontal y rotación pequeña). El notebook explica el porqué y compara los dos modelos en el mismo test.
