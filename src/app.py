"""Clasifica fotos como perro o gato con el modelo guardado en models/.

Uso:
    python src/app.py foto1.jpg [foto2.jpg ...]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras  # noqa: E402

CLASSES = ["cat", "dog"]
IMG_SIZE = 200
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "efficientnet_b0_imagenet.keras"


def main(paths):
    if not paths:
        sys.exit(__doc__)
    model = keras.models.load_model(MODEL_PATH)
    for path in paths:
        img = keras.utils.img_to_array(keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE)))
        proba = model.predict(img[None, ...], verbose=0)[0]
        print(f"{path}\t{CLASSES[proba.argmax()]}\t{proba.max():.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
