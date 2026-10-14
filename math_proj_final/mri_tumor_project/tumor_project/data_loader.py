import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

IMG_SIZE = 64            # every image becomes 64x64 = 4096 features
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")   # folder that contains the 'yes' and 'no' folders


def load_data():
    """Load images, convert to grayscale, resize, normalize, flatten, label."""
    X, y = [], []
    for label, folder in [(1, "yes"), (0, "no")]:   # 1 = tumor, 0 = no tumor
        path = os.path.join(DATA_DIR, folder)
        for name in sorted(os.listdir(path), key=str.lower):   # sorted = same split on Windows/Mac/Linux (Member 4 fix)
            try:
                img = Image.open(os.path.join(path, name)).convert("L")  # grayscale
                img = ImageOps.autocontrast(img, cutoff=1)               # stretch contrast (NEW)
            except Exception:
                continue  # skip anything that isn't an image
            img = img.resize((IMG_SIZE, IMG_SIZE))                       # resize
            vec = np.asarray(img, dtype=np.float64).flatten() / 255.0    # flatten + normalize
            X.append(vec)
            y.append(label)
    return np.array(X), np.array(y)


def get_split(test_size=0.2, seed=42):
    """Train/test split. Do this BEFORE computing PCA/eigenvectors."""
    X, y = load_data()
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Tumor in train:", y_train.sum(), "of", len(y_train))