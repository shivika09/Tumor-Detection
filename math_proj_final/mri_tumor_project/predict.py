"""
predict.py  (Member 4)

Predict "tumor" / "no tumor" for ONE new MRI image, using the models built by
the rest of the team:

    new image
      -> preprocess_image()            same steps as Member 1's data_loader.py
      -> project()                     Member 2's PCA (pca_transform.py), using the saved pca_model.npz
      -> model.predict()               Member 3's SVM (svm_model.py), saved by main.py
      -> label + decision score

Usage (run main.py once first so the models exist):
    python predict.py path/to/new_mri.jpg
    python predict.py path/to/new_mri.jpg --save-figure result.png
    python predict.py --demo            # predict a few images from the hidden TEST set
    python predict.py --demo 8
"""
import argparse
import os
import sys

import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from tumor_project.data_loader import IMG_SIZE, DATA_DIR            # Member 1
from pca_transform import project, reconstruct                      # Member 2
from svm_model import predict_svm, get_decision_scores             # Member 3
from config import (ARTIFACTS_DIR, PCA_MODEL_PATH, SVM_MODEL_PATH,
                    SPLIT_TEST_SIZE, SPLIT_SEED, CLASS_NAMES)

DISCLAIMER = "(Educational project - NOT a medical diagnosis.)"


# ----------------------------------------------------------------------
# 1) Preprocessing ONE image  (mirror of tumor_project/data_loader.py)
# ----------------------------------------------------------------------
def preprocess_image(path):
    """
    Turn one image file into the 4096-number vector the model expects.

    Member 1's load_data() only works on whole folders, so this repeats its
    per-image steps IN THE SAME ORDER:
        grayscale -> autocontrast(cutoff=1) -> resize to 64x64 -> /255 -> flatten
    tests/test_member4.py proves it gives identical numbers to load_data()
    for every image in the dataset. If Member 1 ever changes data_loader.py,
    change this function the same way and re-run main.py.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        img = Image.open(path).convert("L")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"'{path}' could not be opened as an image ({e}).") from e
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float64).flatten() / 255.0


# ----------------------------------------------------------------------
# 2) Loading the saved models and checking they fit together
# ----------------------------------------------------------------------
def load_models(pca_path=PCA_MODEL_PATH, svm_path=SVM_MODEL_PATH):
    """Load Member 2's PCA (mean, components, k) and Member 3's trained SVM, and check the shapes agree."""
    for p in (pca_path, svm_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing '{p}'. Run  python main.py  first to train and save the models.")
    pca = np.load(pca_path)
    mean, components, k = pca["mean"], pca["components"], int(pca["k"])
    svm = joblib.load(svm_path)

    n_pixels = IMG_SIZE * IMG_SIZE
    if mean.shape != (n_pixels,) or components.shape != (n_pixels, k):
        raise ValueError(f"PCA file has unexpected shapes: mean {mean.shape}, components {components.shape}; "
                         f"expected ({n_pixels},) and ({n_pixels}, {k}). Re-run main.py.")
    if svm.n_features_in_ != k:
        raise ValueError(f"The SVM expects {svm.n_features_in_} features but PCA gives {k}. "
                         f"They come from different runs - re-run main.py.")
    return mean, components, k, svm


# ----------------------------------------------------------------------
# 3) Predicting
# ----------------------------------------------------------------------
def predict_vector(x, mean, components, k, svm):
    """x: one preprocessed image, shape (4096,). Returns a dict with the result."""
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    z = project(x, mean, components, k)                    # (1, 4096) -> (1, k)       Member 2
    x_hat = reconstruct(z, mean, components)               # back to (1, 4096)         Member 2
    label = int(predict_svm(svm, z)[0])                    #                           Member 3
    score = float(get_decision_scores(svm, z)[0])          # + = tumor side            Member 3
    return {
        "label": label,
        "label_name": CLASS_NAMES[label],
        "score": score,
        "reconstruction_error": float(np.mean((x - x_hat) ** 2)),
        "vector": x[0],
        "reconstruction": x_hat[0],
    }


def predict_image(path, mean, components, k, svm):
    """Image file -> prediction dict."""
    result = predict_vector(preprocess_image(path), mean, components, k, svm)
    result["path"] = path
    return result


# ----------------------------------------------------------------------
# 4) Which files are in the hidden TEST set (for the demo)
# ----------------------------------------------------------------------
def dataset_files():
    """
    List (paths, labels) of dataset images in EXACTLY the order data_loader.load_data()
    uses (yes first, then no; sorted names; unreadable files skipped).
    """
    paths, labels = [], []
    for label, folder in [(1, "yes"), (0, "no")]:
        folder_path = os.path.join(DATA_DIR, folder)
        for name in sorted(os.listdir(folder_path), key=str.lower):
            p = os.path.join(folder_path, name)
            try:
                Image.open(p).convert("L")
            except Exception:
                continue
            paths.append(p)
            labels.append(label)
    return paths, np.array(labels)


def test_set_files():
    """(paths, labels) of the images get_split() puts in the TEST set (never seen by PCA or the SVM)."""
    paths, labels = dataset_files()
    idx = np.arange(len(labels))
    _, test_idx = train_test_split(idx, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=labels)
    return [paths[i] for i in test_idx], labels[test_idx]


# ----------------------------------------------------------------------
# 5) Command line
# ----------------------------------------------------------------------
def _describe(result):
    side = "tumor side (+)" if result["score"] > 0 else "no-tumor side (-)"
    return [
        f"Prediction            : {result['label_name'].upper()}   (label {result['label']})",
        f"SVM decision score    : {result['score']:+.3f}   -> {side} of the decision boundary.",
        "                        Bigger |score| = further from the boundary = more confident. It is NOT a probability.",
        f"PCA reconstruction MSE: {result['reconstruction_error']:.4f}   (information only; it cannot tell whether the image is really an MRI)",
    ]


def run_single(path, save_figure=None):
    mean, components, k, svm = load_models()
    result = predict_image(path, mean, components, k, svm)
    print(f"Image: {path}")
    print("\n".join(_describe(result)))
    print(DISCLAIMER)
    if save_figure:
        from visualize import plot_prediction
        plot_prediction(result["vector"], result["reconstruction"], IMG_SIZE,
                        f"{result['label_name'].upper()}  (score {result['score']:+.2f})", save_figure)
        print(f"Figure saved to {save_figure}")
    return result


def run_demo(n=6, save_path=None):
    """Predict n images from the TEST set (half tumor / half no tumor when possible)."""
    mean, components, k, svm = load_models()
    paths, labels = test_set_files()
    rng = np.random.default_rng(0)
    tumor = [i for i, l in enumerate(labels) if l == 1]
    healthy = [i for i, l in enumerate(labels) if l == 0]
    n_t = min(len(tumor), (n + 1) // 2)
    n_h = min(len(healthy), n - n_t)
    chosen = list(rng.choice(tumor, n_t, replace=False)) + list(rng.choice(healthy, n_h, replace=False))

    rows, correct = [], 0
    print(f"Demo on {len(chosen)} images from the held-out TEST set:\n")
    for i in chosen:
        r = predict_image(paths[i], mean, components, k, svm)
        ok = r["label"] == int(labels[i])
        correct += ok
        print(f"  {os.path.basename(paths[i]):<16} true: {CLASS_NAMES[int(labels[i])]:<9} "
              f"predicted: {r['label_name']:<9} score {r['score']:+.2f}   {'correct' if ok else 'WRONG'}")
        rows.append({"vector": r["vector"], "true_label": int(labels[i]),
                     "pred_label": r["label"], "score": r["score"]})
    print(f"\n{correct} of {len(chosen)} correct. {DISCLAIMER}")
    print("Note: a handful of images varies a lot by luck. For the overall result on all test images see outputs/evaluation_report.txt.")
    if save_path is None:
        save_path = os.path.join(ARTIFACTS_DIR, "demo_predictions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from visualize import plot_demo_predictions
    plot_demo_predictions(rows, IMG_SIZE, save_path)
    print(f"Figure saved to {save_path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict tumor / no tumor for a new MRI image.")
    parser.add_argument("image", nargs="?", help="path to a new MRI image (jpg, png, ...)")
    parser.add_argument("--save-figure", metavar="PNG", help="also save a picture of the result")
    parser.add_argument("--demo", nargs="?", const=6, type=int, metavar="N",
                        help="predict N images from the held-out test set instead (default 6)")
    args = parser.parse_args(argv)
    try:
        if args.demo is not None:
            run_demo(args.demo)
        elif args.image:
            run_single(args.image, args.save_figure)
        else:
            parser.print_help()
            return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
