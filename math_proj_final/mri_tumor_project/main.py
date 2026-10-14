"""
main.py  -  runs the WHOLE project, start to finish, using every member's real code.

    Step 1  Member 1  tumor_project/data_loader.py : get_split()    load images, 64x64 grayscale, split
    Step 2  Member 2  pca_transform.py             : run_pca()      PCA from scratch, keep 95% variance
    Step 3  Member 3  svm_model.py                 : optimize_svm() grid-search SVM, predict_svm()
    Step 4  Member 4  metrics.py / visualize.py    : metrics, confusion matrix, PCA reconstruction

Run:    python main.py
Output: everything is written to the  outputs/  folder (safe to delete; main.py recreates it).
Then:   python predict.py path/to/new_image.jpg
"""
import json
import os
import sys
import time

import joblib
import numpy as np

from config import ROOT, ARTIFACTS_DIR, SVM_MODEL_PATH
from tumor_project.data_loader import get_split, IMG_SIZE         # Member 1
from pca_transform import run_pca                                 # Member 2
from svm_model import optimize_svm, predict_svm                   # Member 3
import metrics                                                    # Member 4
import visualize                                                  # Member 4
from predict import test_set_files, preprocess_image              # Member 4


def out(name):
    return os.path.join(ARTIFACTS_DIR, name)


def main():
    t0 = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    report = []                      # lines that will also be saved to outputs/evaluation_report.txt

    def say(line=""):
        print(line)
        report.append(line)

    # ---------------- Step 1: data (Member 1) ----------------
    print("STEP 1/4  Loading images and splitting (Member 1)...")
    X_train, X_test, y_train, y_test = get_split()
    say(f"Data: train {X_train.shape}, test {X_test.shape}  "
        f"(each image = {IMG_SIZE}x{IMG_SIZE} = {IMG_SIZE * IMG_SIZE} numbers)")
    say(f"Tumor images: {int(y_train.sum())} of {len(y_train)} in train, {int(y_test.sum())} of {len(y_test)} in test")
    np.save(out("y_train.npy"), y_train)
    np.save(out("y_test.npy"), y_test)

    # ---------------- Step 2: PCA (Member 2) ----------------
    # run_pca() saves its files in the CURRENT folder, so we step into outputs/ while it runs.
    print("\nSTEP 2/4  PCA from scratch (Member 2)...")
    here = os.getcwd()
    os.chdir(ARTIFACTS_DIR)
    try:
        X_train_pca, X_test_pca, mean, components, k = run_pca(X_train, X_test)
    finally:
        os.chdir(here)
    say(f"PCA: {X_train.shape[1]} features -> k = {k} components")
    assert X_train_pca.shape == (len(y_train), k) and X_test_pca.shape == (len(y_test), k)

    # ---------------- Step 3: SVM (Member 3) ----------------
    print("\nSTEP 3/4  Training the SVM with grid search (Member 3)...")
    model = optimize_svm(X_train_pca, y_train)
    joblib.dump(model, SVM_MODEL_PATH)                 # Member 3's code never saved the model; we do it here
    y_pred_test = predict_svm(model, X_test_pca)
    y_pred_train = predict_svm(model, X_train_pca)
    params = {key: model.get_params()[key] for key in ("kernel", "C", "gamma") if key in model.get_params()}
    say(f"SVM chosen by grid search: {params}")

    # ---------------- Step 4: evaluation (Member 4) ----------------
    print("\nSTEP 4/4  Evaluation and PCA reconstruction (Member 4)...\n")
    test_m = metrics.evaluate(y_test, y_pred_test)
    train_m = metrics.evaluate(y_train, y_pred_train)
    say()
    say(metrics.format_report(test_m, "Test-set evaluation (the number that matters)"))
    say()
    gap = train_m["accuracy"] - test_m["accuracy"]
    say(f"For comparison, accuracy on the TRAIN set = {train_m['accuracy']:.4f} "
        f"(test = {test_m['accuracy']:.4f}).")
    if gap > 0.10:
        say("  -> Train accuracy is much higher than test accuracy: the model partly MEMORISES the training "
            "images instead of learning general patterns (overfitting). With only ~200 training images this is common.")
    else:
        say("  -> Train and test accuracy are close, so there is no strong sign of overfitting.")
    visualize.plot_confusion_matrix(test_m["confusion_matrix"], out("confusion_matrix.png"))

    # --- PCA reconstruction analysis (reusing Member 2's functions) ---
    ks = sorted({n for n in (1, 2, 5, 10, 20, 30, 50, 75, 100) if n <= k} | {k})
    train_mse = metrics.reconstruction_error_curve(X_train, mean, components, ks)
    test_mse = metrics.reconstruction_error_curve(X_test, mean, components, ks)
    var_kept = metrics.variance_kept_curve(X_train, X_train_pca, [n for n in ks])
    say()
    say("PCA reconstruction: error against number of components kept")
    say("   k   variance kept   train MSE   test MSE")
    for n, v, tr, te in zip(ks, var_kept, train_mse, test_mse):
        say(f"{n:4d}   {v:11.2%}   {tr:9.5f}   {te:8.5f}")
    say("MSE = average squared difference between original and reconstructed pixels (pixels are 0..1).")
    say("Test error is higher than train error because PCA was fitted ONLY on the train images - this is expected.")
    visualize.plot_reconstruction_curve(ks, train_mse, test_mse, k, out("reconstruction_error_vs_k.png"))

    # Same pictures for TEST images (PCA never saw them) and TRAIN images (PCA was fitted on them).
    grid_ks = sorted({n for n in (5, 20, 50) if n < k} | {k})
    pick_test = list(np.where(y_test == 1)[0][:2]) + list(np.where(y_test == 0)[0][:2])
    pick_train = list(np.where(y_train == 1)[0][:1]) + list(np.where(y_train == 0)[0][:1])
    visualize.plot_reconstruction_grid(X_test, mean, components, IMG_SIZE, grid_ks, pick_test,
                                       out("reconstruction_test_images.png"), labels=y_test)
    visualize.plot_reconstruction_grid(X_train, mean, components, IMG_SIZE, grid_ks, pick_train,
                                       out("reconstruction_train_images.png"), labels=y_train)
    say("Reconstruction pictures: train images are rebuilt well (PCA was fitted on them); test images come back as a")
    say("generic brain and the specific tumor detail is lost - PCA only keeps patterns common to the training set.")

    # --- list the held-out test images, and double-check the list matches X_test exactly ---
    test_paths, test_labels = test_set_files()
    same_labels = np.array_equal(test_labels, y_test)
    same_pixels = same_labels and all(
        np.allclose(preprocess_image(p), X_test[i]) for i, p in enumerate(test_paths))
    with open(out("test_images.txt"), "w", encoding="utf-8") as f:
        for p, lab in zip(test_paths, test_labels):
            f.write(f"{lab}\t{os.path.relpath(p, ROOT)}\n")      # 1 = tumor, 0 = no tumor; path relative to this folder
    if same_pixels:
        say("\nIntegrity check passed: predict.py's preprocessing reproduces the loader's test vectors exactly.")
    else:
        say("\nWARNING: predict.py's preprocessing does NOT match data_loader.py. "
            "Predictions on new images would be unreliable - see README 'Troubleshooting'.")

    # --- save the numbers and the report ---
    summary = {"k": k, "svm_params": {a: str(b) for a, b in params.items()},
               "test": {a: b for a, b in test_m.items() if a != "confusion_matrix"},
               "train_accuracy": train_m["accuracy"],
               "test_confusion_matrix": test_m["confusion_matrix"].tolist()}
    with open(out("metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out("evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nDone in {time.time() - t0:.0f} s. Files are in: {ARTIFACTS_DIR}")
    print("Try:  python predict.py --demo")
    return 0 if same_pixels else 1


if __name__ == "__main__":
    sys.exit(main())
