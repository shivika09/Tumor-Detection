# MRI Tumor Classification: PCA + SVM (complete project, with Member 4's part)

This folder contains the **whole team project in one place**: Member 1's data loader,
Member 2's PCA, Member 3's SVM, and Member 4's part (PCA reconstruction, evaluation,
confusion matrix, and prediction for a new MRI image).

> Educational project. It is **not** a medical diagnosis tool.

---

## 1. Quick start (5 minutes)

You need Python 3 installed. Open a terminal **inside this folder** (the one that contains `main.py`).

**Windows (PowerShell or Command Prompt)**
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
python predict.py --demo
python predict.py "C:\full\path\to\new_mri.jpg"
```

**Mac / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
python predict.py --demo
python predict.py /full/path/to/new_mri.jpg
```

(The virtual environment lines are optional. They only keep this project's packages separate. If PowerShell
refuses to run `activate`, skip those two lines and just run `pip install -r requirements.txt`.)

What each command does:

| Command | What happens | Time |
|---|---|---|
| `python main.py` | Runs the **whole** pipeline (Member 1 -> 2 -> 3 -> 4), trains everything, saves all results into `outputs/` | about 15 s |
| `python predict.py --demo` | Predicts a few images from the **hidden test set** and saves a picture | 2 s |
| `python predict.py image.jpg` | Predicts one new image. Add `--save-figure result.png` to also save a picture | 2 s |
| `python -m unittest discover -s tests -v` | Runs 15 automatic checks that everything fits together | 5 s |

Run `main.py` **first**. `predict.py` needs the models that `main.py` saves in `outputs/`.

---

## 2. Folder layout, and who owns what

```
mri_tumor_project/
|-- main.py                  Member 4 (+ glue)  runs everything in order
|-- predict.py               Member 4           predict ONE new image
|-- metrics.py               Member 4           confusion matrix, accuracy/precision/recall/F1, reconstruction error
|-- visualize.py             Member 4           all the pictures
|-- config.py                Member 4           shared file names/constants (tiny)
|-- tests/test_member4.py    Member 4           15 automatic checks
|
|-- tumor_project/
|   |-- data_loader.py       Member 1           load, grayscale, autocontrast, resize 64x64, normalize, flatten, split
|   `-- dataset/yes, no/     Member 1           253 MRI images (155 tumor, 98 no tumor)
|
|-- pca_scratch.py           Member 2           PCA from scratch: mean, covariance, eigenvectors
|-- pca_transform.py         Member 2           choose k, project, reconstruct, reconstruction error, run_pca()
|-- run_pca_on_data.py       Member 2           optional stand-alone PCA script (main.py does not need it)
|
|-- svm_model.py             Member 3           grid-search SVM, predict, decision scores
|-- svm_test.py              Member 3           optional: quick SVM check on the Iris toy dataset
|
|-- requirements.txt         packages to install
|-- requirements-tested.txt  exact versions I tested with
|-- expected_results/        what a correct run looks like (report + pictures) so you can compare
`-- outputs/                 CREATED by main.py (models, numbers, pictures). Safe to delete; main.py recreates it.
```

### The ONE change I made to a teammate's file

In `tumor_project/data_loader.py`, line 15 now reads `for name in sorted(os.listdir(path), key=str.lower):`
instead of `for name in os.listdir(path):`.

Why: `os.listdir` returns files in whatever order the operating system likes. Windows sorts them alphabetically;
Mac/Linux do not. The train/test split is made by *position*, so on a different computer **different images
would land in the test set** and everyone would get different accuracy numbers. Sorting makes the split identical
everywhere. **On Windows this changes nothing** (I checked that the sorted order reproduces Member 2's saved files exactly).
Every other teammate file is **byte-for-byte unchanged**. Tell Member 1 about this line.

---

## 3. How the four parts connect (the data flow)

```
 dataset/yes, dataset/no                                  (Member 1)
        |  get_split()
        v
 X_train (202 x 4096), X_test (51 x 4096), y_train, y_test      each image = 64x64 gray pixels, values 0..1
        |  run_pca(X_train, X_test)   - PCA fitted on TRAIN only (Member 2)
        v
 X_train_pca (202 x 109), X_test_pca (51 x 109)  + pca_model.npz  (mean, components 4096x109, k=109)
        |  optimize_svm(X_train_pca, y_train)     (Member 3)
        v
 trained SVM  --->  saved as outputs/svm_model.joblib (main.py does this; Member 3's code never saved it)
        |  predict_svm(model, X_test_pca)
        v
 y_pred  --->  metrics.evaluate(y_test, y_pred)      (Member 4)  -> confusion matrix, accuracy, precision, recall, F1
```

**Predicting a new image** (`predict.py`) walks the same road for one image:
`preprocess_image` (same steps as Member 1) -> `project` with the saved mean/components (Member 2)
-> `predict_svm` and `get_decision_scores` (Member 3).

Facts about the teammates' code that Member 4's code relies on (all verified, see section 6):

| Item | Actual value |
|---|---|
| Image size | `IMG_SIZE = 64` -> 4096 features per image |
| Preprocessing order | grayscale -> `autocontrast(cutoff=1)` -> resize 64x64 -> divide by 255 -> flatten |
| Labels | `1` = tumor (folder `yes`, the positive class), `0` = no tumor (folder `no`) |
| Split | `get_split(test_size=0.2, seed=42)` stratified: 202 train / 51 test |
| PCA | `fit_pca` -> `(mean, eigvals, components)`; `components` is 4096 x d; `project(X, mean, components, k)` |
| PCA file | `pca_model.npz` with keys `mean`, `components`, `k` (k = 109, keeps 95.12% variance) |
| SVM | scikit-learn `SVC`; grid search chose `kernel='rbf', C=10, gamma='scale'` |
| Model objects | `optimize_svm` **returns** the model; nothing saves it, so `main.py` saves it with joblib |

---

## 4. Member 4's part, in simple words (use this when presenting)

### A) PCA reconstruction (`metrics.py`, `visualize.py`)
PCA squeezes each image from 4096 numbers down to k = 109 numbers. **Reconstruction** goes backwards
(`Z @ components.T + mean`) to see what picture those 109 numbers describe. Fewer components means a blurrier picture
and a bigger error. We measure the error as **MSE** (average squared pixel difference).
- `reconstruction_error_curve` shows MSE for k = 1 ... 109, and `variance_kept_curve` shows the % of variance kept.
- Pictures: `reconstruction_test_images.png`, `reconstruction_train_images.png`, `reconstruction_error_vs_k.png`.
- **Worth explaining:** training images are rebuilt well (MSE 0.0020) but test images much worse (MSE 0.0105), and
  rebuilt test brains look like a "generic brain" where the specific tumor detail disappears. That is expected: PCA only knows
  patterns found in the training images.
- Member 2's `project`, `reconstruct` and `reconstruction_error` are **reused**, not rewritten.

### B) Evaluation (`metrics.py`)
Tumor = "positive". The confusion matrix has four boxes:

|  | predicted no tumor | predicted tumor |
|---|---|---|
| **really no tumor** | TN (correct) | FP = false alarm |
| **really tumor** | FN = **missed tumor** | TP (correct) |

- **Accuracy** = (TP+TN) / all. **Precision** = TP / (TP+FP): "when I say tumor, how often am I right?"
- **Recall** = TP / (TP+FN): "of the real tumors, how many did I find?" **F1** = balance of precision and recall.
- In medicine a **missed tumor (FN) is worse than a false alarm (FP)**, so recall deserves attention.
- All of this is written from scratch in NumPy and cross-checked against scikit-learn in the tests.

### C) Prediction (`predict.py`)
Preprocess -> project -> SVM. Output: the label, plus the **SVM decision score** (positive = tumor side of the boundary,
bigger distance = more confident; **not a probability**).

---

## 5. Results from my test run (also saved in `expected_results/`)

Test set = 51 images the model never saw while training:

| Metric | Value |
|---|---|
| Confusion matrix | TN 15, FP 5, FN 3, TP 28 |
| Accuracy | 0.8431 (43 of 51) |
| Precision / Recall / F1 | 0.8485 / 0.9032 / 0.8750 |
| Train accuracy | 1.0000 -> the model **overfits** (memorises the small training set) |

With only 51 test images, one image = about 2 percentage points, so don't over-read small differences.

---

## 6. What was tested and what was not

**Tested here (Linux, Python 3.12.3, numpy 2.4.4, scikit-learn 1.8.0, Pillow 12.1.1, matplotlib 3.10.8) using your real dataset and your teammates' real code:**
- Full pipeline `main.py` end to end, three times, no errors (about 14 s).
- PCA output regenerated by `main.py` is **identical** to Member 2's saved `X_train_pca.npy`, `X_test_pca.npy`, `y_*.npy` and `pca_model.npz` (k = 109).
- Member 3's `optimize_svm` on that data gives the same result (CV 0.8418, test accuracy 0.8431) as I got running Member 3's code directly.
- `metrics.py` against scikit-learn on 300 random cases and edge cases (nothing predicted positive, empty input, bad labels).
- `predict.preprocess_image` equals Member 1's `load_data` output **for all 253 images**; predict.py gives the same labels as the pipeline on all 51 test images.
- `predict.py` on RGB, RGBA, grayscale, palette, CMYK, 16-bit, 1x1-pixel and 3000x3000 images; and on a text file, a missing file, a folder, and before training (clear error messages).
- Running from a different working directory.
- The automated tests also **fail on purpose** if the old data loader is swapped back in (so they would warn you).
- Teammates' own scripts (`pca_scratch.py` self-test, `svm_test.py`, `run_pca_on_data.py`) still run in this layout.

**NOT tested (could not be, from here):**
- **Windows and Python 3.13** (your teammates' `.pyc` files show Python 3.13). I used only standard, portable code (no hard-coded slashes, ASCII-only output), but I have not run it on Windows.
- Other versions of numpy / scikit-learn than those listed in `requirements-tested.txt`.
- **Genuinely new MRI images from outside this dataset.** I only had the dataset, so predictions on external scans are untested, and nothing guarantees they will be accurate.
- I did not check whether the dataset labels are correct.

---

## 7. Known limits (be upfront about these in the presentation)

- **The model cannot recognise "not an MRI".** It always answers tumor / no tumor. I fed it 6 meaningless synthetic images:
  random noise, flat gray, a gradient, a checkerboard and a text drawing were all labelled **"tumor"**, and a picture of colored
  shapes got "no tumor" with a score of -0.05 (right on the boundary). I also tested whether PCA reconstruction error could flag
  such images; it cannot (random noise scored *lower* error than the median real test image), so I did not add such a warning.
  Only give it real brain MRI slices, and treat every answer as a classroom demo.
- **Overfitting:** train accuracy 100% vs test 84%. More data or regularisation would help.
- **Small test set** (51 images), so results are noisy.
- Images are squashed to 64x64 squares, which distorts non-square scans (done consistently for all images).
- The "k = 109" and "84%" numbers depend on the **new** data loader (with autocontrast). Do not mix with the older k = 98 files.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Missing '...pca_model.npz'. Run python main.py first` | Run `python main.py` before `predict.py` |
| `ModuleNotFoundError: No module named 'sklearn'` (or PIL, matplotlib...) | `pip install -r requirements.txt` (Pillow is imported as `PIL`) |
| `ModuleNotFoundError: tumor_project` | Run commands from inside this folder (the one containing `main.py`) |
| Warning about unpickling from a different scikit-learn version | Re-run `python main.py` to re-create `outputs/svm_model.joblib` with your version |
| `WARNING: predict.py's preprocessing does NOT match data_loader.py` | Someone changed `data_loader.py`. Copy the same change into `preprocess_image` in `predict.py`, then re-run `main.py` |
| Your accuracy differs from section 5 | Check the sorted-`listdir` line in `data_loader.py` is still there and `requirements-tested.txt` versions are installed |
| You changed the dataset or `data_loader.py` | Always re-run `python main.py` so PCA, SVM and `predict.py` stay in sync |
