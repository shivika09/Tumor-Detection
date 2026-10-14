# MRI Tumor Screener

Browser page that classifies brain MRI slices as **tumor / no tumor** with the team's PCA + SVM model
(grayscale -> autocontrast -> 64x64 -> PCA 4096->109 -> RBF SVM, C=10). Everything runs in the browser; images are not uploaded.
Educational project, not a medical tool.

## Run it in VS Code
1. Install the **Live Server** extension (Ritwick Dey).
2. Open this folder in VS Code (File > Open Folder).
3. Right-click `index.html` > **Open with Live Server**.

Opening `index.html` directly in Chrome, Edge or Firefox also works (no server needed).
Fonts load from Google Fonts; offline, system fonts are used instead.

No Live Server? In a terminal inside this folder: `python -m http.server 5500`, then open http://localhost:5500

## Files
| File | Purpose |
|---|---|
| `index.html` | Page markup |
| `css/style.css` | Styles (light and dark theme follow the OS) |
| `js/core.js` | Preprocessing (same as `data_loader.py`) + PCA projection + RBF SVM decision |
| `js/app.js` | Upload, drag and drop, cards, summary |
| `js/model-data.js` | Trained model (generated) |
| `js/samples-data.js` | 8 held-out sample scans (generated) |
| `training/train_export.py` | Retrains with the project code and regenerates `model-data.js` |

## Retrain
`python training/train_export.py /path/to/mri_tumor_project` (needs numpy, scikit-learn, Pillow).
If you change the dataset or `data_loader.py`, retrain and also mirror the change in `js/core.js`.
