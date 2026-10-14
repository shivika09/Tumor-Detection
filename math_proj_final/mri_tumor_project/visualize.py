"""
visualize.py  (Member 4)

All the pictures for Member 4's part. Every function SAVES a PNG file
(no pop-up windows), so it works on any computer, with or without a screen.

Reuses Member 2's project() and reconstruct() for the reconstruction pictures.
"""
import matplotlib
matplotlib.use("Agg")            # draw to files only - avoids GUI problems on any OS
import matplotlib.pyplot as plt
import numpy as np

from pca_transform import project, reconstruct   # Member 2
from config import CLASS_NAMES


def plot_confusion_matrix(cm, path, title="Confusion matrix (test set)"):
    """Colored 2x2 grid; each cell shows its name (TN/FP/FN/TP) and count."""
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"predicted\n{CLASS_NAMES[0]}", f"predicted\n{CLASS_NAMES[1]}"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"true\n{CLASS_NAMES[0]}", f"true\n{CLASS_NAMES[1]}"])
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_names[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > threshold else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstruction_grid(X, mean, components, img_size, ks, indices, path, labels=None):
    """
    Rows = chosen images, columns = Original, then reconstruction with k components for each k.
    More components -> sharper picture. (Member 2's plot_reconstruction shows ONE image; this shows several.)
    """
    n_rows, n_cols = len(indices), len(ks) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.3 * n_rows), squeeze=False)
    for r, idx in enumerate(indices):
        axes[r][0].imshow(X[idx].reshape(img_size, img_size), cmap="gray", vmin=0, vmax=1)
        title = "Original"
        if labels is not None:
            title += f"\n({CLASS_NAMES[int(labels[idx])]})"
        axes[r][0].set_title(title, fontsize=9)
        for c, k in enumerate(ks, start=1):
            Z = project(X[idx:idx + 1], mean, components, k)
            img = reconstruct(Z, mean, components).reshape(img_size, img_size)
            axes[r][c].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r][c].set_title(f"k = {k}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstruction_curve(ks, train_mse, test_mse, chosen_k, path):
    """Reconstruction error (MSE) against the number of components kept."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, train_mse, marker="o", label="train images (used to fit PCA)")
    ax.plot(ks, test_mse, marker="s", label="test images (never seen by PCA)")
    ax.axvline(chosen_k, ls=":", c="red", label=f"chosen k = {chosen_k} (95% variance)")
    ax.set_xlabel("Number of components kept (k)")
    ax.set_ylabel("Reconstruction error (mean squared error)")
    ax.set_title("More components -> smaller reconstruction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction(original_vec, reconstructed_vec, img_size, headline, path):
    """One new image: what the model saw (64x64), its PCA reconstruction, and the verdict."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))
    axes[0].imshow(original_vec.reshape(img_size, img_size), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image as the model sees it\n(grayscale, contrast-stretched, 64x64)", fontsize=8)
    axes[1].imshow(np.clip(reconstructed_vec.reshape(img_size, img_size), 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("PCA reconstruction", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(headline, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_demo_predictions(rows, img_size, path):
    """
    rows: list of dicts with keys: vector, true_label, pred_label, score.
    Draws one small picture per image, titled with true/predicted and green (correct) or red (wrong).
    """
    n = len(rows)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        ax.imshow(row["vector"].reshape(img_size, img_size), cmap="gray", vmin=0, vmax=1)
        ok = row["true_label"] == row["pred_label"]
        ax.set_title(f"true: {CLASS_NAMES[row['true_label']]}\npredicted: {CLASS_NAMES[row['pred_label']]}\n"
                     f"score {row['score']:+.2f}", fontsize=9, color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
