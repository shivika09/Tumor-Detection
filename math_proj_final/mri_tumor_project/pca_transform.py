import numpy as np
from pca_scratch import fit_pca


def explained_variance_ratio(eigvals):
    eigvals = np.clip(eigvals, 0, None)
    return eigvals / eigvals.sum()


def choose_k(eigvals, target=0.95):
    cumulative = np.cumsum(explained_variance_ratio(eigvals))
    return int(np.searchsorted(cumulative, target) + 1)


def project(X, mean, components, k):
    return (X - mean) @ components[:, :k]


def reconstruct(Z, mean, components):
    k = Z.shape[1]
    return Z @ components[:, :k].T + mean


def reconstruction_error(X, mean, components, k):
    X_hat = reconstruct(project(X, mean, components, k), mean, components)
    return float(np.mean((X - X_hat) ** 2))


def plot_variance(eigvals, k=None, path="variance_curve.png"):
    import matplotlib.pyplot as plt
    cumulative = np.cumsum(explained_variance_ratio(eigvals))
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.axhline(0.95, ls="--", c="gray", label="95%")
    if k is not None:
        plt.axvline(k, ls=":", c="red", label=f"k = {k}")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_reconstruction(X, mean, components, img_size, ks=(10, 50, 100), index=0,
                        path="reconstruction.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3))
    axes[0].imshow(X[index].reshape(img_size, img_size), cmap="gray")
    axes[0].set_title("Original")
    for ax, k in zip(axes[1:], ks):
        Z = project(X[index:index + 1], mean, components, k)
        ax.imshow(reconstruct(Z, mean, components).reshape(img_size, img_size), cmap="gray")
        ax.set_title(f"k = {k}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_pca(X_train, X_test, target=0.95, k=None):
    mean, eigvals, components = fit_pca(X_train)

    if k is None:
        k = choose_k(eigvals, target)
    kept = explained_variance_ratio(eigvals)[:k].sum()
    print(f"k = {k} components keep {kept:.2%} of the variance "
          f"({X_train.shape[1]} -> {k} features)")

    X_train_pca = project(X_train, mean, components, k)
    X_test_pca = project(X_test, mean, components, k)

    print("Reconstruction MSE (train):", reconstruction_error(X_train, mean, components, k))
    print("Reconstruction MSE (test): ", reconstruction_error(X_test, mean, components, k))

    np.save("X_train_pca.npy", X_train_pca)
    np.save("X_test_pca.npy", X_test_pca)
    np.savez("pca_model.npz", mean=mean, components=components[:, :k], k=k)
    plot_variance(eigvals, k)
    return X_train_pca, X_test_pca, mean, components, k


if __name__ == "__main__":
    try:
        X_train, X_test = np.load("X_train.npy"), np.load("X_test.npy")
        print("Loaded real data:", X_train.shape, X_test.shape)
    except FileNotFoundError:
        print("X_train.npy not found, using random demo data")
        rng = np.random.default_rng(0)
        mixing = rng.normal(size=(5, 30))
        make = lambda n: rng.normal(size=(n, 5)) @ mixing + 0.1 * rng.normal(size=(n, 30))
        X_train, X_test = make(100), make(30)

    X_train_pca, X_test_pca, mean, components, k = run_pca(X_train, X_test)
    print("X_train_pca:", X_train_pca.shape, "| X_test_pca:", X_test_pca.shape)