import numpy as np
from tumor_project.data_loader import get_split, IMG_SIZE
from pca_transform import run_pca, plot_reconstruction

X_train, X_test, y_train, y_test = get_split()

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

X_train_pca, X_test_pca, mean, components, k = run_pca(X_train, X_test)
plot_reconstruction(X_train, mean, components, IMG_SIZE, ks=(5, k, 100))

np.savez(
    "pca_data.npz",
    X_train_pca=X_train_pca,
    X_test_pca=X_test_pca,
    y_train=y_train,
    y_test=y_test
)

print("PCA data saved successfully!")