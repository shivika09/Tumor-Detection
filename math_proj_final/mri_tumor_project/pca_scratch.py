import numpy as np
def mean_center(X):
    mean=X.mean(axis=0)
    return X-mean,mean

def covariance_matrix(Xc):
    n=Xc.shape[0]
    return (Xc.T @ Xc)/(n-1)  #dxd features matrix

def eigen_sorted(C):
    eigvals, eigvecs = np.linalg.eigh(C) #eigen vectors - important directions 
                                         #eigen values - variance around those important directions
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order] 


#"I'll combine the information from these pixels into a few important patterns." - these patterns are called PCs
def fit_pca(X_train, k=None): #k - used for dimensionality reduction, instead of 1000 components there will be 20-30 compononents
    Xc, mean = mean_center(X_train)
    C = covariance_matrix(Xc)
    eigvals, components = eigen_sorted(C)
    if k is not None:
        eigvals, components = eigvals[:k], components[:, :k]
    return mean, eigvals, components
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
 
    # 1. centering: every column mean should be ~0
    Xc, mean = mean_center(X)
    print("1. column means after centering ~0:", np.allclose(Xc.mean(axis=0), 0))
 
    # 2. covariance: should match NumPy's own
    C = covariance_matrix(Xc)
    print("2. covariance matches np.cov:      ", np.allclose(C, np.cov(X.T)))
 
    # 3. eigen equation: C v = lambda v
    eigvals, W = eigen_sorted(C)
    print("3. C v = lambda v holds:           ", np.allclose(C @ W[:, 0], eigvals[0] * W[:, 0]))
 
    # 4. sorted descending, unit-length columns
    print("4. sorted descending:              ", bool(np.all(np.diff(eigvals) <= 0)))
    print("   unit-length eigenvectors:       ", np.allclose(np.linalg.norm(W, axis=0), 1))
 
    # 5. compare with scikit-learn (components match up to sign)
    from sklearn.decomposition import PCA
    mean_, eigvals_, comps_ = fit_pca(X, k=3)
    sk = PCA(n_components=3).fit(X)
    same_vals = np.allclose(eigvals_, sk.explained_variance_)
    same_dirs = np.allclose(np.abs(np.sum(comps_.T * sk.components_, axis=1)), 1)
    print("5. matches sklearn eigenvalues:    ", same_vals)
    print("   matches sklearn components:     ", same_dirs)
 