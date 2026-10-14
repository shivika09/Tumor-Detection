from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



def train_svm(X_train_pca, y_train, kernel='linear'):
    model = SVC(kernel=kernel)
    model.fit(X_train_pca, y_train)

    return model


def predict_svm(model, X_test_pca):
    return model.predict(X_test_pca)

param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1]
    }
]

def optimize_svm(X_train_pca, y_train):
    svm = SVC()

    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train_pca, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    return grid_search.best_estimator_

def get_support_vectors(model):
    return model.support_vectors_

def get_decision_scores(model, X_test_pca):
    return model.decision_function(X_test_pca)
    
    