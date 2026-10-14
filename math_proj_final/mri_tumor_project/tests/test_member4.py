"""
Automated checks for Member 4's part and for how it plugs into Members 1-3.

Run from the project folder:
    python -m unittest discover -s tests -v

  * Tests in class TestWithoutTraining need only the dataset.
  * Tests in class TestAfterTraining need  outputs/  (run  python main.py  first);
    if it is missing they are SKIPPED, not failed.
"""
import inspect
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config                                                    # noqa: E402
import metrics                                                   # noqa: E402
import predict                                                   # noqa: E402
from tumor_project.data_loader import load_data, get_split, IMG_SIZE   # noqa: E402
from pca_transform import project, reconstruct, reconstruction_error   # noqa: E402

HAVE_OUTPUTS = os.path.isfile(config.PCA_MODEL_PATH) and os.path.isfile(config.SVM_MODEL_PATH)


class TestMetrics(unittest.TestCase):
    """metrics.py must agree with scikit-learn (used here ONLY as a referee)."""

    def test_matches_sklearn_on_random_cases(self):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                     confusion_matrix as sk_cm)
        rng = np.random.default_rng(123)
        for _ in range(200):
            n = int(rng.integers(5, 80))
            yt, yp = rng.integers(0, 2, n), rng.integers(0, 2, n)
            m = metrics.evaluate(yt, yp)
            self.assertTrue(np.array_equal(m["confusion_matrix"], sk_cm(yt, yp, labels=[0, 1])))
            self.assertAlmostEqual(m["accuracy"], accuracy_score(yt, yp))
            self.assertAlmostEqual(m["precision"], precision_score(yt, yp, zero_division=0))
            self.assertAlmostEqual(m["recall"], recall_score(yt, yp, zero_division=0))
            self.assertAlmostEqual(m["f1"], f1_score(yt, yp, zero_division=0))

    def test_known_example(self):
        # 3 TP, 4 TN, 1 FP, 2 FN
        yt = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        yp = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        m = metrics.evaluate(yt, yp)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (4, 1, 2, 3))
        self.assertAlmostEqual(m["accuracy"], 0.7)
        self.assertAlmostEqual(m["precision"], 3 / 4)
        self.assertAlmostEqual(m["recall"], 3 / 5)

    def test_no_crash_when_nothing_predicted_positive(self):
        m = metrics.evaluate([1, 0, 1, 0], [0, 0, 0, 0])
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["f1"], 0.0)

    def test_bad_input_gives_clear_error(self):
        with self.assertRaises(ValueError):
            metrics.evaluate([0, 1], [0])
        with self.assertRaises(ValueError):
            metrics.evaluate([0, 2], [0, 1])
        with self.assertRaises(ValueError):
            metrics.evaluate([], [])


class TestWithoutTraining(unittest.TestCase):
    """Compatibility with Member 1's loader (needs only the dataset)."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_data()

    def test_data_format(self):
        self.assertEqual(self.X.shape[1], IMG_SIZE * IMG_SIZE)
        self.assertEqual(set(np.unique(self.y)), {config.LABEL_NO_TUMOR, config.LABEL_TUMOR})
        self.assertGreaterEqual(self.X.min(), 0.0)
        self.assertLessEqual(self.X.max(), 1.0)

    def test_config_split_matches_get_split_defaults(self):
        sig = inspect.signature(get_split).parameters
        self.assertEqual(sig["test_size"].default, config.SPLIT_TEST_SIZE)
        self.assertEqual(sig["seed"].default, config.SPLIT_SEED)

    def test_preprocess_image_equals_loader_for_every_image(self):
        paths, labels = predict.dataset_files()
        self.assertEqual(len(paths), len(self.X))
        self.assertTrue(np.array_equal(labels, self.y))
        for p, row in zip(paths, self.X):
            np.testing.assert_allclose(predict.preprocess_image(p), row, atol=1e-12, err_msg=p)

    def test_test_file_list_matches_get_split(self):
        _, X_test, _, y_test = get_split()
        paths, labels = predict.test_set_files()
        self.assertTrue(np.array_equal(labels, y_test))
        for p, row in zip(paths, X_test):
            np.testing.assert_allclose(predict.preprocess_image(p), row, atol=1e-12, err_msg=p)

    def test_preprocess_errors(self):
        with self.assertRaises(FileNotFoundError):
            predict.preprocess_image(os.path.join(ROOT, "no_such_file.jpg"))
        with self.assertRaises(ValueError):
            predict.preprocess_image(os.path.join(ROOT, "README.md"))   # exists, but is not an image


@unittest.skipUnless(HAVE_OUTPUTS, "outputs/ not found - run  python main.py  first")
class TestAfterTraining(unittest.TestCase):
    """Compatibility with Members 2 and 3's trained objects (needs outputs/)."""

    @classmethod
    def setUpClass(cls):
        cls.mean, cls.components, cls.k, cls.svm = predict.load_models()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_split()

    def test_shapes_fit_together(self):
        self.assertEqual(self.mean.shape, (IMG_SIZE * IMG_SIZE,))
        self.assertEqual(self.components.shape, (IMG_SIZE * IMG_SIZE, self.k))
        self.assertEqual(self.svm.n_features_in_, self.k)
        self.assertEqual(list(self.svm.classes_), [0, 1])   # so a positive decision score means "tumor"

    def test_saved_pca_features_match_projection_of_current_data(self):
        saved = np.load(os.path.join(config.ARTIFACTS_DIR, "X_test_pca.npy"))
        np.testing.assert_allclose(project(self.X_test, self.mean, self.components, self.k), saved, atol=1e-9)

    def test_predict_py_agrees_with_pipeline_on_all_test_images(self):
        Z_test = np.load(os.path.join(config.ARTIFACTS_DIR, "X_test_pca.npy"))
        expected = self.svm.predict(Z_test)
        paths, _ = predict.test_set_files()
        got = [predict.predict_image(p, self.mean, self.components, self.k, self.svm)["label"] for p in paths]
        self.assertTrue(np.array_equal(got, expected))

    def test_metrics_json_matches_recomputation(self):
        import json
        with open(os.path.join(config.ARTIFACTS_DIR, "metrics.json")) as f:
            saved = json.load(f)
        m = metrics.evaluate(self.y_test, self.svm.predict(project(self.X_test, self.mean, self.components, self.k)))
        self.assertEqual(saved["test_confusion_matrix"], m["confusion_matrix"].tolist())

    def test_reconstruction_error_shrinks_as_k_grows(self):
        ks = [1, 5, 20, 50, self.k]
        errs = metrics.reconstruction_error_curve(self.X_train, self.mean, self.components, ks)
        self.assertTrue(all(a > b for a, b in zip(errs, errs[1:])), errs)
        # my per-image helper and Member 2's overall function must agree
        per_image = metrics.reconstruction_error_per_image(self.X_train, self.mean, self.components, self.k)
        self.assertAlmostEqual(float(per_image.mean()),
                               reconstruction_error(self.X_train, self.mean, self.components, self.k))

    def test_reconstruction_has_original_shape(self):
        Z = project(self.X_test[:3], self.mean, self.components, self.k)
        self.assertEqual(reconstruct(Z, self.mean, self.components).shape, (3, IMG_SIZE * IMG_SIZE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
