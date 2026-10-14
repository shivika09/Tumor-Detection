"""
config.py  (Member 4)

Small shared settings so main.py, predict.py and the tests all agree on
file names. Nothing here is "new maths" - it is just paths and constants.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Everything the pipeline GENERATES goes in this folder (safe to delete and regenerate).
ARTIFACTS_DIR = os.path.join(ROOT, "outputs")

PCA_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "pca_model.npz")      # written by Member 2's run_pca()
SVM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "svm_model.joblib")   # written by main.py (Member 3's model)

# These two numbers MUST equal the defaults of get_split() in tumor_project/data_loader.py.
# (tests/test_member4.py checks that they do.)
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42

# Label convention from Member 1's data_loader.py
LABEL_NO_TUMOR = 0     # folder "no"
LABEL_TUMOR = 1        # folder "yes"  (this is the "positive" class)
CLASS_NAMES = {LABEL_NO_TUMOR: "no tumor", LABEL_TUMOR: "tumor"}
