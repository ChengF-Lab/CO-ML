# -*- coding: utf-8 -*-

import sys
import time
import json
import random
import numpy as np
import sklearn
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.linear_model import LogisticRegression  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.svm import SVC  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn.ensemble import RandomForestClassifier  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import GradientBoostingClassifier  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

random.seed(1024)
print("sklearn: %s" % sklearn.__version__)


# ==========================================================
def Evaluate(y_prob, y_true):
    y_pred = y_prob.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ACC"  : accuracy_score(y_true, y_pred),
        "SE"   : tp / (tp + fn),
        "SP"   : tn / (tn + fp),
        "PPV"  : tp / (tp + fp),
        "NPV"  : tn / (tn + fn),
        "AUROC": roc_auc_score(y_true, y_prob[:, 1]),
        "AUPR" : average_precision_score(y_true, y_prob[:, 1]),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC"  : matthews_corrcoef(y_true, y_pred),
    }


def Product(d):
    keys = d.keys()
    for element in product(*[d[k] for k in keys]):
        yield dict(zip(keys, element))


# ==========================================================
TARGET = ("CVD", "HF", "AFib", "CAD", "MI", "Stroke", "DENOVO")
FEATURE = {
    "ALL" : slice(0, 95),
    "LAB" : slice(0, 40),
    "ECHO": slice(40, 95),
}
ALGORITHM = {
    "KNN": (KNeighborsClassifier, {"n_jobs": 8}, {
        "n_neighbors": (3, 5, 7),
        "metric"     : ("euclidean", "correlation"),
    }),
    "LR": (LogisticRegression, {"random_state": 1024, "max_iter": 1000, "solver": "lbfgs", "penalty": "l2"}, {
        "C": (0.01, 0.1, 1, 10, 100, 1000),
    }),
    "SVM": (SVC, {"random_state": 1024, "kernel": "rbf", "probability": True}, {
        "C"    : (0.1, 1, 10, 100),
        "gamma": (1e-3, 1e-2, 1e-1),
    }),
    "RF": (RandomForestClassifier, {"random_state": 1024, "n_jobs": 1, "n_estimators": 100, "criterion": "gini", "oob_score": True}, {
        "max_features": (0.1, 0.2, 0.4, 0.8),
        "max_depth"   : (4, 8, 12),
    }),
    "GB": (GradientBoostingClassifier, {"random_state": 1024, "loss": "deviance", "criterion": "friedman_mse"}, {
        "n_estimators" : (100, 500),
        "max_depth"    : (2, 3, 4),
        "learning_rate": (0.01, 0.05, 0.1),
        "subsample"    : (0.33, 0.66, 1),
    }),
}

# ==========================================================
ITERATION = range(100) if len(sys.argv) == 1 else range(int(sys.argv[1]), int(sys.argv[2]))

for target in TARGET:
    path_out = "../hp/%s/" % target
    for iteration in ITERATION:
        path_in = "../data/%s/%s/" % (target, iteration)
        Xtr2_ = np.load(path_in + "Xtr2.npy")
        Xval_ = np.load(path_in + "Xval.npy")
        ytr2 = np.load(path_in + "ytr2.npy")
        yval = np.load(path_in + "yval.npy")
        for feature in FEATURE:
            Xtr2 = Xtr2_[:, FEATURE[feature]]
            Xval = Xval_[:, FEATURE[feature]]
            for algorithm in ALGORITHM:
                classifier, arg, hp_set = ALGORITHM[algorithm]
                result = []
                for hp in Product(hp_set):
                    start = time.time()
                    cls = classifier(**arg, **hp)
                    cls.fit(Xtr2, ytr2)
                    y_prob_tr2 = cls.predict_proba(Xtr2)
                    y_prob_val = cls.predict_proba(Xval)
                    result_tr2 = Evaluate(y_prob_tr2, ytr2)
                    result_val = Evaluate(y_prob_val, yval)
                    if algorithm == "RF":
                        result_tr2 = Evaluate(cls.oob_decision_function_, ytr2)
                    result.append({"HP": hp, "TR": result_tr2, "VAL": result_val})
                    print("*-%05.1f-*" % (time.time() - start), target, iteration, feature, algorithm, hp, result_tr2["AUROC"], result_val["AUROC"])
                with open("%s%s-%s-%s.json" % (path_out, iteration, feature, algorithm), "w") as fo:
                    json.dump(result, fo, indent=4)
