# -*- coding: utf-8 -*-

import sys
import time
import json
import random
import numpy as np
import sklearn
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve
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


# ==========================================================
TARGET = ("CVD", "HF", "AFib", "CAD", "MI", "Stroke", "DENOVO")
FEATURE = {
    "ALL" : slice(0, 95),
    "LAB" : slice(0, 40),
    "ECHO": slice(40, 95),
}
ALGORITHM = {
    "KNN": (KNeighborsClassifier, {"n_jobs": 8}),
    "LR" : (LogisticRegression, {"random_state": 1024, "max_iter": 1000, "solver": "lbfgs", "penalty": "l2"}),
    "SVM": (SVC, {"random_state": 1024, "kernel": "rbf", "probability": True}),
    "RF" : (RandomForestClassifier, {"random_state": 1024, "n_jobs": 1, "n_estimators": 100, "criterion": "gini", "oob_score": True}),
    "GB" : (GradientBoostingClassifier, {"random_state": 1024, "loss": "deviance", "criterion": "friedman_mse"}),
}

# ==========================================================
ITERATION = range(100) if len(sys.argv) == 1 else range(int(sys.argv[1]), int(sys.argv[2]))

for target in TARGET:
    path_hp = "../hp/%s/" % target
    path_out = "../test/%s/" % target
    for iteration in ITERATION:
        path_in = "../data/%s/%s/" % (target, iteration)
        Xtr_ = np.load(path_in + "Xtr.npy")
        Xte_ = np.load(path_in + "Xte.npy")
        ytr = np.load(path_in + "ytr.npy")
        yte = np.load(path_in + "yte.npy")
        for feature in FEATURE:
            Xtr = Xtr_[:, FEATURE[feature]]
            Xte = Xte_[:, FEATURE[feature]]
            for algorithm in ALGORITHM:
                with open("%s%s-%s-%s.json" % (path_hp, iteration, feature, algorithm)) as fi:
                    result = json.load(fi)
                    result.sort(key=lambda x: x["VAL"]["AUROC"], reverse=True)
                    hp = result[0]["HP"]
                classifier, arg = ALGORITHM[algorithm]
                start = time.time()
                cls = classifier(**arg, **hp)
                cls.fit(Xtr, ytr)
                y_prob_tr = cls.predict_proba(Xtr)
                y_prob_te = cls.predict_proba(Xte)
                result_tr = Evaluate(y_prob_tr, ytr)
                result_te = Evaluate(y_prob_te, yte)
                if algorithm == "RF":
                    result_tr = Evaluate(cls.oob_decision_function_, ytr)
                elif algorithm == "LR":
                    name_prefix = "../result/%s_%s_%s_lr_" % (target, iteration, feature)
                    np.save(name_prefix + "weight.npy", cls.coef_[0])
                    fpr, tpr, _ = roc_curve(yte, y_prob_te[:, 1])
                    np.save(name_prefix + "roc.npy", np.vstack((fpr, tpr)))
                    precision, recall, _ = precision_recall_curve(yte, y_prob_te[:, 1])
                    np.save(name_prefix + "pr.npy", np.vstack((precision, recall)))
                print("*-%05.1f-*" % (time.time() - start), target, iteration, feature, algorithm, hp, result_tr["AUROC"], result_te["AUROC"])
                with open("%s%s-%s-%s.json" % (path_out, iteration, feature, algorithm), "w") as fo:
                    json.dump({"HP": hp, "TR": result_tr, "TE": result_te}, fo, indent=4)
