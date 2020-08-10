# -*- coding: utf-8 -*-

import os
import json
import shutil
import numpy as np
from scipy.stats import ttest_rel

TARGET = ("CVD", "HF", "AFib", "CAD", "MI", "Stroke", "DENOVO")
ITERATION = tuple(range(100))
FEATURE = ("ALL", "LAB", "ECHO")
ALGORITHM = ("KNN", "LR", "SVM", "RF", "GB")
METRIC = ("AUROC", "AUPR", "ACC", "SE", "SP", "PPV", "NPV", "Kappa", "MCC")

if os.path.exists("result.npy"):
    RESULT = np.load("result.npy")
else:
    RESULT = np.zeros((len(TARGET), len(ITERATION), len(FEATURE), len(ALGORITHM), len(METRIC)))
    for d0, target in enumerate(TARGET):
        for d1, iteration in enumerate(ITERATION):
            for d2, feature in enumerate(FEATURE):
                for d3, algorithm in enumerate(ALGORITHM):
                    with open("../test/%s/%s-%s-%s.json" % (target, iteration, feature, algorithm)) as fi:
                        result = json.load(fi)["TE"]
                    for d4, metric in enumerate(METRIC):
                        RESULT[d0, d1, d2, d3, d4] = result[metric]
    np.save("result.npy", RESULT)

for idx, metric in enumerate(METRIC):
    result = RESULT[:, :, :, :, idx]
    np.savetxt("AVG_%s.txt" % metric, result.mean(axis=1).reshape(7, -1), delimiter="\t", fmt="%.6f")
    np.savetxt("STD_%s.txt" % metric, result.std(axis=1).reshape(7, -1), delimiter="\t", fmt="%.6f")

AUROC = RESULT[:, :, :, :, 0]

print("\nP value for LAB vs ECHO (LR)")
LR = AUROC[:, :, :, 1]
for i in range(7):
    print(ttest_rel(LR[i, :, 1], LR[i, :, 2])[1])

print("\nP value for LR vs others (ALL)")
ALL = AUROC[:, :, 0, :]
for i in range(7):
    for j in (0, 2, 3, 4):
        print(ttest_rel(ALL[i, :, 1], ALL[i, :, j])[1])

MAX = AUROC.argmax(axis=1)
for d0, target in enumerate(TARGET):
    for d1, feature in enumerate(FEATURE):
        for d2, algorithm in enumerate(ALGORITHM):
            if algorithm == "LR":
                iteration = MAX[d0, d1, d2]
                shutil.copyfile(
                    "../result/%s_%s_%s_%s_roc.npy" % (target, iteration, feature, algorithm.lower()),
                    "./max/%s_%s_%s_roc.npy" % (target, feature, algorithm),
                )
                shutil.copyfile(
                    "../result/%s_%s_%s_%s_pr.npy" % (target, iteration, feature, algorithm.lower()),
                    "./max/%s_%s_%s_pr.npy" % (target, feature, algorithm),
                )
