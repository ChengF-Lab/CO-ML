# -*- coding: utf-8 -*-

import numpy as np

TARGET = ("CVD", "HF", "AFib", "CAD", "MI", "Stroke", "DENOVO")
ITERATION = tuple(range(100))

WEIGHT = np.zeros((len(TARGET), len(ITERATION), 95))

for d0, target in enumerate(TARGET):
    for d1, iteration in enumerate(ITERATION):
        WEIGHT[d0, d1] = np.load("../result/%s_%s_ALL_lr_weight.npy" % (target, iteration))

np.savetxt("WEIGHT_AVG.txt", WEIGHT.mean(axis=1).T, delimiter="\t", fmt="%.6f")
np.savetxt("WEIGHT_STD.txt", WEIGHT.std(axis=1).T, delimiter="\t", fmt="%.6f")
