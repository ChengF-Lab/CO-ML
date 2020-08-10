# -*- coding: utf-8 -*-

import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import auc

TARGET = ("CVD", "HF", "AFib", "CAD", "MI", "Stroke", "DENOVO")
ITERATION = tuple(range(100))
FEATURE = ("ALL", "LAB", "ECHO")

RESULT = np.load("result.npy")
AUROC = RESULT[:, :, :, :, 0]
LR = AUROC[:, :, :, 1]

for d0, target in enumerate(TARGET):
    if target != "CVD":
        fig = plt.figure(figsize=(4.5, 4), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        plt.plot([0.75, 0.75], [0.75, 0.75], lw=0, label=(target if target !="AFib" else "AF") if target != "DENOVO" else "$de$ $novo$ CTRCD")
        for idx, c1, c2, label in (
                (0, "#7b7b7b", "#888888", "Echo & \nlab test"),
                (1, "#22a0cc", "#2baedc", "Lab test"),
                (2, "#d45da5", "#d971b0", "Echo")
        ):
            box = ax.boxplot([LR[d0, :, idx]], notch=True, positions=[idx], widths=0.75, patch_artist=True,
                             capprops={"color": c1, "linewidth": 1.5},
                             boxprops={"color": c1, "linewidth": 1.5},
                             whiskerprops={"color": c1, "linewidth": 1.5},
                             medianprops={"color": c1, "linewidth": 1.5},
                             flierprops={"markerfacecolor": c1 + "40", "markersize": 6, "markeredgecolor": c1},
                             )
            box["boxes"][-1].set_facecolor(c2 + "40")

        if target == "Stroke":
            ax.set_ylim(0.35, 0.9)
            ax.set_yticks((0.40, 0.50, 0.60, 0.7, 0.8))
        elif target == "HF":
            ax.set_ylim(0.65, 1.05)
            ax.set_yticks((0.70, 0.80, 0.90, 1.0))
        else:
            ax.set_ylim(0.55, 0.95)
            ax.set_yticks((0.60, 0.70, 0.80, 0.90))
        ax.xaxis.set_ticklabels(tuple())
        ax.tick_params(labelsize=16)
        plt.legend(prop={"size": 22}, frameon=False, labelspacing=0, loc="upper right",bbox_to_anchor=(1.05, 1.05), bbox_transform=ax.transAxes)
        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.98, top=0.98)
        plt.savefig("B_%s.png" % target)
