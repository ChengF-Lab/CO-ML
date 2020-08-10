# -*- coding: utf-8 -*-

import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import auc

base_fpr = np.linspace(0, 1, 101)


def PlotROC(save, *files):
    for l, f, c0, c, auroc in files:
        fig = plt.figure(figsize=(4.5, 4), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        plt.plot([0, 1], [0, 1], lw=1, c="#aaaaaa", linestyle="--", alpha=0.75)
        plt.text(x=0.48, y=0.16, s="AUROC=%.3f" % auroc, fontsize=16)
        #
        tprs = []
        for iteration in range(100):
            data = np.load("../result/%s_%s_ALL_lr_roc.npy" % (f, iteration))
            plt.plot(data[0], data[1], lw=1, solid_joinstyle="miter", c=c, alpha=0.075)
            tpr = interp(base_fpr, data[0], data[1])
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        plt.plot(base_fpr, mean_tprs, lw=2, color=c0, label=l)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=c, alpha=0.3, lw=0)
        #
        ax.set_xlabel("1 - Specificity", fontsize=20)
        ax.set_ylabel("Sensitivity", fontsize=20)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        ax.tick_params(labelsize=16)
        plt.legend(prop={"size": 20 if f!="DENOVO" else 18}, frameon=False, labelspacing=0, loc="lower right")
        plt.subplots_adjust(left=0.19, bottom=0.17, right=0.98, top=0.98)
        plt.savefig("%s_%s.png" % (save, f))


def PlotPR(save, *files):
    for l, f, c0, c, b, aupr in files:
        fig = plt.figure(figsize=(4.5, 4), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        plt.plot([0, 1], [b, b], lw=1, c="#aaaaaa", linestyle="--", alpha=0.75)
        plt.text(x=0.52, y=0.8, s="AUPR=%.3f" % aupr, fontsize=16)
        plt.text(x=0.62, y=0.7, s="BL=%.3f" % b, fontsize=16)
        #
        tprs = []
        for iteration in range(100):
            data = np.load("../result/%s_%s_ALL_lr_pr.npy" % (f, iteration))
            plt.plot(data[1], data[0], lw=1, solid_joinstyle="miter", c=c, alpha=0.075)
            tpr = interp(base_fpr, data[1][::-1], data[0][::-1])
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        plt.plot(base_fpr, mean_tprs, lw=2, color=c0, label=l)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=c, alpha=0.3, lw=0)
        #
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision", fontsize=20)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        ax.tick_params(labelsize=16)
        plt.legend(prop={"size": 20 if f!="DENOVO" else 16}, frameon=False, labelspacing=0, loc="upper right")
        plt.subplots_adjust(left=0.19, bottom=0.17, right=0.98, top=0.98)
        plt.savefig("%s_%s.png" % (save, f))


PlotROC(
    "ROC",
    ("HF", "HF", "#ea4121", "#ec5538", 0.882),
    ("AF", "AFib", "#fbb20f", "#fbba28", 0.787),
    ("CAD", "CAD", "#7faf3d", "#8cbf46", 0.821),
    ("MI", "MI", "#22a0cc", "#2baedc", 0.807),
    ("Stroke", "Stroke", "#8566dd", "#967be2", 0.660),
    ("$de$ $novo$ CTRCD", "DENOVO", "#d45da5", "#d971b0", 0.802),
)

PlotPR(
    "PR",
    ("HF", "HF", "#ea4121", "#ec5538", 0.138, 0.651),
    ("AF", "AFib", "#fbb20f", "#fbba28", 0.151, 0.401),
    ("CAD", "CAD", "#7faf3d", "#8cbf46", 0.156, 0.481),
    ("MI", "MI", "#22a0cc", "#2baedc", 0.045, 0.220),
    ("Stroke", "Stroke", "#8566dd", "#967be2", 0.064, 0.138),
    ("$de$ $novo$ CTRCD", "DENOVO", "#d45da5", "#d971b0", 0.234, 0.592),
)
