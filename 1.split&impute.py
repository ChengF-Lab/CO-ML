# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

random.seed(1024)

Data = np.loadtxt("data.txt")
Label = np.loadtxt("label.txt")
for idx, target in enumerate(("CVD", "HF", "AFib", "CAD", "MI", "Stroke")):
    print(target)
    label = Label[:, idx]
    for iteration in range(100):
        folder = "../data/%s/%s/" % (target, iteration)
        if not os.path.exists(folder):
            os.mkdir(folder)

        Xtr, Xte, ytr, yte = train_test_split(Data, label, test_size=0.1, stratify=label)
        Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr)

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp.fit(Xtr2)
        Xtr = imp.transform(Xtr)
        Xte = imp.transform(Xte)
        Xtr2 = imp.transform(Xtr2)
        Xval = imp.transform(Xval)

        scale = StandardScaler()
        scale.fit(Xtr2[:, 17:85])
        Xtr[:, 17:85] = scale.transform(Xtr[:, 17:85])
        Xte[:, 17:85] = scale.transform(Xte[:, 17:85])
        Xtr2[:, 17:85] = scale.transform(Xtr2[:, 17:85])
        Xval[:, 17:85] = scale.transform(Xval[:, 17:85])

        np.save(folder + "Xtr.npy", Xtr)
        np.save(folder + "Xte.npy", Xte)
        np.save(folder + "Xtr2.npy", Xtr2)
        np.save(folder + "Xval.npy", Xval)
        np.save(folder + "ytr.npy", ytr)
        np.save(folder + "yte.npy", yte)
        np.save(folder + "ytr2.npy", ytr2)
        np.save(folder + "yval.npy", yval)

Data = np.loadtxt("data_denovo.txt")
Label = np.loadtxt("label_denovo.txt")
for idx, target in enumerate(("DENOVO",)):
    print(target)
    label = Label
    for iteration in range(100):
        folder = "../data/%s/%s/" % (target, iteration)
        if not os.path.exists(folder):
            os.mkdir(folder)

        Xtr, Xte, ytr, yte = train_test_split(Data, label, test_size=0.1, stratify=label)
        Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr)

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp.fit(Xtr2)
        Xtr = imp.transform(Xtr)
        Xte = imp.transform(Xte)
        Xtr2 = imp.transform(Xtr2)
        Xval = imp.transform(Xval)

        scale = StandardScaler()
        scale.fit(Xtr2[:, 17:85])
        Xtr[:, 17:85] = scale.transform(Xtr[:, 17:85])
        Xte[:, 17:85] = scale.transform(Xte[:, 17:85])
        Xtr2[:, 17:85] = scale.transform(Xtr2[:, 17:85])
        Xval[:, 17:85] = scale.transform(Xval[:, 17:85])

        np.save(folder + "Xtr.npy", Xtr)
        np.save(folder + "Xte.npy", Xte)
        np.save(folder + "Xtr2.npy", Xtr2)
        np.save(folder + "Xval.npy", Xval)
        np.save(folder + "ytr.npy", ytr)
        np.save(folder + "yte.npy", yte)
        np.save(folder + "ytr2.npy", ytr2)
        np.save(folder + "yval.npy", yval)
