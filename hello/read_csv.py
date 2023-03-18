import os
import numpy as np


annotation_path = "data/dst/train/annotations"

annotations = []
for i, fname in enumerate(os.listdir(annotation_path)):
    fpath = os.path.join(annotation_path, fname)
    annotation = np.genfromtxt(fpath, delimiter=",", dtype=np.float32)
    annotations.append(annotation)

    print(i, annotation)
