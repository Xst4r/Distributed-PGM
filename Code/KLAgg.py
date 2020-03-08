import os
import pxpy as px

from src.conf.settings import ROOT_DIR
from src.model.aggregation import KL
path =  os.path.join(ROOT_DIR, "experiments", "COVERTYPE", "100_50_1_1583660364", "models")

models = []
theta = []
logpartitions = []
suff_stats = []
for file in os.listdir(path):
    if "k0" in file:
        models.append(px.load_model(os.path.join(path, file)))

for model in models:
    m, A = model.infer()
    logpartitions.append(A)
    suff_stats.append(model.statistics)
    theta.append(model.weights)

agg = KL(models, logpartitions, suff_stats)


# KL(theta1 |theta2) = A2 - A1  - mu1(theta2 - theta1)
# KL(i,j) = A[j] - A[i] - suff_stats[i]*(theta[j] - theta[i])