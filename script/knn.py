import os
import shutil
from tqdm import tqdm
import random
import math
from glob import glob

import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans


proj_root_dir = os.path.abspath(os.path.join(os.getcwd(),".."))


# load
tSNE_result_dir = os.path.join(proj_root_dir, 'state_dict', 'tSNE_3D.pkl')
with open(tSNE_result_dir, 'rb') as f:
    tSNE_result = pkl.load(f)

idx_dir = os.path.join(proj_root_dir, 'state_dict', 'fft_result_wo_simplify', 'trnIdx.pkl')
with open(idx_dir, 'rb') as f:
    Idx = pkl.load(f)

# print(tSNE_result)
print(len(Idx))

pd.reset_option('display.float_format')
tSNE_result_pd = pd.DataFrame(tSNE_result)

parameters = [16, 12, 20, 24]

# instantiating ParameterGrid, pass number of clusters as input
parameter_grid = ParameterGrid({'n_clusters': parameters})

best_score = -1
kmeans_model = KMeans()     # instantiating KMeans model
silhouette_scores = []

# evaluation based on silhouette_score
for p in tqdm(parameter_grid):
    kmeans_model.set_params(**p)    # set current hyper parameter
    kmeans_model.fit(tSNE_result_pd)          # fit model on wine dataset, this will find clusters based on parameter p
    with open(os.path.join('/data1/home/xiruiling/course/AdvanceArtificialIntelligence/AAI_Proj/state_dict/knn_model', 'knn_'+str(p)+'_.pkl'), 'wb') as f:
        pkl.dump(kmeans_model, f)
    ss = metrics.silhouette_score(tSNE_result_pd, kmeans_model.labels_)   # calculate silhouette_score
    silhouette_scores += [ss]       # store all the scores

    print('Parameter:', p, 'Score', ss)

    # check p which has the best score
    if ss > best_score:
        best_score = ss
        best_grid = p

# plotting silhouette score
plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
plt.xticks(range(len(silhouette_scores)), list(parameters))
plt.title('Silhouette Score', fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

print(best_grid['n_clusters'])
