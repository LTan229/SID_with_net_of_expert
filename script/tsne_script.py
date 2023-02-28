import os
import shutil
from tqdm import tqdm
import random
import math
from glob import glob

import soundfile as sf            # To read .flac files.   
import speech_recognition as sr   # pip install SpeechRecognition.
from sklearn.manifold import TSNE
# import cuml
# import cudf
# from cuml.manifold import TSNE as cudaTSNE
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# os.environ['OPENBLAS_NUM_THREADS'] = '32'

proj_root_dir = os.path.abspath(os.path.join(os.getcwd(),".."))
train_folder_dir = os.path.join(proj_root_dir, 'data', 'train')
state_dict_folder_dir = os.path.join(proj_root_dir, 'state_dict')
print("state_dict_folder_dir: ", state_dict_folder_dir)


with open(os.path.join(state_dict_folder_dir, 'fft_result_wo_simplify', 'trnX.pkl'), 'rb') as f:
    trnX = pkl.load(f)
print("The original dimension: ", np.array(trnX).shape)
x_tsne = TSNE(n_components=3,random_state=0).fit_transform(trnX)
# x_tsne = cudaTSNE(n_components=2, perplexity=50, learning_rate=20).fit_transform(trnX)
with open(os.path.join(state_dict_folder_dir, 'tSNE_3D.pkl'), 'wb') as f:
    pkl.dump(x_tsne, f)
print("The reduced dimension: ", np.array(x_tsne).shape)