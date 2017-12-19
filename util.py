import os
import sys
import time
import pydub
import numpy
import numpy as np
import logging
import argparse
import cPickle
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics
#from sklearn.metrics.pairwise import cosine_distances
from sklearn import preprocessing
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import kaldi_io

def rePrint(string):
    print string
    logging.info(string)


def check_file(files):
    if not os.path.isfile(files):
        raise IOError('cannot found file: {:s}'.format(files))

def check_path(paths):
    if not os.path.exists(paths):
        raise IOError('cannot found path: {:s}'.format(paths))

def make_path(paths):
    if not os.path.exists(paths):
        os.makedirs(paths)


def plot_wave(y, sr=22050, logspec=None, mono=True, audioName=None, pltSave=None, show=True):

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.ylim(-1, 1)
    librosa.display.waveplot(y, sr=sr)
    plt.title(audioName)

    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    if logspec is not None:
        D = logspec
    else:
        if mono:
            S = librosa.stft(y)
        else:
            S = librosa.stft(y[0])
        D = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title('Log-frequency power spectrogram')

    plt.tight_layout()
    if pltSave is not None:
        plt.savefig(pltSave)
    if show == True:
        plt.show()
    plt.close()


def get_file_dict(files):
    check_file(files)
    lines = open(files, 'rb').readlines()
    key2list = {}
    flag = True
    for items in lines:
        items = items.split('\n')[0].split(' ')
        key = items[0]
        key2list[key] = []
        for val in items[1:]:
            key2list[key].append(val)
            if flag == True:
                print 'get_file_dict from [{:s}]'.format(files)
                print 'key: ', type(key)
                print 'val: ', type(val)
                flag = False
    print 'Finished get_file_dict Load\n'
    return key2list

def get_vad_dict(files):
    check_file(files)
    key2numpy = {}
    flag = True
    #for key, mat in kaldi_io.read_mat_scp(files):
    for key, mat in kaldi_io.read_vec_flt_scp(files):
        key2numpy[key] = mat
        if flag == True:
            print 'get_vad_dict from [{:s}]'.format(files)
            print 'key: ', type(key), key
            print 'mat: ', type(mat), mat.shape
            flag = False
        #print key, mat.shape
    print 'Finished get_vad_dict Load\n'
    return key2numpy

def get_data_dict(files):
    check_file(files)
    key2numpy = {}
    flag = True
    id_ = 0
    stime = time.time()
    lines = open(files, 'rb').readlines()
    for key, mat in kaldi_io.read_mat_scp(files):
        key2numpy[key] = mat
        if flag == True:
            print 'get_data_dict from [{:s}]'.format(files)
            print 'key: ', type(key), key
            print 'mat: ', type(mat), mat.shape
            flag = False
        #print key, mat.shape
        print '{:d} {:d} {:f}\t\r'.format(id_, lines.__len__(), time.time()-stime),
        sys.stdout.flush()
        id_ += 1
    print ''
    print 'Finished get_data_dict Load\n'
    return key2numpy

