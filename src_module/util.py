import os
import sys
import math
import time
import copy
import pydub
import numpy
import numpy as np
import random
import logging
import argparse
import cPickle
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo


import kaldi_io
#sys.path.append('/Users/ljpc/Tools/kaldi2py')
#import kaldi_io

