import os
import sys
sys.path.append('../')
from util import *
from general import *
import numpy as np
import time
import librosa
import cPickle as pickle
from python_speech_features import logfbank, mfcc
import scipy.io.wavfile as wav


def extract_feature(files, path):
    check_file(files)
    check_path(path)
    lines = open(files, 'rb').readlines()
    for id_, items in enumerate(lines):
        items = items.split('\n')[0].split()
        uttName = items[0]
        uttFile = items[1]
        y, sr = librosa.load(uttFile, sr=16000, mono=True)
        fbankFeat = logfbank(y, samplerate=sr, winlen=0.025, winstep=0.01, nfilt=40, nfft=512)
        pickle.dump(fbankFeat, open(os.path.join(path, uttName+'.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print 'Extract Feature [{:d}|{:d}] {:d} ({:d},{:d}) {:s}\t\r'.format(
            id_+1, lines.__len__(), sr, fbankFeat.shape[0], fbankFeat.shape[1], uttFile),
        sys.stdout.flush()
    print ''


if __name__ == '__main__':
    #files = '/aifs1/users/kxd/sre/data/test/xytx_far_1214_volumeup/test/wav.scp'
    #files = '/aifs1/users/kxd/sre/data/xytx_aug_fbank/train/wav.scp'
    #files = '/aifs1/users/kxd/sre/data/data_aug_fbank/train/wav.scp'
    files = '/aifs1/users/kxd/sre/data/xiaoyutongxue/wav.scp'
    path = '../../logfbank'
    make_path(path)

    extract_feature(files, path)


