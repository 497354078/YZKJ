import os
import sys
import cPickle as pickle

def merge_train_dict(d1, d2):
    d3 = {}
    spkID = 0
    key = None
    val = None
    for k1 in d1:
        if k1 in d3:
            raise IOError('key1[{:s}] has already existed in dict3'.format(str(k1)))
        d3[spkID] = d1[k1]
        spkID += 1
        key = k1
        val = d1[k1]
    #print 'key: ', key, 'val: ', val
    for k2 in d2:
        if k2+len(d1) in d3:
            raise IOError('key2[{:s}] has already existed in dict3'.format(str(k2+len(d1))))
        d3[spkID] = d2[k2]
        spkID += 1
        key = k2
        val = d2[k2]
    #print 'key: ', key, 'val: ', val
    print 'len(dict3): ', len(d3)
    #print ''
    return d3

def merge_vad_dict(d1, d2):
    d3 = {}
    key = None
    val = None
    for k1 in d1:
        if k1 in d3:
            raise IOError('key1[{:s}] has already existed in dict3'.format(str(k1)))
        d3[k1] = d1[k1]
        key = k1
        val = d1[k1]
    #print 'key: ', key, 'val: ', val
    for k2 in d2:
        if k2 in d3:
            raise IOError('key2[{:s}] has already existed in dict3'.format(str(k2)))
        d3[k2] = d2[k2]
        key = k2
        val = d2[k2]
    #print 'key: ', key, 'val: ', val
    print 'len(dict3): ', len(d3)
    #print ''
    return d3

def process_train(files1, files2, files3):
    d1 = pickle.load(open(files1, 'rb'))
    d2 = pickle.load(open(files2, 'rb'))
    d3 = merge_train_dict(d1, d2)
    pickle.dump(d3, open(files3, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def process_vad(files1, files2, files3):
    d1 = pickle.load(open(files1, 'rb'))
    d2 = pickle.load(open(files2, 'rb'))
    d3 = merge_vad_dict(d1, d2)
    pickle.dump(d3, open(files3, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    rawPath1 = '../../data/far-accent-speech-novad/train'
    rawPath2 = '../../data/far-mandarin-speech-novad/train'
    destPath = '../../data/far-am-speech-novad/train'
    if not os.path.exists(destPath):
        os.makedirs(destPath)

    
    fileName = ['train.0.dict', 'train.1.dict', 'train.2.dict', 'train.3.dict',
                'train.4.dict']

    for items in fileName:
        process_train(os.path.join(rawPath1, items),
                os.path.join(rawPath2, items),
                os.path.join(destPath, items))
    
    process_vad(os.path.join(rawPath1, 'vad.dict'),
            os.path.join(rawPath2, 'vad.dict'),
            os.path.join(destPath, 'vad.dict'))

