import os
import sys
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_wav_list(path, files):
    print 'load files from [{:s}]'.format(os.path.join(path, files))
    lines = open(os.path.join(path, files), 'rb').readlines()
    uttDict = {}
    key = None
    val = None
    for items in lines:
        uttName, _ = os.path.splitext(os.path.basename(items))
        uttDict[uttName] = True
        key = uttName
        val = uttDict[uttName]
    print 'len(uttDict): {:d}  key:{:s} val:{:s}\n'.format(len(uttDict), str(key), str(val))
    return uttDict

def write_list(list_, files):
    with open(files, 'wb') as f:
        for items in list_:
            f.write(items)
        f.close()
    print 'file saved in [{:s}]'.format(files)

def split_file(path, files, accUttDict, manUttDict):
    print 'load files from [{:s}]'.format(os.path.join(path, files))
    lines = open(os.path.join(path, files), 'rb').readlines()
    accList = []
    manList = []
    key = None
    val = None
    for items in lines:
        uttName = items.split()[0]
        uttName = uttName.replace('-', 'S')
        uttName = uttName.split('_')[0]
        if uttName in accUttDict:
            accList.append(items)
        elif uttName in manUttDict:
            manList.append(items)
        else:
            raise IOError('uttName[{:s}] cannot found'.format(uttName))
        val = items

    print 'len(accList): {:d}  len(manList): {:d}  val:{:s}'.format(len(accList), len(manList), str(val))
    write_list(accList, os.path.join(accPath, files))
    write_list(manList, os.path.join(manPath, files))
    print ''

if __name__ == '__main__':
    print ''
    basePath = '/aifs1/users/kxd/sre/data/2000h-baseline-201709-revised'

    accUttDict = get_wav_list(basePath, 'accent.wavlist')
    manUttDict = get_wav_list(basePath, 'mandarin.wavlist')

    totPath = '/aifs1/users/kxd/sre/data/data_aug_fbank/train'
    accPath = '../../data/accent'
    manPath = '../../data/mandarin'
    make_path(accPath)
    make_path(manPath)
 
    split_file(totPath, 'feats.scp', accUttDict, manUttDict)
    split_file(totPath, 'wav.scp', accUttDict, manUttDict)
    split_file(totPath, 'vad.scp', accUttDict, manUttDict)
    split_file(totPath, 'utt2spk', accUttDict, manUttDict)


