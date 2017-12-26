import os
import sys

def get_spk_list(path, files):
    print 'load files from [{:s}]'.format(os.path.join(path, files))
    lines = open(os.path.join(path, files), 'rb').readlines()
    spkDict = {}
    key = None
    val = None
    for items in lines:
        spkName = items.split('\n')[0]
        spkDict[spkName] = True
        key = spkName
        val = spkDict[spkName]
    print 'len(uttDict): {:d}  key:{:s} val:{:s}\n'.format(len(spkDict), str(key), str(val))
    return spkDict

def write_list(list_, files):
    with open(files, 'wb') as f:
        for items in list_:
            f.write(items)
        f.close()
    print 'file saved in [{:s}]'.format(files)

def split_file(path, files, accSpkDict, manSpkDict):
    print 'load files from [{:s}]'.format(os.path.join(path, files))
    lines = open(os.path.join(path, files), 'rb').readlines()
    accList = []
    manList = []
    key = None
    val = None
    for items in lines:
        #spkName = items.split()[0]
        spkName = 'G'+items.split()[0].split('G')[1]
        if spkName in accSpkDict:
            accList.append(items)
        elif spkName in manSpkDict:
            manList.append(items)
        else:
            print 'spkName[{:s}] cannot found'.format(spkName)
            #raise IOError('spkName[{:s}] cannot found'.format(spkName))
        val = items

    print 'len(accList): {:d}  len(manList): {:d}  val:{:s}'.format(len(accList), len(manList), str(val))
    write_list(accList, os.path.join(accPath, files))
    write_list(manList, os.path.join(manPath, files))
    print ''


if __name__ == '__main__':
    print ''
    basePath = '/aifs1/users/kxd/sre/data/2000h-baseline-201709-revised'

    accSpkDict = get_spk_list(basePath, 'spk/accent.spklist')
    manSpkDict = get_spk_list(basePath, 'spk/mandarin_read.spklist')

    totPath = '/aifs1/users/kxd/sre/data/data_aug_fbank/train'
    accPath = '../../data/accent'
    manPath = '../../data/mandarin'

    split_file(totPath, 'spk2utt', accSpkDict, manSpkDict)

