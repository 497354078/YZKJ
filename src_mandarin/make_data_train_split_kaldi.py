import sys
sys.path.append('../')
from general import *
'''
def vad_process(data, vad):
    newData = []
    assert data.shape[0] == len(vad) or data.shape[0]-1 == len(vad) or data.shape[0]-2 == len(vad)
    #print vad
    for i in range(len(vad)):
        if vad[i] == 1:
            newData.append(data[i, :])

    if len(newData) < 80:
        return None

    newData = np.asarray(newData)
    newData = (newData-newData.mean()) / (newData.std()+eps)
    assert newData.shape[1] == data.shape[1]
    return newData
'''
def get_data_from_kaldi_split(files, spk2utt, vadDict, splitNum, splitID):
    print 'get_data_from_kaldi_split splitNum:{:d} splitID:{:d}'.format(splitNum, splitID)
    uttDict = get_data_dict(files)
    dataDict = {}
    stime = time.time()
    for spkID, spkName in enumerate(spk2utt):
        dataDict[spkID] = []
        for id_, uttName in enumerate(spk2utt[spkName]):
            if id_ % splitNum != splitID:
                continue
            feaData = uttDict[uttName].astype(np.float32)
            #feaData = vad_process(feaData, vadDict[uttName])
            dataDict[spkID].append((uttName, feaData))
            print 'splitNum:{:d}  splitID:{:d}  spkID:{:4d}  id_:{:3d}  utstime:{:f}\t\r'.format(
                splitNum, splitID, spkID, id_, time.time()-stime),
            sys.stdout.flush()
    print ''
    print 'Finished get_data_from_kaldi_split Load\n'
    return dataDict


def process(basePath, savePath, mode='train', splitNum=5):
    print '----------------------------------------------------------------'
    check_path(basePath)
    dataPath = os.path.join(savePath, mode)
    make_path(dataPath)

    stime = time.time()

    spk2utt = get_file_dict(os.path.join(basePath, 'spk2utt'))
    utt2spk = get_file_dict(os.path.join(basePath, 'utt2spk'))

    if os.path.isfile(os.path.join(dataPath, 'vad.dict')):
        print 'load_vad_dict from [{:s}]\n'.format(os.path.join(dataPath, 'vad.dict'))
        vadDict = pickle.load(open(os.path.join(dataPath, 'vad.dict'), 'rb'))
    else:
        vadDict = get_vad_dict(os.path.join(basePath, 'vad.scp'))
        pickle.dump(vadDict, open(os.path.join(dataPath, 'vad.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)


    utt2path = get_file_dict(os.path.join(basePath, 'feats.scp'))
    splitDict = {}
    for spkID, spkName in enumerate(spk2utt):
        for id_, uttName in enumerate(spk2utt[spkName]):
            splitID = id_ % splitNum
            uttPath = utt2path[uttName][0]
            if splitID not in splitDict:
                splitDict[splitID] = []
            splitDict[splitID].append(uttName+' '+uttPath+'\n')
    for splitID in splitDict:
        f = open(os.path.join(dataPath, 'train.{:d}.scp'.format(splitID)), 'wb')
        for item in splitDict[splitID]:
            f.write(item)
        f.close()


    for splitID in range(splitNum):
        trainDict = get_data_from_kaldi_split(
                            os.path.join(dataPath, 'train.{:d}.scp'.format(splitID)),
                            spk2utt, vadDict, splitNum, splitID
                            )
        pickle.dump(trainDict, open(os.path.join(dataPath, 'train.{:d}.dict'.format(splitID)), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print 'spk2utt: ', len(spk2utt)
    print 'utt2spk: ', len(utt2spk)
    print 'vadDict: ', len(vadDict)
    print 'Load time: {:f}\n'.format(time.time()-stime)


if __name__ == '__main__':
    print '----------------------------------------------------------------'
    #trainPath = '/aifs1/users/kxd/sre/data/xytx_aug_fbank/train' # xiaoyutongxue 43w
    #trainPath = '/aifs1/users/kxd/sre/data/data_aug_fbank/train' # accent & mandarin 173w
    #feaPath = '../../logfbank'
    trainPath = '../../data/mandarin'
    savePath = '../../data/far-mandarin-kaldi-novad'
    splitNum = 5
    process(trainPath, savePath, 'train', splitNum)

