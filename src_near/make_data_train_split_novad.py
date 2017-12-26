import sys
sys.path.append('../src_module/')
from general import *

def vad_process(data, vad):
    newData = []
    assert data.shape[0] == len(vad) or data.shape[0]-1 == len(vad) or data.shape[0]-2 == len(vad)
    #print vad
    for i in range(len(vad)):
        if vad[i] == 1:
            newData.append(data[i, :])

    if len(newData) < 80:
        return None

    newData = np.asarray(newData, dtype=np.float32)
    newData = (newData-newData.mean()) / (newData.std()+eps)
    assert newData.shape[1] == data.shape[1]
    return newData

def get_data_from_logfbank_split(feaPath, spk2utt, vadDict, splitNum, splitID):
    check_path(feaPath)
    print 'get_data_from_logfbank splitNum:{:d} splitID:{:d}'.format(splitNum, splitID)
    dataDict = {}
    stime = time.time()
    for spkID, spkName in enumerate(spk2utt):
        dataDict[spkID] = []
        for id_, uttName in enumerate(spk2utt[spkName]):
            if id_ % splitNum != splitID:
                continue
            if uttName.find('noise') != -1 or uttName.find('reverb') != -1:
                continue
            feaData = pickle.load(open(os.path.join(feaPath, uttName+'.pkl'), 'rb'))
            feaData = feaData.astype(np.float32)
            #feaData = vad_process(feaData, vadDict[uttName])
            dataDict[spkID].append((uttName, feaData))
            print 'splitNum:{:d}  splitID:{:d}  spkID:{:4d}  id_:{:3d}  utstime:{:f}\t\r'.format(
                splitNum, splitID, spkID, id_, time.time()-stime),
            sys.stdout.flush()
    print ''
    print 'Finished get_data_from_logfbank Load\n'
    return dataDict


def process(basePath, feaPath, savePath, mode='train', splitNum=5):
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

    for splitID in range(splitNum):
        trainDict = get_data_from_logfbank_split(feaPath, spk2utt, vadDict, splitNum, splitID)
        pickle.dump(trainDict, open(os.path.join(dataPath, 'train.{:d}.dict'.format(splitID)), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print 'spk2utt: ', len(spk2utt)
    print 'utt2spk: ', len(utt2spk)
    print 'vadDict: ', len(vadDict)
    print 'Load time: {:f}\n'.format(time.time()-stime)


if __name__ == '__main__':
    print '----------------------------------------------------------------'
    #trainPath = '/aifs1/users/kxd/sre/data/xytx_aug_fbank/train' # xiaoyutongxue 43w
    trainPath = '/aifs1/users/kxd/sre/data/data_aug_fbank/train' # accent & mandarin 173w
    feaPath = '../../logfbank'
    savePath = '../../data/near-am-speech-novad'
    splitNum = 5
    process(trainPath, feaPath, savePath, 'train', splitNum)



