import sys
sys.path.append('../')
from general import *

def vad_process(data, vad):
    newData = []
    assert data.shape[0] == len(vad) or data.shape[0]-1 == len(vad) or data.shape[0]-2 == len(vad)
    for i in range(len(vad)):
        if vad[i] == 1:
            newData.append(data[i, :])
    if len(newData) < 80:
        return None
    newData = np.asarray(newData)
    assert newData.shape[1] == data.shape[1]
    return newData

def get_data_from_logfbank(feaPath, spk2utt, vadDict, FLAG):
    check_path(feaPath)
    print 'get_data_from_logfbank [{:s}]'.format(feaPath)
    dataDict = {}
    stime = time.time()
    for spkID, spkName in enumerate(spk2utt):
        dataDict[spkID] = []
        for id_, uttName in enumerate(spk2utt[spkName]):
            feaData = pickle.load(open(os.path.join(feaPath, uttName+'.pkl'), 'rb'))
            feaData = feaData.astype(np.float32)
            if FLAG == True:
                feaData = vad_process(feaData, vadDict[uttName])
            dataDict[spkID].append((uttName, feaData))

    print 'Finished get_data_from_logfbank Load\n'
    return dataDict


def process(basePath, feaPath, savePath, mode, FLAG):
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

    trainDict = get_data_from_logfbank(feaPath, spk2utt, vadDict, FLAG)
    pickle.dump(trainDict, open(os.path.join(dataPath, mode+'.dict'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    print 'spk2utt: ', len(spk2utt)
    print 'utt2spk: ', len(utt2spk)
    print 'vadDict: ', len(vadDict)
    print 'Load time: {:f}\n'.format(time.time()-stime)


if __name__ == '__main__':
    print '----------------------------------------------------------------'
    FLAG = True
    feaPath = '../../logfbank'
    feaType = 'far-mandarin-speech-novad'
    #enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/enro'
    #testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/test'
    enroFarPath = '../../data/xytx_far_1214/enro'
    testFarPath = '../../data/xytx_far_1214/test'
    savePath = '../../data/'+feaType
    process(enroFarPath, feaPath, savePath, 'enro_far_1214', FLAG)
    process(testFarPath, feaPath, savePath, 'test_far_1214', FLAG)

    #enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/enro'
    #testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/test'
    #savePath = '../../data/far'
    #process(enroFarPath, 'enro_far_1208')
    #process(testFarPath, 'test_far_1208')


