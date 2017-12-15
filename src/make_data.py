import sys
sys.path.append('../')
from util import *


def vad_process(data, vad):
    newData = []
    assert data.shape[0] == len(vad)
    #print vad
    for i in range(len(vad)):
        if vad[i] == 1:
            newData.append(data[i, :])

    if len(newData) < 80:
        return None

    newData = np.asarray(newData)
    assert newData.shape[1] == data.shape[1]

    return newData


def split_data(spk2utt, uttDict, vadDict):
    trainDict = {}
    validDict = {}
    flag = True
    spkID = 0
    for _, spk in enumerate(spk2utt):
        dataList = []
        for id_, utt in enumerate(spk2utt[spk]):
            if utt not in vadDict:
                raise IOError('utt not in vadDict')
            data = uttDict[utt]
            data = vad_process(data, vadDict[utt])
            if data is None:
                continue
            dataList.append(data)
            if flag == True:
                print 'spk: ', type(spk), spk
                print 'data: ', type(data), data.shape
                flag = False
        if len(dataList) < 10:
            print 'len(dataList) {:d}<10: spkID:{:d} spkName:{:s}'.format(len(dataList), spkID, spk)
            continue
        trainDict[spkID] = []
        validDict[spkID] = []
        split_num = int(len(dataList)*0.8)
        for id_ in range(len(dataList)):
            if id_ <= split_num:
                trainDict[spkID].append(dataList[id_])
            else:
                validDict[spkID].append(dataList[id_])
        spkID += 1

    print 'get trainDict and validDict finished\n'
    return trainDict, validDict

def make_data(spk2utt, uttDict, vadDict):
    dataDict = {}
    flag = True
    for _, spk in enumerate(spk2utt):
        dataDict[spk] = []
        for id_, utt in enumerate(spk2utt[spk]):
            if utt not in vadDict:
                raise IOError('utt not in vadDict')
            data = uttDict[utt]
            data = vad_process(data, vadDict[utt])
            if data is None:
                continue
            if flag == True:
                print 'spk: ', type(spk), spk
                print 'data: ', type(data), data.shape
                flag = False
            dataDict[spk].append(data)
    print 'get dataDict finished\n'
    return dataDict

def process(basePath, mode='train'):
    print '----------------------------------------------------------------'
    check_path(basePath)
    dataPath = os.path.join(savePath, mode)
    make_path(dataPath)

    stime = time.time()

    spk2utt = get_file_dict(os.path.join(basePath, 'spk2utt'))
    utt2spk = get_file_dict(os.path.join(basePath, 'utt2spk'))

    if os.path.isfile(os.path.join(dataPath, 'utt.dict')):
        print 'load_data_dict from [{:s}]\n'.format(os.path.join(dataPath, 'utt.dict'))
        uttDict = pickle.load(open(os.path.join(dataPath, 'utt.dict'), 'rb'))
    else:
        uttDict = get_data_dict(os.path.join(basePath, 'feats.scp'))
        pickle.dump(uttDict, open(os.path.join(dataPath, 'utt.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(os.path.join(dataPath, 'vad.dict')):
        print 'load_vad_dict from [{:s}]\n'.format(os.path.join(dataPath, 'vad.dict'))
        vadDict = pickle.load(open(os.path.join(dataPath, 'vad.dict'), 'rb'))
    else:
        vadDict = get_vad_dict(os.path.join(basePath, 'vad.scp'))
        pickle.dump(vadDict, open(os.path.join(dataPath, 'vad.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print 'spk2utt: ', len(spk2utt)
    print 'utt2spk: ', len(utt2spk)
    print 'uttDict: ', len(uttDict)
    print 'vadDict: ', len(vadDict)
    print 'Load time: {:f}\n'.format(time.time()-stime)

    if mode == 'train':
        trainDict, validDict = split_data(spk2utt, uttDict, vadDict)
        pickle.dump(trainDict, open(os.path.join(dataPath, 'train.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(validDict, open(os.path.join(dataPath, 'valid.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    else:
        dataDict = make_data(spk2utt, uttDict, vadDict)
        pickle.dump(dataDict, open(os.path.join(dataPath, '{:s}.dict'.format(mode)), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print 'make_dataDict time: {:f}\n'.format(time.time()-stime)

if __name__ == '__main__':
    print '----------------------------------------------------------------'
    #trainPath = '/aifs1/users/kxd/sre/data/xytx_aug_fbank/train'
    #savePath = '../../data/far'
    #process(trainPath, 'train')


    #enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/enro'
    #testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/test'
    #savePath = '../../data/far'
    #process(enroFarPath, 'enro_far_1208')
    #process(testFarPath, 'test_far_1208')

    enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/enro'
    testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/test'
    savePath = '../../data/far'
    process(enroFarPath, 'enro_far_1214')
    process(testFarPath, 'test_far_1214')

    #enroNearPath = '/aifs1/users/kxd/sre/data/test/xiaoyutongxue/enro'
    #testNearPath = '/aifs1/users/kxd/sre/data/test/xiaoyutongxue/test'
    #savePath = '../../data/near'
    #process(enroNearPath, 'enro_near')
    #process(testNearPath, 'test_near')



