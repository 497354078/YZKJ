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

    if len(newData) < 120:
        print len(newData),
        return None

    newData = np.asarray(newData)
    assert newData.shape[1] == data.shape[1]

    return newData

def make_data(uttDict, vadDict):
    dataDict = {}
    flag = True
    for id_, utt in enumerate(uttDict):
        if utt not in vadDict:
            raise IOError('utt not in vadDict')
        data = uttDict[utt]
        data = vad_process(data, vadDict[utt])
        if data is None:
            print 'after vad the [{:s}] data is None'.format(utt)
            continue
        if flag == True:
            print 'data: ', type(data), data.shape
            flag = False
        dataDict[utt] = data
    print 'get dataDict finished\n'
    return dataDict


def process(basePath, mode='enro'):
    print '----------------------------------------------------------------'
    check_path(basePath)
    dataPath = os.path.join(savePath, mode)
    make_path(dataPath)

    stime = time.time()

    if os.path.isfile(os.path.join(dataPath, 'vad.dict')):
        print 'load_vad_dict from [{:s}]\n'.format(os.path.join(dataPath, 'vad.dict'))
        vadDict = pickle.load(open(os.path.join(dataPath, 'vad.dict'), 'rb'))
    else:
        vadDict = get_vad_dict(os.path.join(basePath, 'vad.scp'))
        pickle.dump(vadDict, open(os.path.join(dataPath, 'vad.dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    uttDict = get_data_dict(os.path.join(basePath, 'feats.scp'))
    uttDict = make_data(uttDict, vadDict)
    pickle.dump(uttDict, open(os.path.join(dataPath, mode+'.dict'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    print 'uttDict: ', len(uttDict)
    print 'vadDict: ', len(vadDict)
    print 'make_dataDict time: {:f}\n'.format(time.time()-stime)

if __name__ == '__main__':
    print '----------------------------------------------------------------'
    #enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/enro' # 19
    #testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_1208/test'
    #savePath = '../../data/far'
    #process(enroFarPath, 'enro_far_1208')
    #process(testFarPath, 'test_far_1208')

    #enroFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/enro' # 183
    #testFarPath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/test'
    enroFarPath = '../../data/xytx_far_1214/enro'
    testFarPath = '../../data/xytx_far_1214/test'
    savePath = '../../data/far'
    process(enroFarPath, 'enro_far_1214')
    process(testFarPath, 'test_far_1214')

    #enroNearPath = '/aifs1/users/kxd/sre/data/test/xiaoyutongxue/enro' # 183
    #testNearPath = '/aifs1/users/kxd/sre/data/test/xiaoyutongxue/test'
    #savePath = '../../data/near'
    #process(enroNearPath, 'enro_near')
    #process(testNearPath, 'test_near')




