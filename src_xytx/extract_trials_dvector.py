import sys
sys.path.append('../')
from util import *

def extract_dvector(net, data):
    net.eval()

    #print data.shape
    stime = time.time()
    data = torch.from_numpy(data).type(torch.FloatTensor)
    data = Variable(data.cuda())

    # Forward
    ttime = time.time()
    #print ttime-stime
    dvec, _ = net(data)
    etime = time.time()
    #print etime-ttime

    dvec = dvec.data.cpu().numpy()
    dvec = dvec.mean(axis=0)
    assert len(dvec) == 512
    #print time.time()-etime
    #exit(0)
    return dvec

def process(files, saveFiles):
    check_file(files)
    dataDict = pickle.load(open(files, 'rb'))
    mean = pickle.load(open('../../data/far/train/train.mean', 'rb'))
    std = pickle.load(open('../../data/far/train/train.std', 'rb'))
    eps = 1e-8

    dataDvec = {}
    for utt in dataDict:
        uttData = dataDict[utt]
        if uttData is None:
            raise IOError('[{:s}] uttData is None'.format(str(utt)))
        data = []
        index = 0
        frame = 40
        step = 10
        while index + frame <= uttData.shape[0]:
            tmpData = uttData[index:index+frame, :]
            tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
            data.append(tmpData)
            index += step
        data = np.asarray(data)
        #data = (data-mean)/(std+eps)
        dvec = extract_dvector(net, data)
        dataDvec[utt] = dvec
    pickle.dump(dataDvec, open(saveFiles, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    print '----------------------------------------------------------------'
    '''
    enroData = '../../data/far/enro_far_1208/enro_far_1208.dict'
    enroDvec = '../../data/far/enro_far_1208/enro_far_1208.dvec'
    testData = '../../data/far/test_far_1208/test_far_1208.dict'
    testDvec = '../../data/far/test_far_1208/test_far_1208.dvec'
    '''
    #'''
    enroData = '../../data/far/enro_far_1214/enro_far_1214.dict'
    enroDvec = '../../data/far/enro_far_1214/enro_far_1214.dvec'
    testData = '../../data/far/test_far_1214/test_far_1214.dict'
    testDvec = '../../data/far/test_far_1214/test_far_1214.dvec'
    #'''
    '''
    enroData = '../../data/near/enro_near/enro_near.dict'
    enroDvec = '../../data/near/enro_near/enro_near.dvec'
    testData = '../../data/near/test_near/test_near.dict'
    testDvec = '../../data/near/test_near/test_near.dvec'
    '''
    print 'Load model'
    #net = torch.load('../../model/far/resnet18-/resnet18.14.model')
    net = torch.load('../../model/far/resnet50-normal-resnet50-normal-2017-12-19 14:45:26/resnet50.49.model')
    print net
    #net = torch.load('../../model/far/resnet18-2017-12-14 18:07:47/resnet.37.model')
    net.cuda()

    process(enroData, enroDvec)
    process(testData, testDvec)

    print '[Done]'



