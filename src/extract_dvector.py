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
    dataDvec = {}
    for spk in dataDict:
        dataDvec[spk] = []
        for uttData in dataDict[spk]:
            if uttData is None:
                raise IOError('[{:s}] uttData is None'.format(str(spk)))
            data = []
            index = 0
            frame = 40
            step = 20
            while index + frame <= uttData.shape[0]:
                tmpData = uttData[index:index+frame, :]
                tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                data.append(tmpData)
                index += step
            data = np.asarray(data)
            dvec = extract_dvector(net, data)
            dataDvec[spk].append(dvec)
    pickle.dump(dataDvec, open(saveFiles, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    print '----------------------------------------------------------------'

    enroData = '../data/far/enroFar/enroFar.dict'
    enroDvec = '../data/far/enroFar/enroFar.dvec'
    testData = '../data/far/testFar/testFar.dict'
    testDvec = '../data/far/testFar/testFar.dvec'
    
    #trainData = '../data/far/train/train.dict'
    #trainDvec = '../data/far/train/train.dvec'

    #net = torch.load('../model/far/resnet-1513165808/resnet.49.model')
    net = torch.load('../model/far/resnet18/resnet.36.model')
    net.cuda()

    process(enroData, enroDvec)
    process(testData, testDvec)
    #process(trainData, trainDvec)
    
    print '[Done]'


