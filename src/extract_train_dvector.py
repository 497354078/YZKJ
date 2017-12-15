import sys
sys.path.append('../')
from util import *

def extract_dvector(net, data):
    net.eval()

    data = torch.from_numpy(data).type(torch.FloatTensor)
    data = Variable(data.cuda())

    # Forward
    dvec, _ = net(data)

    dvec = dvec.data.cpu().numpy()
    #dvec = dvec.mean(axis=0)
    assert dvec.shape[1] == 512

    return dvec

def process(files, saveFiles):
    check_file(files)
    dataDict = pickle.load(open(files, 'rb'))
    dataDvec = {}
    spkCnt = 0
    uttCnt = 0
    stime = time.time()
    for spkID in dataDict:
        if len(dataDict[spkID]) < 10:
            #raise IOError('len(dataDict[spkID:{:d}]) == 0'.format((spkID)))
            continue
        dataDvec[spkID] = []
        uttIndex = []
        data = []
        for uttData in dataDict[spkID]:
            if uttData is None:
                raise IOError('[{:s}] uttData is None'.format(str(spkID)))
            if uttData.shape[0] < 80:
                raise IOError('uttData.shape[0] < 80')
            index = 0
            frame = 40
            step = 20
            count = 0
            while index + frame <= uttData.shape[0]:
                tmpData = uttData[index:index+frame, :]
                tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                data.append(tmpData)
                index += step
                count += 1
            uttIndex.append(count)
        assert sum(uttIndex) == len(data)
        data = np.asarray(data)
        shape = data.shape

        ttime = time.time()
        batchDvec = extract_dvector(net, data)
        etime = time.time()

        spoint = 0
        for count in uttIndex:
            dvec = batchDvec[spoint:spoint+count, :]
            dvec = dvec.mean(axis=0)
            assert len(dvec) == 512
            dataDvec[spkID].append(dvec)
            spoint += count
            uttCnt += 1
        spkCnt += 1

        print spkID, shape, etime-ttime, time.time()-etime, time.time()-stime

    print 'spkCnt: ', spkCnt, 'uttCnt: ', uttCnt
    pickle.dump(dataDvec, open(saveFiles, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    print '----------------------------------------------------------------'

    trainData = '../data/far/train/train.dict'
    trainDvec = '../data/far/train/train.dvec'



    net = torch.load('../model/far/resnet-1513165808/resnet.49.model')
    net.cuda()

    process(trainData, trainDvec)

    print '[Done]'


