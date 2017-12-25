from general import *
from model_resnet18 import resnet18
from model_resnet34 import resnet34
from model_resnet50 import resnet50

def feed_network(net, data):
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

    #print time.time()-etime
    #exit(0)
    return dvec

def extract_dvector(net, files, save_files):
    rePrint('  [extract_dvector...]')
    check_file(files)
    dataDict = pickle.load(open(files, 'rb'))

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
        dvec = feed_network(net, data)
        assert len(dvec) == 512
        dataDvec[utt] = dvec
    pickle.dump(dataDvec, open(save_files, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    rePrint('  [dvector save in {:s}]'.format(save_files))
    rePrint('')


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
    #trainData = '../data/far/train/train.dict'
    #trainDvec = '../data/far/train/train.dvec'

    net = resnet34(num_classes=8470)
    net.load_state_dict(torch.load('../../model/resnet34far-am-kaldi/epoch.47.model'))
    print type(net)
    #print net
    net.cuda()

    extract_dvector(net, enroData, enroDvec)
    extract_dvector(net, testData, testDvec)

    print '[Done]'

