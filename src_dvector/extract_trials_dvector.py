import sys
sys.path.append('../')
sys.path.append('../src_module/')
from general import *
from model_resnet34 import resnet34
#from train_model import train_model
#from valid_model import valid_model
#from load_data_train_vad import LoadModelDataVAD
#from load_data_train import LoadModelData

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

def process(net, files, saveFiles, vadFiles, alpha, FLAG):
    print ''
    print '======= [extract dvector] ====================================='
    print 'loadFiles: [{:s}]'.format(files)
    print 'saveFiles: [{:s}]'.format(saveFiles)
    print 'vadFiles : [{:s}]'.format(vadFiles)
    print 'alpha: [{:s}]  FLAG: [{:s}]'.format(str(alpha), str(FLAG))
    print ''
    check_file(files)
    dataDict = pickle.load(open(files, 'rb'))

    check_file(vadFiles)
    vadDict = pickle.load(open(vadFiles, 'rb'))

    dataDvec = {}
    for spkID in dataDict:
        for uttName, uttData in dataDict[spkID]:
            if uttData is None:
                #raise IOError('spk: [{:s}] data is None'.format(str(spkID)))
                #rePrint('[spk:{:s}] [spkName:{:s}] data is None'.format(str(spkID), uttName))
                continue
            vad = vadDict[uttName]
            data = uttData
            comp = data.shape[0] == len(vad) or data.shape[0]-1 == len(vad) or data.shape[0]-2 == len(vad)
            if FLAG == True: assert not comp
            if FLAG == False:assert comp
            uttData = (uttData-uttData.mean(axis=0)) / (uttData.std(axis=0)+eps)
            index = 0
            frame = 40
            step = 20
            data = []
            while index + frame <= uttData.shape[0]:
                #print uttName, index, index+frame, alpha
                #print type(vadDict[uttName])
                if FLAG == True or 1.0*sum(vadDict[uttName][index:index+frame])/frame >= alpha:
                    tmpData = uttData[index:index+frame, :]
                    tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                    data.append(tmpData)
                index += step
            if len(data) == 0:
                #rePrint('[{:s}] [spk:{:s}] [spkName:{:s}] data is None'.format(str(alpha), str(spkID), uttName))
                continue
            data = np.asarray(data)
            dvec = extract_dvector(net, data)
            dataDvec[uttName] = dvec
    pickle.dump(dataDvec, open(saveFiles, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    print '----------------------------------------------------------------'
    feaType = 'far-am-speech'
    alpha = 0.5
    modelID = 7
    modelName = 'resnet34'
    modelFile = '../../model/{:s}{:s}/epoch.{:d}.model'.format(modelName, feaType, modelID)
    print 'Load model '
    net = resnet34(num_classes=8470) # 2426 6044 8470
    net.load_state_dict(torch.load(modelFile))
    net = net.cuda()

    #enroData = '../../data/{:s}/enro_far_1208/enro_far_1208.dict'.format(feaType)
    #enroDvec = '../../data/{:s}/enro_far_1208/enro_far_1208.dvec'.format(feaType)
    #testData = '../../data/{:s}/test_far_1208/test_far_1208.dict'.format(feaType)
    #testDvec = '../../data/{:s}/test_far_1208/test_far_1208.dvec'.format(feaType)

    enroData = '../../data/{:s}/enro_far_1214/enro_far_1214.dict'.format(feaType)
    enroDvec = '../../data/{:s}/enro_far_1214/enro_far_1214.dvec'.format(feaType)
    enroVAD  = '../../data/{:s}/enro_far_1214/vad.dict'.format(feaType)
    testData = '../../data/{:s}/test_far_1214/test_far_1214.dict'.format(feaType)
    testDvec = '../../data/{:s}/test_far_1214/test_far_1214.dvec'.format(feaType)
    testVAD  = '../../data/{:s}/test_far_1214/vad.dict'.format(feaType)
    process(net, enroData, enroDvec, enroVAD, alpha, True)
    process(net, testData, testDvec, testVAD, alpha, True)

    print '[Done]'




