import sys
sys.path.append('../')
from util import *

def extract_dvector(net, data):
    net.eval()

    data = torch.from_numpy(data).type(torch.FloatTensor)
    data = Variable(data.cuda())

    # Forward
    embedd, _ = net(data)

    embedd = embedd.data.cpu().numpy()
    assert embedd.shape[1] == 128

    return embedd

def process(files):
    rePrint('----------------------------------------------------------------')
    rePrint('Load Dataset from [{:s}]'.format(files))
    check_file(files)
    dataDict = pickle.load(open(files, 'rb'))
    embeddDict = {}

    for audioID in dataDict:
        data = []
        labs = []
        idx = []
        for subData in dataDict[audioID]:
            if subData is None:
                raise IOError('spk: [{:s}] data is None'.format(audioID))
            index = 0
            frame = 65
            step = 1
            assert subData.shape[0] == 64
            count = 0
            while index + frame <= subData.shape[1]:
                tmpData = subData[:, index:index+frame]
                tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                data.append(tmpData)
                labs.append(audioID)
                index += step
            idx.append(count)
            print count,
        print ''

        data = np.asarray(data)
        shape = data.shape
        ttime = time.time()
        batchDvec = extract_dvector(net, data)
        etime = time.time()

        embeddDict[audioID] = []
        spoint = 0
        for count in idx:
            embedd = batchDvec[spoint:spoint+count, :]
            embeddDict[audioID].append(embedd.T)
            spoint += count
            print embedd.shape[0],
        print ''
        exit(0)
    rePrint('')

if __name__ == '__main__':
    print '-----------------------------------------------------------------'
    timeMark = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    #timeMark = '2017-12-14 18:07:47'
    make_path('log')
    logName = os.path.basename(sys.argv[0])+'-'+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s : %(message)s')

    rePrint('-----------------------------------------------------------------')

    fold = 'fold0'
    dataPath = '../../data/data_ESC-10/{:s}'.format(fold)
    modelPath = '../../model/model_ESC-10/{:s}'.format(fold)
    modelFile = 'Basic_CNN.{:d}.model'
    make_path(modelPath)
    net = torch.load(os.path.join(modelPath, modelFile.format(26)))
    net.cuda()

    process(os.path.join(dataPath, 'train.dict'))


    rePrint('[Done]')
    rePrint('-----------------------------------------------------------------')




