import sys
sys.path.append('../')
from general import *


class LoadModelDataVAD(Dataset):
    def __init__(self, files, vadDict, alpha, mode):
        self.data, self.labs = self.load_data(files, vadDict, alpha, mode)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, item):
        return self.data[item], self.labs[item]

    def load_data(self, files, vadDict, alpha, mode):
        #rePrint('----------------------------------------------------------------')
        #rePrint('  Load Dataset from [{:s}] & mode:{:s}'.format(files, mode))
        stime = time.time()
        check_file(files)
        dataDict = pickle.load(open(files, 'rb'))
        #rePrint('  Load Dataset [usetime: {:f}] & [length: {:d}] & slip-window-sample...'.format(
        #         time.time()-stime, len(dataDict)))
        stime = time.time()
        data = []
        labs = []
        for spkID in dataDict:
            for uttName, uttData in dataDict[spkID]:
                if uttData is None:
                    #raise IOError('spk: [{:s}] data is None'.format(str(spkID)))
                    #rePrint('[spk:{:s}] [spkName:{:s}] data is None'.format(str(spkID), uttName))
                    continue
                uttData = (uttData-uttData.mean(axis=0))/(uttData.std(axis=0)+eps)
                index = 0
                frame = 40
                step = 20
                while index + frame <= uttData.shape[0]:
                    if 1.0*sum(vadDict[uttName][index:index+frame])/frame >= alpha:
                        tmpData = uttData[index:index+frame, :]
                        tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                        data.append(tmpData)
                        labs.append(spkID)
                    index += step
        #rePrint('  Slip-window-sample usetime: {:f}'.format(time.time()-stime))
        #rePrint('  [data: {:d}  shape: {:s}]'.format(len(data), data[0].shape))
        #rePrint('  [labs: {:d}  minVal:{:d}  maxVal:{:d}'.format(len(labs), min(labs), max(labs)))
        #rePrint('')
        return data, labs


if __name__ == '__main__':
    print '----------------------------------------------------------------'
    featType = 'far-mandarin-speech-novad' # speech & kaldi
    splitNum = 5
    splitID = 0
    alpha = 0.5
    trainFile = '../../data/{:s}/train/train.{:d}.dict'.format(featType, splitID)
    validFile = '../../data/{:s}/train/train.{:d}.dict'.format(featType, splitNum-1)
    vadFile = '../../data/{:s}/train/vad.dict'.format(featType)
    vadDict = pickle.load(open(vadFile, 'rb'))

    trainData = LoadModelDataVAD(trainFile, vadDict, alpha, 'train')
    trainLoad = torch.utils.data.DataLoader(trainData, batch_size=512, shuffle=True, num_workers=4)

    for data, labs in trainLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

    validData = LoadModelDataVAD(validFile, vadDict, alpha, 'valid')
    validLoad = torch.utils.data.DataLoader(validData, batch_size=512, shuffle=True, num_workers=4)

    for data, labs in validLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

    print ''



