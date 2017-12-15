import sys
sys.path.append('../')

from util import *

class LoadModelData(Dataset):
    def __init__(self, files):
        self.data, self.labs = self.load_data(files)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, item):
        return self.data[item], self.labs[item]

    def load_data(self, files):
        print '----------------------------------------------------------------'
        check_file(files)
        dataDict = pickle.load(open(files, 'rb'))
        print 'Load Dataset from [{:s}] {:d}'.format(files, len(dataDict))

        data = []
        labs = []
        for spkID in dataDict:
            for uttData in dataDict[spkID]:
                if uttData is None:
                    raise IOError('spk: [{:s}] data is None'.format(str(spkID)))
                    #continue
                index = 0
                frame = 40
                step = 20
                while index + frame <= uttData.shape[0]:
                    tmpData = uttData[index:index+frame, :]
                    tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                    data.append(tmpData)
                    labs.append(spkID)
                    index += step

        print 'data: ', len(data), data[0].shape
        print 'labs: ', len(labs)
        print ''
        return data, labs


if __name__ == '__main__':
    print '----------------------------------------------------------------'

    trainFile = '../data/far/train/train.dict'
    validFile = '../data/far/train/valid.dict'

    trainData = LoadModelData(trainFile)
    trainLoad = torch.utils.data.DataLoader(trainData, batch_size=512, shuffle=True, num_workers=8)

    for data, labs in trainLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

    validData = LoadModelData(validFile)
    validLoad = torch.utils.data.DataLoader(validData, batch_size=512, shuffle=True, num_workers=8)

    for data, labs in validLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

