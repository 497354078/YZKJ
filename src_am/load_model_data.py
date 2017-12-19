import sys
sys.path.append('../')
from util import *

class LoadModelData(Dataset):
    def __init__(self, files, mode):
        self.eps = 1e-8

        self.data, self.labs = self.load_data(files, mode)

        self.mean = pickle.load(open('../../data/far/train/train.mean', 'rb'))
        self.std = pickle.load(open('../../data/far/train/train.std', 'rb'))


    def __len__(self):
        return len(self.labs)

    def __getitem__(self, item):
        #return self.data[item], self.labs[item]
        return (self.data[item]-self.mean)/(self.std+self.eps), self.labs[item]

    def load_data(self, files, mode):
        print '----------------------------------------------------------------'
        check_file(files)
        dataDict = pickle.load(open(files, 'rb'))
        print 'Load Dataset from [{:s}] {:d}'.format(files, len(dataDict))

        data = []
        labs = []
        for spkID in dataDict:
            for uttName, uttData in dataDict[spkID]:
                if uttData is None:
                    raise IOError('spk: [{:s}] data is None'.format(str(spkID)))
                    #continue
                uttData = (uttData-uttData.mean(axis=0))/(uttData.std(axis=0)+self.eps)
                index = 0
                frame = 40
                step = 10
                while index + frame <= uttData.shape[0]:
                    tmpData = uttData[index:index+frame, :]
                    tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                    data.append(tmpData)
                    labs.append(spkID)
                    index += step
        if mode == 'train':
            mean = np.asarray(data).mean(axis=0)
            std = np.asarray(data).std(axis=0)
            pickle.dump(mean, open(files.replace('.dict', '.mean'), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(std, open(files.replace('.dict', '.std'), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

        print 'data: ', len(data), data[0].shape
        print 'labs: ', len(labs)
        print ''
        return data, labs


if __name__ == '__main__':
    print '----------------------------------------------------------------'

    trainFile = '../../data/far/train/train.dict'
    validFile = '../../data/far/train/valid.dict'

    trainData = LoadModelData(trainFile, 'train')
    trainLoad = torch.utils.data.DataLoader(trainData, batch_size=512, shuffle=True, num_workers=8)

    for data, labs in trainLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

    validData = LoadModelData(validFile, 'valid')
    validLoad = torch.utils.data.DataLoader(validData, batch_size=512, shuffle=True, num_workers=8)

    for data, labs in validLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break


