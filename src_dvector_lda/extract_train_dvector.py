import sys
sys.path.append('../')
sys.path.append('../src_module/')
from general import *
from model_resnet34 import resnet34
from model_resnet35 import resnet35
from model_resnet18 import resnet18
from model_resnet19 import resnet19

def extract_embedding(net, data):
    net.eval()
    data = torch.from_numpy(data).type(torch.FloatTensor)
    data = Variable(data.cuda())

    # Forward
    dvec, _ = net(data)

    dvec = dvec.data.cpu().numpy()
    #dvec = dvec.mean(axis=0)

    return dvec

def main(args):

    modelPath = os.path.join(args.model_path, args.model+args.fea_type)
    load_file = os.path.join(modelPath, args.model_file).format(args.load)
    rePrint('  [Reload model from: {:s}]'.format(load_file))
    net = eval(args.model)(num_classes=args.num_classes)
    net.load_state_dict(torch.load(load_file))
    net = net.cuda()

    vadDict = pickle.load(open(args.vad_file, 'rb'))
    dvecData = []
    dvecLabs = []
    dvecIdx = []
    for splitID in range(args.split_num-1):
        train_file = os.path.join(args.train_path, args.train_file.format(splitID))
        rePrint('  [LoadModelData {:s}]'.format(train_file))
        dataDict = pickle.load(open(train_file, 'rb'))
        for spkID in dataDict:
            data = []
            labs = []
            for uttName, uttData in dataDict[spkID]:
                if uttData is None:
                    continue
                uttData = (uttData-uttData.mean(axis=0))/(uttData.std(axis=0)+eps)
                index = 0
                frame = 40
                step = 20
                count = 0
                while index + frame <= uttData.shape[0]:
                    if 1.0*sum(vadDict[uttName][index:index+frame])/frame >= args.alpha:
                        tmpData = uttData[index:index+frame, :]
                        tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                        data.append(tmpData)
                        labs.append(spkID)
                        count += 1
                    index += step
                dvecIdx.append(count)
            if len(data) == 0:
                continue
            data = np.asarray(data).astype(np.float32)
            dvec = extract_embedding(net, data)
            print 'spkID: ', spkID, 'data: ', data.shape, 'dvec: ', dvec.shape, '\t\r',
            sys.stdout.flush()

            #pickle.dump(dvec, open(train_file.replace('.dict', '.dvec'), 'wb'),
            #            protocol=pickle.HIGHEST_PROTOCOL)

            for i in range(dvec.shape[0]):
                dvecData.append(dvec[i, :])
                dvecLabs.append(labs[i])
        print ''

    dvecData = np.asarray(dvecData)
    dvecLabs = np.asarray(labs)
    dvecIdx = np.asarray(dvecIdx)
    print 'dvecData: ', dvecData.shape
    print 'dvecLabs: ', dvecLabs.shape
    print 'dvecIdx: ', dvecIdx.shape
    pickle.dump(dvecData, open(os.path.join(args.train_path, 'dvecData.numpy'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dvecLabs, open(os.path.join(args.train_path, 'dvecLabs.numpy'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dvecIdx, open(os.path.join(args.train_path, 'dvecIdx.numpy'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    print '-------------------------------------------------------------------------------------'
    # create argparse
    parser = argparse.ArgumentParser(description='PyTorch For Far Voice Verification')

    # data setting
    feaType = 'far-accent-speech-novad'
    parser.add_argument('--fea_type',   type=str, default=feaType,help='data to use')
    parser.add_argument('--fea_path',   type=str, default='../../logfbank',help='data to use')

    parser.add_argument('--split_num',  type=int, default=5,     help='number of split')
    parser.add_argument('--train_path', type=str, default='/home/lj/work/voice/data/{:s}/train'.format(feaType),
                                        help='train files for model')
    parser.add_argument('--train_file', type=str, default='train.{:d}.dict',
                                        help='train files to load')

    parser.add_argument('--save_path',  type=str, default='../../data/{:s}'.format(feaType),
                                        help='save path to data')

    parser.add_argument('--alpha',      type=float,default=0.5,     help='thread for vad')
    parser.add_argument('--vad_file',   type=str, default='/home/lj/work/voice/data/{:s}/train/vad.dict'.format(feaType),
                                        help='vad files to load')

    # model parameter setting
    parser.add_argument('--load',       type=int,   default=-1,     help='reload model (default: -1)')

    parser.add_argument('--model_path', type=str, default='/home/lj/work/voice/model',
                                        help='model files to save')
    parser.add_argument('--model_file', type=str, default='epoch.{:d}.model',
                                        help='model files to save')
    parser.add_argument('--num_classes',type=int,   default=2426,     help='number of classes')
    parser.add_argument('--model',      type=str,   default='resnet34',help='model to use')


    # gpu setting
    parser.add_argument('--gpu', type=str, default='4', help='gpu to use (default: 7)')

    args = parser.parse_args()

    make_path('log')
    timeMark = ''#str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    logName = 'test-{:s}-{:s}'.format(args.model, args.fea_type)+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s: %(message)s',
                    datefmt='%a %d/%b/%Y %H:%M:%S',)

    rePrint('')
    rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [log save in {:s}.log]'.format(logName))
    rePrint('  [{:s}]'.format(args))
    rePrint('')

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    main(args)




