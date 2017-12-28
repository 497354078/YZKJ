import sys
sys.path.append('../')
sys.path.append('../src_module/')
from general import *
from model_resnet34 import resnet34
from train_model import train_model
from valid_model import valid_model
from load_data_train_vad_accent import LoadModelDataVAD

def main(args):
    rePrint('-------------------------------------------------------------------------------------')
    vadDict = pickle.load(open(args.vad_file, 'rb'))

    modelPath = os.path.join(args.model_path, args.model+args.fea_type)
    make_path(modelPath)

    rePrint('  [Create model: {:s} num_classes: {:d}]'.format(args.model, args.num_classes))
    net = eval(args.model)(num_classes=args.num_classes)
    if args.load != -1:
        load_file = os.path.join(modelPath, args.model_file).format(args.load)
        rePrint('  [Reload model from: {:s}]'.format(load_file))
        net.load_state_dict(torch.load(load_file))
    net = net.cuda()


    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=args.factor,
                                  patience=args.patience,
                                  verbose=False, threshold=1e-6,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

    rePrint('')
    rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [Train {:s}]'.format(args.model))
    rePrint('-------------------------------------------------------------------------------------')

    for epoch in range(args.load+1, args.epochs):
        #print type(epoch), epoch
        save_file = os.path.join(modelPath, args.model_file).format(epoch)

        for splitID in range(args.split_num):
            train_file = os.path.join(args.train_path, args.train_file.format(splitID))
            rePrint('  [LoadModelData {:s}]'.format(train_file))
            #trainData = LoadModelData(train_file, 'train')
            trainData = LoadModelDataVAD(train_file, vadDict, args.alpha, 'train')
            trainLoad = torch.utils.data.DataLoader(trainData,
                                batch_size=args.batch_size, shuffle=True, num_workers=4)

            if splitID != args.split_num-1:
                net = train_model(net=net,
                            trainLoad=trainLoad,
                            optimizer=optimizer,
                            log_interval=args.log_interval,
                            epoch=epoch,
                            batch_size=args.batch_size,
                            lr=optimizer.param_groups[0]['lr'],
                            save_file=None)
            else:
                eval_loss = valid_model(net=net, validLoad=trainLoad)
                scheduler.step(eval_loss)

        torch.save(net.state_dict(), save_file)
        rePrint(save_file)    



        rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [Done]')
    rePrint('-------------------------------------------------------------------------------------')


if __name__ == '__main__':
    print '-------------------------------------------------------------------------------------'
    # create argparse
    parser = argparse.ArgumentParser(description='PyTorch For Far Voice Verification')

    # data setting
    feaType = 'far-accent-speech-novad'
    parser.add_argument('--split_num',  type=int, default=5,     help='number of split')
    parser.add_argument('--fea_type',   type=str, default=feaType,help='data to use')
    parser.add_argument('--train_path', type=str, default='/home/lj/work/voice/data/{:s}/train'.format(feaType),
                                        help='train files for model')
    parser.add_argument('--train_file', type=str, default='train.{:d}.dict',
                                        help='train files to load')
    parser.add_argument('--train_mean', type=str, default='/home/lj/work/voice/data/{:s}/train/train.mean'.format(feaType),
                                        help='train mean files for model')
    parser.add_argument('--train_std',  type=str, default='/home/lj/work/voice/data/{:s}/train/train.std'.format(feaType),
                                        help='train std files for model')
    parser.add_argument('--valid_path', type=str, default='/home/lj/work/voice/data/{:s}/train'.format(feaType),
                                        help='valid files for model')
    parser.add_argument('--valid_file', type=str, default='train.{:d}.dict',
                                        help='valid files to load')
    parser.add_argument('--vad_file',   type=str, default='/home/lj/work/voice/data/{:s}/train/vad.dict'.format(feaType),
                                        help='vad files to load')
    parser.add_argument('--alpha',      type=float,default=0.5,     help='thread for vad')
    parser.add_argument('--model_path', type=str, default='/home/lj/work/voice/model',
                                        help='model files to save')
    parser.add_argument('--model_file', type=str, default='epoch.{:d}.model',
                                        help='model files to save')
    parser.add_argument('--num_classes',type=int,   default=2426,     help='number of classes')
    parser.add_argument('--model',      type=str,   default='resnet34',help='model to use')


    # model parameter setting
    parser.add_argument('--load',       type=int,   default=-1,     help='reload model (default: -1)')
    parser.add_argument('--epochs',     type=int,   default=50,     help='number of epochs')
    parser.add_argument('--lr',         type=float, default=0.01,   help='learning rate')
    parser.add_argument('--factor',     type=float, default=0.7,    help='learning decay factor')
    parser.add_argument('--patience',   type=int,   default=7,      help='learning patience')
    parser.add_argument('--momentum',   type=float, default=0.9,    help='momentum')
    parser.add_argument('--weight_decay',type=float,default=1e-6,   help='weight_decay')
    parser.add_argument('--batch_size', type=int,   default=512,    help='input batch size')
    parser.add_argument('--log_interval',type=int,  default=200,    help='show batches')
    parser.add_argument('--seed',       type=int,   default=1,      help='random seed (default: 1)')

    # gpu setting
    parser.add_argument('--gpu', type=str, default='4', help='gpu to use (default: 7)')

    args = parser.parse_args()

    make_path('log')
    timeMark = ''#str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    logName = 'train-{:s}-{:s}'.format(args.model, args.fea_type)+timeMark
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

    

