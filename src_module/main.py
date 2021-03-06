from general import *
from model_resnet34 import resnet34
from train_model import train_model
from valid_model import valid_model
from load_data_train import LoadModelData

def main(args):
    rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [LoadModelData {:s}]'.format(args.train_file))
    trainData = LoadModelData(args.train_file, 'train')
    trainLoad = torch.utils.data.DataLoader(trainData,
                        batch_size=args.batch_size, shuffle=True, num_workers=4)

    rePrint('  [LoadModelData {:s}]'.format(args.valid_file))
    validData = LoadModelData(args.valid_file, 'valid')
    validLoad = torch.utils.data.DataLoader(validData,
                        batch_size=args.batch_size, shuffle=False, num_workers=4)

    rePrint('  [Create model: {:s} num_classes: {:d}]'.format(args.model, args.num_classes))
    net = eval(args.model)(num_classes=args.num_classes)
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
    make_path(os.path.join(args.model_path, args.model))
    for epoch in range(args.epochs):
        save_file = os.path.join(args.model_path, args.model, args.model_file).format(epoch)
        train_model(net=net,
                    trainLoad=trainLoad,
                    optimizer=optimizer,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    batch_size=args.batch_size,
                    lr=optimizer.param_groups[0]['lr'],
                    save_file=save_file)
        eval_loss = valid_model(net=net, validLoad=validLoad)
        scheduler.step(eval_loss)
        rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [Done]')
    rePrint('-------------------------------------------------------------------------------------')


if __name__ == '__main__':
    print '-------------------------------------------------------------------------------------'
    # create argparse
    parser = argparse.ArgumentParser(description='PyTorch For Far Voice Verification')

    # data setting
    parser.add_argument('--train_file', type=str, default='/home/lj/work/voice/data/far/train/train.dict',
                                        help='train files for model')
    parser.add_argument('--train_mean', type=str, default='/home/lj/work/voice/data/far/train/train.mean',
                                        help='train mean files for model')
    parser.add_argument('--train_std',  type=str, default='/home/lj/work/voice/data/far/train/train.std',
                                        help='train std files for model')
    parser.add_argument('--valid_file', type=str, default='/home/lj/work/voice/data/far/train/valid.dict',
                                        help='valid files for model')
    parser.add_argument('--model_path', type=str, default='/home/lj/work/voice/model/far',
                                        help='model files to save')
    parser.add_argument('--model_file', type=str, default='model.{:s}.pkl',
                                        help='model files to save')
    parser.add_argument('--num_classes',type=int,   default=8470,     help='number of classes')
    parser.add_argument('--model',      type=str,   default='resnet34',help='model to use')


    # model parameter setting
    parser.add_argument('--epochs',     type=int,   default=50,     help='number of epochs')
    parser.add_argument('--lr',         type=float, default=0.01,   help='learning rate')
    parser.add_argument('--factor',     type=float, default=0.7,    help='learning decay factor')
    parser.add_argument('--patience',   type=int,   default=7,      help='learning patience')
    parser.add_argument('--momentum',   type=float, default=0.9,    help='momentum')
    parser.add_argument('--weight_decay',type=float,default=1e-6,   help='weight_decay')
    parser.add_argument('--batch_size', type=int,   default=256,    help='input batch size')
    parser.add_argument('--log_interval',type=int,  default=500,    help='show batches')
    parser.add_argument('--seed',       type=int,   default=1,      help='random seed (default: 1)')

    # gpu setting
    parser.add_argument('--gpu', type=str, default='7', help='gpu to use (default: 7)')

    args = parser.parse_args()

    make_path('log')
    timeMark = ''#str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    logName = 'train-{:s}'.format(args.model)+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s: %(message)s')

    rePrint('-------------------------------------------------------------------------------------')
    rePrint('  [log save in {:s}.log]'.format(logName))
    rePrint('  [{:s}]'.format(args))
    rePrint('')

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    main(args)

