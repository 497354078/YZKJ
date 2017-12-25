from general import *
from model_resnet18 import resnet18
from model_resnet34 import resnet34
from model_resnet50 import resnet50
from extract_trials_dvector import extract_dvector
from eval_trials_dvector import get_enro_dvector, eval_dvector

def main(args):
    pass

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
    parser.add_argument('--gpu', type=str, default='6', help='gpu to use (default: 6)')

    args = parser.parse_args()

    make_path('log')
    timeMark = ''#str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    logName = 'eval-{:s}'.format(args.model)+timeMark
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

