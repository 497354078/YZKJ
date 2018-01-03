import sys
sys.path.append('../')
sys.path.append('../src_module/')
from general import *
from model_resnet34 import resnet34
from model_resnet35 import resnet35
from model_resnet18 import resnet18
from model_resnet19 import resnet19

from make_data_trials import process as make_data_trials
#from make_data_kaldi import process as make_data_kaldi
#from make_data_speech import process as make_data_speech
from extract_trials_dvector import process as extract_dvector
from eval_trials_dvector import eval_dvector, get_enro_dvector

def main(args):
    
    modelPath = os.path.join(args.model_path, args.model+args.fea_type)
    load_file = os.path.join(modelPath, args.model_file).format(args.load)
    rePrint('  [Reload model from: {:s}]'.format(load_file))
    net = eval(args.model)(num_classes=args.num_classes)
    net.load_state_dict(torch.load(load_file))
    net = net.cuda()

    args.enro_path = os.path.join(args.base_path, 'enro')
    args.test_path = os.path.join(args.base_path, 'test')
    make_data_trials(feaType=args.use_feat, basePath=args.enro_path, mode=args.enro_mode, 
                     feaPath=args.fea_path, savePath=args.save_path, FLAG=args.use_vad)
    make_data_trials(feaType=args.use_feat, basePath=args.test_path, mode=args.test_mode,
                     feaPath=args.fea_path, savePath=args.save_path, FLAG=args.use_vad)
    rePrint('')
    
    enroFile = os.path.join(args.save_path, args.enro_mode, args.enro_mode+'.dict')
    testFile = os.path.join(args.save_path, args.test_mode, args.test_mode+'.dict')
    enroVad = os.path.join(args.save_path, args.enro_mode, 'vad.dict')
    testVad = os.path.join(args.save_path, args.test_mode, 'vad.dict')
    
    enroDvecFile = enroFile.replace('.dict', '.dvec')
    testDvecFile = testFile.replace('.dict', '.dvec')
    extract_dvector(net, enroFile, enroDvecFile, enroVad, args.alpha, args.use_vad)
    extract_dvector(net, testFile, testDvecFile, testVad, args.alpha, args.use_vad)
    rePrint('')

    enroDvecDict = pickle.load(open(enroDvecFile, 'rb'))
    testDvecDict = pickle.load(open(testDvecFile, 'rb'))
    args.trials = os.path.join(args.base_path, args.trials)
    eval_dvector(get_enro_dvector(args.base_path, enroDvecDict, args.dims),
                 testDvecDict, args.trials, args.dims)
    rePrint('')
    rePrint('[Done]')


if __name__ == '__main__':
    print '-------------------------------------------------------------------------------------'
    # create argparse
    parser = argparse.ArgumentParser(description='PyTorch For Far Voice Verification')

    # data setting
    feaType = 'far-accent-speech-novad'
    enroType = 'xytx_far_1214_volumeup'
    #enroType = 'xiaoyutongxue'
    parser.add_argument('--use_vad',    default=False, action='store_true', help='vad to use, default 0.5vad')
    parser.add_argument('--use_feat',   type=str, default='speech',  help='feat to use')
    parser.add_argument('--fea_type',   type=str, default=feaType,help='data to use')
    parser.add_argument('--fea_path',   type=str, default='../../logfbank',help='data to use')

    parser.add_argument('--enro_path',  type=str, default='enro'.format(enroType),
                                        help='enro path')
    parser.add_argument('--test_path',  type=str, default='test'.format(enroType),
                                        help='test path')
    parser.add_argument('--enro_mode',  type=str, default='enro_'+enroType, help='enro_mode')
    parser.add_argument('--test_mode',  type=str, default='test_'+enroType, help='test_mode')

    parser.add_argument('--save_path',  type=str, default='../../data/{:s}'.format(feaType),
                                        help='save path to data')

    parser.add_argument('--alpha',      type=float,default=0.5,     help='thread for vad')

    # trials setting
    parser.add_argument('--dims',       type=int,default=512,     help='dims for dvector')
    parser.add_argument('--trials',     type=str,
                        default='test/trials'.format(enroType),
                        help='enro_mode')
    parser.add_argument('--base_path',  type=str,
                        default='/aifs1/users/kxd/sre/data/test/{:s}'.format(enroType),
                        help='save path to data')

    # model parameter setting
    parser.add_argument('--load',       type=int,   default=-1,     help='reload model (default: -1)')

    parser.add_argument('--model_path', type=str, default='/home/lj/work/voice/model',
                                        help='model files to save')
    parser.add_argument('--model_file', type=str, default='epoch.{:d}.model',
                                        help='model files to save')
    parser.add_argument('--num_classes',type=int,   default=8470,     help='number of classes')
    parser.add_argument('--model',      type=str,   default='resnet34',help='model to use')


    # gpu setting
    parser.add_argument('--gpu', type=str, default='0', help='gpu to use (default: 7)')

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



