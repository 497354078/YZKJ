import sys
sys.path.append('../')
from util import *

def audio_normal(mode):
    make_path(os.path.join(destPath, mode, 'audio'))
    newscp = os.path.join(destPath, mode, 'wav.scp')
    fw = open(newscp, 'wb')

    wavscp = os.path.join(basePath, mode, 'wav.scp')
    lines = open(wavscp, 'rb').readlines()

    for items in lines:
        items = items.split('\n')[0].split(' ')
        spkName = items[0]
        wavFile = items[1]
        #y, sr = librosa.load(wavFile, sr=16000, mono=True)
        #y_ = y / y.max()
        #librosa.output.write_wav(os.path.join(destPath, mode, 'audio', spkName+'.wav'), y_, sr)
        destFile = os.path.join(destPath, mode, 'audio', spkName+'.wav')
        fw.write(spkName+' '+destFile+'\n')

    fw.close()


if __name__ == '__main__':
    print '------------------------------------------------------------'
    basePath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214'
    destPath = '/aifs1/users/lj/voice/data/xytx_far_1214'
    make_path(destPath)

    audio_normal('enro')
    audio_normal('test')

