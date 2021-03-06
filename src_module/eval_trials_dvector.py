from general import *


def get_enro_dvector(basePath, uttDict, dims):
    spk2utt = get_file_dict(os.path.join(basePath, 'enro/spk2utt'))
    utt2spk = get_file_dict(os.path.join(basePath, 'enro/utt2spk'))
    dataDict = {}
    for utt in uttDict:
        if utt not in utt2spk:
            raise IOError('utt[{:s}] not in utt2spk'.format(utt))
        if utt2spk[utt][0] not in dataDict:
            dataDict[utt2spk[utt][0]] = []
        dataDict[utt2spk[utt][0]].append(uttDict[utt])

    enroDict = {}
    for spk in dataDict:
        enroDvec = dataDict[spk]
        if len(enroDvec) == 0:
            print 'len(enroDvec[{:s}]) == 0'.format(spk)
            continue
        enroDvec = np.asarray(enroDvec)
        enroDvec = enroDvec.mean(axis=0)
        assert len(enroDvec) == dims
        enroDict[spk] = enroDvec

    return enroDict

def eval_dvector(enroDict, testDict, files, dims):
    check_file(files)
    lines = open(files, 'rb').readlines()
    score = []
    label = []
    positive = 0
    negative = 0
    for items in lines:
        items = items.split('\n')[0].split(' ')
        spkName = items[0]
        uttName = items[1]
        judge = items[2]
        #print items
        if spkName not in enroDict:
            #print 'spkName[{:s}] not in enroDict'.format(spkName)
            continue
        if uttName not in testDict:
            #print 'uttName[{:s}] not in testDict'.format(uttName)
            continue
        enroDvec = enroDict[spkName].reshape((1, dims))
        testDvec = testDict[uttName].reshape((1, dims))
        similar = cosine_similarity(enroDvec, testDvec)
        score.append(similar[0, 0])
        if judge == 'target':
            label.append(1)
            positive += 1
        else:
            label.append(0)
            negative += 1

    totFpr, totTpr, thresholds = roc_curve(label, score, pos_label=1)
    EER = totFpr[np.abs(totFpr-(1-totTpr)).argmin(0)]
    AUC = roc_auc_score(label, score, average='macro', sample_weight=None)

    rePrint('EER: {:f}  AUC: {:f}'.format(EER, AUC))
    rePrint('positive: {:f}  negative: {:f}  sum(label): {:f}'.format(positive, negative, np.sum(label)))
    rePrint('thresholds: {:s}  shape:{:s}'.format(type(thresholds), thresholds.shape))
    rePrint(thresholds)
    rePrint('min score: {:f}  max score: {:f}  mean_score: {:f}'.format(
            np.min(thresholds), np.max(thresholds), np.mean(thresholds)))
    rePrint('')




if __name__ == '__main__':
    print '----------------------------------------------------------------'

    #enroDvecDict = pickle.load(open('../../data/near/enro_near/enro_near.dvec', 'rb'))
    #testDvecDict = pickle.load(open('../../data/near/test_near/test_near.dvec', 'rb'))

    enroDvecDict = pickle.load(open('../../data/far/enro_far_1214/enro_far_1214.dvec', 'rb'))
    testDvecDict = pickle.load(open('../../data/far/test_far_1214/test_far_1214.dvec', 'rb'))

    #enroDvecDict = pickle.load(open('../../data/far/enro_far_1208/enro_far_1208.dvec', 'rb'))
    #testDvecDict = pickle.load(open('../../data/far/test_far_1208/test_far_1208.dvec', 'rb'))

    basePath = '/aifs1/users/kxd/sre/data/test/xytx_far_1214'
    trails = '/aifs1/users/kxd/sre/data/test/xytx_far_1214/test/trials'
    dims = 512

    eval_dvector(get_enro_dvector(basePath, enroDvecDict, dims), testDvecDict, trails, dims)




