import sys
sys.path.append('../')
from util import *



if __name__ == '__main__':
    print '----------------------------------------------------------------'

    #enroDvecDict = pickle.load(open('../data/far/enro/enro.dvec', 'rb'))
    #testDvecDict = pickle.load(open('../data/far/test/test.dvec', 'rb'))

    enroDvecDict = pickle.load(open('../data/far/enroFar/enroFar.dvec', 'rb'))
    testDvecDict = pickle.load(open('../data/far/testFar/testFar.dvec', 'rb'))

    score = []
    label = []
    positive = 0
    negative = 0
    for enroName in enroDvecDict:
        enroDvector = enroDvecDict[enroName]
        enroDvector = np.asarray(enroDvector).mean(axis=0)
        assert len(enroDvector) == 512
        enroDvector = enroDvector.reshape(1, 512)
        #enroDvector = preprocessing.normalize(enroDvector, norm='l2')
        if enroDvector.min() < 0:
            raise IOError(enroName)
        for testName in testDvecDict:
            for testDvector in testDvecDict[testName]:
                assert len(testDvector) == 512
                testDvector = testDvector.reshape(1, 512)
                #testDvector = preprocessing.normalize(testDvector, norm='l2')
                if testDvector.min() < 0:
                    raise IOError(testName)
                #exit(0)
                #similar = euclidean_distances(enroDvector, testDvector)
                similar = cosine_distances(enroDvector, testDvector) # 0.93
                #similar = cosine_similarity(enroDvector, testDvector)
                #print 'similar: ', similar.shape, similar[0, 0]
                #exit(0)
                score.append(1-similar[0, 0])
                if enroName == testName:
                    label.append(1)
                    positive += 1
                else:
                    label.append(0)
                    negative += 1
    totFpr, totTpr, thresholds = roc_curve(label, score, pos_label=1)
    EER = totFpr[np.abs(totFpr-(1-totTpr)).argmin(0)]
    AUC = roc_auc_score(label, score, average='macro', sample_weight=None)
    
    print 'EER: ', EER, 'AUC: ', AUC
    print 'positive: ', positive, ' negative: ', negative, 'sum(label): ', np.sum(label)
    print 'thresholds: ', type(thresholds), thresholds.shape
    print thresholds
    print np.min(thresholds), np.max(thresholds), np.mean(thresholds)
    print 'min score: ', min(score), 'max score: ', max(score)
    




