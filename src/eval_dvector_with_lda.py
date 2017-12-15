import sys
sys.path.append('../')
from util import *




if __name__ == '__main__':
    print '----------------------------------------------------------------'

    #enroDvecDict = pickle.load(open('../data/far/enro/enro.dvec', 'rb'))
    #testDvecDict = pickle.load(open('../data/far/test/test.dvec', 'rb'))

    enroDvecDict = pickle.load(open('../data/far/enroFar/enroFar.dvec', 'rb'))
    testDvecDict = pickle.load(open('../data/far/testFar/testFar.dvec', 'rb'))


    trainDvecDict = pickle.load(open('../data/far/train/train.dvec', 'rb'))
    X = []
    y = []
    for spkID in trainDvecDict:
        for dvec in trainDvecDict[spkID]:
            X.append(dvec)
            y.append(spkID)
            print spkID, dvec.shape
            print len(trainDvecDict), len(trainDvecDict[spkID])
            exit(0)

    stime = time.time()
    print 'X: ', len(X), X[0].shape
    lda = LDA(n_components=100)
    lda.fit(X, y)
    print 'train lda time: {:f}'.format(time.time()-stime)

    score = []
    label = []
    positive = 0
    negative = 0
    for enroName in enroDvecDict:
        enroDvector = enroDvecDict[enroName]
        enroDvector = np.asarray(enroDvector).mean(axis=0)
        assert len(enroDvector) == 512
        enroDvector = enroDvector.reshape(1, 512)
        enroDvector = lda.transform(enroDvector)
        #assert enroDvector.shape == (1, 200)

        for testName in testDvecDict:
            for testDvector in testDvecDict[testName]:
                assert len(testDvector) == 512
                testDvector = testDvector.reshape(1, 512)
                testDvector = lda.transform(testDvector)
                #assert testDvector.shape == (1, 200)

                similar = cosine_similarity(enroDvector, testDvector)
                score.append(similar[0, 0])
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





