from general import *


def valid_model(net=None, validLoad=None):
    net.eval()

    totacc = 0
    totsum = 0
    totloss = 0
    totcnt = 0
    stime = time.time()

    for idx, (data, labs) in enumerate(validLoad):
        #data = data.resize_(data.size(0), 1, data.size(1), data.size(2))
        data, labs = Variable(data.cuda()), Variable(labs.cuda())

        # Forward
        net.eval()
        _, pred = net(data)
        loss = F.cross_entropy(pred, labs)
        #loss = criterion(pred, labs)
        loss = 1.0*loss.data.cpu().numpy()
        totloss += loss
        totcnt += 1

        # Acc
        _, pred = torch.max(pred.data, 1)
        acc = (pred == labs.data).sum()
        totacc += acc
        totsum += labs.data.size(0)

    rePrint('EvalTotal acc: %f loss: %f usetime: %f' %
            (1.0*totacc/totsum, 1.0*totloss/totcnt, time.time()-stime))
    return totloss


