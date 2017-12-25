from general import *


def train_model(net=None,
                trainLoad=None,
                optimizer=None,
                log_interval=100, epoch=0, batch_size=256, lr=0.01,
                save_file=None):
    net.train()
    stime = time.time()
    for id_, (data, labs) in enumerate(trainLoad):
        data, labs = Variable(data.cuda()), Variable(labs.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, pred = net(data)
        loss = F.cross_entropy(pred, labs)
        loss.backward()
        optimizer.step()

        # ACC
        _, pred = torch.max(pred.data, 1)
        acc = (pred == labs.data).sum()

        if (id_ + 1) % log_interval == 0:
            printStr = "Train [%d|%d] lr: %f Loss: %.4f Acc: %.4f UseTime: %f" % (
                epoch + 1, (id_ + 1)*batch_size, lr, loss.data[0],
                1.0*acc/labs.data.size(0), time.time()-stime)
            rePrint(printStr)

    #if save_file is not None:
        #torch.save(net.state_dict(), save_file)
        #rePrint(save_file)
    return net


