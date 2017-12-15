import sys
sys.path.append('../')
from util import *
from load_data import *
from model_resnet import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Resnet For Far Voice Verification')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-6, metavar='M',
                    help='SGD momentum (default: 1e-6)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

def train_model(saveFile):
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

        if (id_ + 1) % args.log_interval == 0:
            printStr = "Train [%d|%d] lr: %f Loss: %.4f Acc: %.4f BatchTime: %f" % (
                epoch + 1, (id_ + 1)*args.batch_size, lr, loss.data[0], 
                1.0*acc/labs.data.size(0), time.time()-stime)
            rePrint(printStr)

    torch.save(net, saveFile)


def valid_model():
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
        #loss = F.cross_entropy(pred, labs)
        loss = criterion(pred, labs)
        loss = 1.0*loss.data.cpu().numpy()
        totloss += loss
        totcnt += 1

        # Acc
        _, pred = torch.max(pred.data, 1)
        acc = (pred == labs.data).sum()
        totacc += acc
        totsum += labs.data.size(0)


    printStr = 'EvalTotal acc: %f loss: %f' % (1.0*totacc/totsum, 1.0*totloss/totcnt)
    print printStr
    logging.info(printStr)
    return totloss




if __name__ == '__main__':
    print '-----------------------------------------------------------------'
    make_path('log')
    #timeMark = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    timeMark = '2017-12-14 18:07:47'
    #logName = os.path.basename(sys.argv[0])+'-'+str(int(time.time()))+'-'+str(args)
    logName = os.path.basename(sys.argv[0])+'-'+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s : %(message)s')

    rePrint('-----------------------------------------------------------------')
    rePrint(str(args))

    trainFile = '../data/far/train/train.dict'
    validFile = '../data/far/train/valid.dict'
    modelPath = '../model/far/resnet18'+'-'+timeMark
    make_path(modelPath)
    modelFile = 'resnet.{:d}.model'

    trainData = LoadModelData(trainFile)
    trainLoad = torch.utils.data.DataLoader(trainData,
                        batch_size=args.batch_size, shuffle=True, num_workers=8)

    validData = LoadModelData(validFile)
    validLoad = torch.utils.data.DataLoader(validData,
                        batch_size=args.batch_size, shuffle=True, num_workers=8)


    #net = resnet34(num_classes=8784)
    #net = resnet18(num_classes=8784)
    net = resnet18(num_classes=8760)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                  patience=5, verbose=True, threshold=1e-6,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        train_model(os.path.join(modelPath, modelFile.format(epoch)))
        eval_loss = valid_model()
        scheduler.step(eval_loss)
        rePrint('-----------------------------------------------------------------')

    rePrint('[Done]')
    rePrint('-----------------------------------------------------------------')

