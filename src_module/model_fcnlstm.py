import sys
sys.path.append('../')
from general import *
from load_random_data import LoadRandomData

class DeConv(nn.Module):
    def __init__(self,ninp=40, nhid=40, nlayers=2, num_classes=10):
        super(DeConv, self).__init__()

        self.num_classes = num_classes
        self.N_conv = 5

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1, bias=True)

        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.dconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=True)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=True)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=True)
        self.dconv5 = nn.ConvTranspose2d(64, 1, 3, 2, 1, bias=True)

        self.lstm = nn.LSTM(ninp, nhid, nlayers, batch_first=True)

        self.fc = nn.Linear(nhid, num_classes)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU()


        self.FLAG = True
        self.SizeQueue = []

    def forward(self, x):
        if self.FLAG:
            print 'init x: ', x.size()
            self.SizeQueue.append(x.size())

        for i in range(self.N_conv):
            conv_name = 'self.conv{:d}'.format(i+1)
            x = eval(conv_name)(x)
            bn_name = 'self.bn{:d}'.format(i+1)
            x = eval(bn_name)(x)
            x = self.relu(x)
            if self.FLAG:
                print 'conv{:d} x: '.format(i+1), x.size()
                self.SizeQueue.append(x.size())

        SizeCnt = len(self.SizeQueue)-2

        for i in range(self.N_conv):
            dconv_name = 'self.dconv{:d}'.format(i+1)
            x = eval(dconv_name)(x, output_size=self.SizeQueue[SizeCnt])
            bn_name = 'self.bn{:d}'.format(self.N_conv-i-1)
            x = eval(bn_name)(x)
            x = self.relu(x)
            SizeCnt -= 1
            if self.FLAG:
                print 'dconv{:d} x: '.format(i+1), x.size()

        x = x.view(x.size(0), x.size(2), x.size(3))
        if self.FLAG: print 'xview x: ', x.size()
        x = torch.transpose(x, 1, 2)
        if self.FLAG: print 'transpose x: ', x.size()

        x, _ = self.lstm(x)
        if self.FLAG: print 'lstm x: ', x.size()

        x = torch.max(x, dim=1)[0]
        #x = x[:, -1, :] # [batch, seq, feature] last hidden
        if self.FLAG:print 'tmax x:', x.size()
        x = x.view(x.size(0), -1)
        y = copy.copy(x)
        if self.FLAG:print 'view y:', x.size()
        x = self.fc(x)
        if self.FLAG:print 'fc x:', x.size()

        self.FLAG = False
        return y, x

if __name__ == '__main__':

    print ''
    num_classes = 10
    trainData = LoadRandomData(N=100, num_classes=num_classes)
    trainLoad = torch.utils.data.DataLoader(trainData,  batch_size=20, shuffle=True, num_workers=4)

    net = DeConv(ninp=40, nhid=40, nlayers=2, num_classes=num_classes)

    for _, (data, labs) in enumerate(trainLoad):
        data, labs = Variable(data), Variable(labs)
        print type(data), data.size()
        print type(labs), labs.size()

        out = net(data)

        break


