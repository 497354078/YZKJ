import sys
import copy
sys.path.append('../')
from util import *


class DNN(nn.Module):
    def __init__(self, num_classes):
        super(DNN, self).__init__()

        self.fc0 = nn.Linear(1600, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        y = copy.copy(x)

        x = self.relu(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(x)
        return y, x

