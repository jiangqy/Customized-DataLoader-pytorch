from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset_processing
import torch.optim as optim
from torch.autograd import Variable
import torch

DATA_PATH = 'data'
TRAIN_DATA = 'train_img'
TEST_DATA = 'test_img'
TRAIN_IMG_FILE = 'train_img.txt'
TEST_IMG_FILE = 'test_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE = 'test_label.txt'

NLABELS = 5

batch_size = 4

transformations = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])
dset_train = dataset_processing.DatasetProcessing(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transformations)

dset_test = dataset_processing.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transformations)

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )

test_loader = DataLoader(dset_test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4
                         )

class MultiLabelNN(nn.Module):
    def __init__(self, nlabel):
        super(MultiLabelNN, self).__init__()
        self.nlabel = nlabel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(179776,1024)
        self.fc2 = nn.Linear(1024, nlabel)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = x.view(-1, 179776)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

use_gpu = torch.cuda.is_available()

model = MultiLabelNN(NLABELS)
if use_gpu:
    model = model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MultiLabelMarginLoss()

epochs = 3
for epoch in range(epochs):
    ### training phase
    total_training_loss = 0.0
    # total = 0.0
    for iter, traindata in enumerate(train_loader, 0):
        train_inputs, train_labels = traindata
        if use_gpu:
            train_inputs, train_labels = Variable(train_inputs.cuda()), Variable(train_labels.cuda())
        else: train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)

        optimizer.zero_grad()

        train_outputs = model(train_inputs)
        loss = criterion(train_outputs, train_labels)
        loss.backward()
        optimizer.step()

        # total += train_labels.size(0)
        total_training_loss += loss.data[0]
        print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.3f' %
              (iter, epoch, epochs, loss.data[0] / train_labels.size(0)))
    ### testing phase
    for iter, testdata in enumerate(test_loader, 0):
        test_inputs, test_labels = testdata
        if use_gpu:
            test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
        else: test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)

        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_labels)
        print('Testing Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.3f' %
              (iter, epoch, epochs, test_loss.data[0] / test_labels.size(0)))
