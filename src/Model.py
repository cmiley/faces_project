from torch import nn, optim
import torch
from torch.autograd import Variable


class AttributeNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttributeNetwork, self).__init__()

        c1 = {'size': 75, 'filter': 7}
        c2 = {'size': 200, 'filter': 5}
        c3 = {'size': 300, 'filter': 3}
        fc1 = 512
        fc2 = 512

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, c1['size'], c1['filter'], stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(c1['size'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1['size'], c2['size'], c2['filter']),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(c2['size']),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2['size'], c3['size'], c3['filter']),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2),
            nn.LocalResponseNorm(c3['size'])
        )

        self.fc_layers = nn.Sequential(
            # fully connected layer 1
            nn.Linear(2 * c3['size'], fc1),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # fully connected layer 2
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # output layer
            nn.Linear(fc2, output_size)
        )

        if torch.cuda.is_available():
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.fc_layers.cuda()

    def forward(self, input_value):
        out = self.conv1(input_value)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layers(out)
        return out

    def m_train(self, labeled_data_batch, optimizer, criterion):
        input_value, observed_output_value = labeled_data_batch['image'], labeled_data_batch['attribs']
        optimizer.zero_grad()

        if torch.cuda.is_available():
            input_value = Variable(input_value).float().cuda()
            observed_output_value = Variable(observed_output_value).float().cuda()

        predicted_value = self.forward(input_value)
        loss = criterion(predicted_value, observed_output_value.float())
        loss.backward()
        optimizer.step()

        return loss.item()
