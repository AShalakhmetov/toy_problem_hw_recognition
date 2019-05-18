from torchvision import models
from torch import nn
from torch import optim

import time
import copy
import torch
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, img_w = 140, img_h = 28, timesteps=5, batch_size=32, outsize=10, loss='CTC', optim='Adam'):
        super(NNModel, self).__init__()
        self.img_w_ = img_w
        self.img_h_ = img_h
        self.timesteps_ = timesteps
        self.batch_ = batch_size
        self.outsize_ = outsize
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_ = loss
        self.optim_ = optim
        self.set_params_(criterion=self.loss_, optimizer=self.optim_)
        self.build_net_()

    def set_params_(self, criterion='CTC', criterion_params=None, optimizer='Adam', lr=1e-3):
        if criterion == 'CrossEntropyLoss' or criterion == 'CEL':
            self.criterion_ = nn.CrossEntropyLoss()
        elif criterion == 'CTCLoss' or criterion == 'CTC':
            if criterion_params is not None:
                self.criterion_ = nn.CTCLoss(criterion_params)
            else:
                self.criterion_ = nn.CTCLoss(reduction='sum')
        else:
            self.criterion_ = nn.CrossEntropyLoss()

        if optimizer == 'Adam':
            self.optimizer_ = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer_ = optim.SGD(self.parameters(), lr=lr)
        else:
            self.optimizer_ = optim.Adam(self.parameters(), lr=lr)

    def build_net_(self, input_channels=1, conv_filters=16, kernel_size=(3, 3), rnn_input=3920, rnn_size=512):
        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size)  # TODO: what padding we must set?
        # TODO: ReLU
        # TODO: MaxPool2D = (2,2)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size)
        # TODO: ReLU
        # TODO: MaxPool2D = (2,2)

        # reshaped_size = ((self.img_w // (2**2)) * (self.img_h // (2**2)) * conv_filters)
        # self.fc1 = nn.Linear(reshaped_size, time_dense_size)
        # TODO: ReLU

        self.gru1 = nn.GRU(rnn_input, rnn_size, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(rnn_size, rnn_size, bidirectional=True, batch_first=True)

        self.fc2 = nn.Linear(rnn_size, self.outsize_) # TODO: reset output size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        # Reorder dims
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous()

        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))

        # Reshape to RNN size
        x = x.view(self.batch_, self.timesteps_, -1)

        x = self.gru1(x)
        x = self.gru2(x)

        x = self.fc2(x)
        x = F.softmax(x)

        return x

    def reshape_tensor(self, x, shape=[32, 256]):
        return x.reshape(shape)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




    def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, is_inception=False):

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)

                    inputs = inputs.to(model.device_)
                    labels = labels.to(model.device_)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

model = NNModel()

# print(model.model)
print(model.criterion)
print(model.optimizer)
print(model)

print('-' * 10)
model.set_params(lr=1e-5)
print(model.optimizer)
