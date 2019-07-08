from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision.models import resnet18
from torchvision import transforms

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils import utils


class OCRNet(nn.Module):
    def __init__(self, img_w, img_h, timesteps, outsize, batch_size=32, lr=0.001,  loss='CTC', optim='Adam'):
        super(OCRNet, self).__init__()
        self.img_w_ = img_w
        self.img_h_ = img_h
        self.timesteps_ = timesteps
        self.batch_ = batch_size
        self.outsize_ = outsize
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_ = loss
        self.optim_ = optim
        self.lr_ = lr
        self.build_net_()
        self.set_params_(criterion=self.loss_, optimizer=self.optim_, lr=self.lr_)

    def set_params_(self, criterion, optimizer, lr, criterion_params=None):
        if criterion == 'CrossEntropyLoss' or criterion == 'CEL':
            self.criterion_ = nn.CrossEntropyLoss()
        elif criterion == 'CTCLoss' or criterion == 'CTC':
            if criterion_params is not None:
                self.criterion_ = nn.CTCLoss(criterion_params)
            else:
                self.criterion_ = nn.CTCLoss(blank=0)
        else:
            self.criterion_ = nn.CrossEntropyLoss()

        if optimizer == 'Adam':
            self.optimizer_ = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer_ = optim.SGD(self.parameters(), lr=lr)
        else:
            self.optimizer_ = optim.Adam(self.parameters(), lr=lr)

    def build_net_(self, kernel_size=(3, 3), rnn_size=512):
        # Pretrained ResNet is used as backbone
        # Last 3 layers are replaced by Conv and FC layers and finetuned
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]

        self.resnet = nn.Sequential(*modules)
        self.conv6 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512, 256)

        self.gru1 = nn.GRU(256, rnn_size, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(rnn_size*2, rnn_size, bidirectional=True, batch_first=True)

        self.fc2 = nn.Linear(rnn_size * 2, self.outsize_)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv6(x)
        x = F.relu(self.batch_norm6(x))

        # Reorder dims
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        # Reshape to RNN size
        x = x.view(x.shape[0], x.size(1), -1)

        x = self.fc1(x)

        x, (h_n, h_c) = self.gru1(x)
        x, (h_n, h_c) = self.gru2(x)

        out = self.fc2(x)

        return out

    def reshape_tensor(self, x, shape=[32, 256]):
        return x.reshape(shape)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def train_model(self, train_loader, epoch):
        train_loss = 0
        self.train()
        for batch_idx, (data, target, length) in enumerate(train_loader):
            self.optimizer_.zero_grad()
            output = self(data)
            output = output.transpose(0, 1)
            input_len, batch_size, vocab_size = output.size()

            # encode inputs
            logits = output.log_softmax(2).to(torch.float64)
            length = torch.full(size=(batch_size,), fill_value=5, dtype=torch.int32)

            target_lengths = Variable(length)
            logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)

            # calculate ctc
            loss = self.criterion_(logits, target, logits_lens, target_lengths)
            loss.backward()
            self.optimizer_.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0 or batch_idx % len(train_loader.dataset) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))
        return train_loss

    def test_model(self, test_loader, criterion, total_len=0):
        self.eval()
        if total_len == 0:
            total_len = len(test_loader.dataset)
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, length) in enumerate(test_loader):
                output = self(data)
                output = output.transpose(0, 1)

                input_len, batch_size, vocab_size = output.size()
                # encode inputs
                logits = output.log_softmax(2).to(torch.float64)
                length = torch.full(size=(batch_size,), fill_value=5, dtype=torch.int32)

                target_lengths = Variable(length)
                logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)

                # calculate ctc
                loss = criterion(logits, target, logits_lens, target_lengths)
                test_loss += loss.item()

        test_loss /= total_len
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))

        return test_loss


    def predict(model, pr_loader, do_plot=False):
        sample = pr_loader.get_random_sample()

        dim_sample = sample.unsqueeze(0)
        model.eval()
        output = model(dim_sample)

        tokens = output.softmax(2).argmax(2)
        tokens = tokens.squeeze(1).numpy()

        text = utils.decode(tokens[0], pr_loader.__get_chars__())

        if do_plot:
            i = transforms.ToPILImage().__call__(sample)
            #     i.save('my.png')
            plt.imshow(i)
            plt.show()

        return text