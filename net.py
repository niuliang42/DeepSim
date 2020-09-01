import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numpy import prod
import config

class SiameseResNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained ResNet.
    '''

    def __init__(self):
        super(SiameseResNet, self).__init__()
        pretrained_model = torchvision.models.resnet34(pretrained=True)
        if config.RESNET_POOLING == 'fixed' and str(pretrained_model.avgpool)[:8] == 'Adaptive':
            pretrained_model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(prod(config.RES34_960x720_SHAPE), 2048),
                                nn.BatchNorm1d(2048),
                                nn.ReLU(inplace=True),

                                nn.Linear(2048, 512),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(512, 8))

    def forward_once(self, x):
        output = self.model_conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class FCSiameseNet(nn.Module):
    '''
    The actual Semi-Siamese network for experiments.
    '''

    def __init__(self):
        super(FCSiameseNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(prod(config.RES34_960x720_SHAPE), 2048),
                                nn.BatchNorm1d(2048),
                                nn.ReLU(),

                                nn.Linear(2048, 256),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(256, 8))

    def forward(self, input1, input2):
        f1 = input1.view(input1.size(0), -1)
        f2 = input2.view(input2.size(0), -1)
        output1 = self.fc(f1)
        output2 = self.fc(f2)
        return output1, output2

class FCNet(nn.Module):
    '''
    The actual Semi-Siamese network for experiments.
    '''

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2 * prod(config.RES34_960x720_SHAPE), 2048),
                                 nn.BatchNorm1d(2048),
                                 nn.ReLU(),

                                 nn.Linear(2048, 256),
                                 nn.Dropout(0.2),
                                 nn.PReLU(1),

                                 nn.Linear(256, 1))
        self.sigm = nn.Sigmoid()

    def forward(self, input1, input2):
        f1 = input1.view(input1.size(0), -1)
        f2 = input2.view(input2.size(0), -1)
        f = torch.cat((f1,f2), 1)
        fc_output = self.fc(f)
        output = self.sigm(fc_output)
        return output

class SiameseSqueezeNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained SqueezeNet.
    '''

    def __init__(self):
        super(SiameseSqueezeNet, self).__init__()
        pretrained_model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.reduce_dim = nn.Conv2d(512, 16, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Sequential(nn.Linear(16 * prod(config.SQUEEZE_960x720_SHAPE), 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),

                                nn.Linear(512, 64),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(64, 8))

    def forward_once(self, x):
        output = self.model_conv(x)
        # print(output.shape)
        output = self.reduce_dim(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Copied from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
        output1, output2 = outputs[0], outputs[1]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# ---------------------- Deprecated --------------------- #

class SemiSiameseNet(nn.Module):
    '''
    This is just a try, please refer to FCNet for actual experiments.
    '''

    def __init__(self):
        quit("This network is not implemented properly, please do not use it. Please refer to FCNet.")
        super(SemiSiameseNet, self).__init__()
        pretrained_model = torchvision.models.resnet34(pretrained=True)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(2 * 512 * 17 * 24, 2048),
                                nn.BatchNorm1d(2048),
                                nn.ReLU(inplace=True),

                                nn.Linear(2048, 256),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(256, 1))
        self.sigm = nn.Sigmoid()

    def forward(self, input1, input2):
        f1 = self.model_conv(input1)
        f2 = self.model_conv(input2)

        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        output = torch.cat((f1,f2), 1)
        output = self.fc(output)
        output = self.sigm(output)
        return output

class SiamesePiNet(nn.Module):
    '''
    Siamese Network transfer learning use pretrained ResNet for Raspberry Pi.
    '''

    def __init__(self):
        quit("This network is deprecated. Please refer to SiameseSqueezeNet.")
        super(SiamesePiNet, self).__init__()
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.model_conv = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.reduce_dim = nn.Conv2d(512, 16, 1)
        self.fc = nn.Sequential(nn.Linear(16 * 17 * 24, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),

                                nn.Linear(512, 64),
                                nn.Dropout(0.2),
                                nn.PReLU(1),

                                nn.Linear(64, 8))

    def forward_once(self, x):
        output = self.model_conv(x)
        output = self.reduce_dim(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

