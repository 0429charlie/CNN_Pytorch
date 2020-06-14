import torch
from torch import nn
import torchvision
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import copy

# Let create the CNN now
class Net(nn.Module):
    def __init__(self, number_classes, underline_model):
        super(Net, self).__init__()
        # Now, depends on the underline model that we are using, the last fully connected layer are named differently
        # ResNet use fc while VGG use classifier[6]. DenseNet, on the other hand, use classifier
        # Please visit https://github.com/pytorch/vision/tree/master/torchvision/models for detail
        if underline_model == 'ResNet':
            if number_classes <= 10:
                self.model = models.resnet18(pretrained=True)
                self.model_name = 'ResNet18'
            else:
                self.model = models.resnet34(pretrained=True)
                self.model_name = 'ResNet34'
            self.num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.num_features, number_classes)
        elif underline_model == 'VGG':
            if number_classes <= 10:
                self.model = models.vgg11_bn(pretrained=True)
                self.model_name = 'VGG11'
            else:
                self.model = models.vgg16_bn(pretrained=True)
                self.model_name = 'VGG16'
            self.num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(self.num_features, number_classes)
        elif underline_model == 'DenseNet':
            if number_classes <= 10:
                self.model = models.densenet121(pretrained=True)
                self.model_name = 'DenseNet121'
            else:
                self.model = models.densenet161(pretrained=True)
                self.model_name = 'DenseNet161'
            self.num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.num_features, number_classes)
        else:
            print("Please input one of VGG, DenseNet, ResNet for a valid underline model to use")
        # Let add more underline model in future!!

    def forward(self, x):
        x = self.model(x)
        return x

