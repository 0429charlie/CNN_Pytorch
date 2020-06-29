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

import os
import wget

from flask import Flask, request, jsonify
import base64
import io

app = Flask(__name__)


# The network class
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


# Let create our data
# MyDataset get the file location and read the files into the sample attribute.
# sample attribute is list of tuple (PIL-img, class_id)
class UnderlineDataset(Dataset):
    def __init__(self, file, transform):
        self.samples = []
        self.size = 0
        self.transform = transform
        self.file = file
        self.map = None
        # Here, file is the matadata about the image in the MyData folder!!
        # We assume metadata is in the format of the following
        # image_path1 image_catagory1
        # image_path2 image_catagory2
        # ...
        with open("./MyData/"+file, "r") as f:
            for line in f:
                img_path, val = line.split(" ")
                val = int(val)

                # This is the same num_classes as the input into the network above
                #if val >= NUM_CLASSES:
                #    break

                self.samples.append((self.transform(Image.open("./MyData/"+img_path)), val))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img, val = self.samples[idx]
        if self.map:
            return img, self.map[val]
        else:
            return img, val

    def updatemap(self, newmap, mymetadata):
        # Given newmap to bind to, update the self.map so that data is retrieve correctly and return the updated mymetadata
        inv_newmap = {v: k for k, v in newmap.items()}
        class_index = len(newmap)
        newdic = {}
        # For each old data entry
        for i in mymetadata:
            # the old data is already in the provided new data so no change in meta-data and mapping
            if mymetadata[i] in inv_newmap:
                newdic[i] = inv_newmap[mymetadata[i]]
            else:
                # the old data is not in the provided new data. Add it to the end of the new data and recod the new mapping
                newmap[class_index] = mymetadata[i]
                newdic[i] = class_index
                class_index = class_index+1
        self.map = newdic
        return newmap


# Wrapper class
class MyDataset:
    def __init__(self, file, transform):
        # underlinedataset is one that we need when calling Dataloader
        self.underlinedataset = UnderlineDataset(file, transform)
        self.metadata = None
        # Setup the meta data
        # Assume there is a file named metadata in the directory MyData
        # metadata has the name of the class on each line as following
        # car
        # bike
        # ...
        dic = {}
        with open("./MyData/metadata", "r") as f:
            i = 0
            for line in f:
                dic[i] = line
                i = i + 1
        self.metadata = dic
        # Now, metadta is a dictionary of key is the integer corresponding to the category
        # {0:car, 1:bike,...}

    def getdataset(self):
        return self.underlinedataset

    def getmatadata(self):
        # return the mata data according to the specific dataset from torch vision
        # the meta data is in the form of dictionary. dic[index] = class_name
        return self.metadata

    def updatemap(self, bindto):
        self.metadata = self.underlinedataset.updatemap(bindto, self.metadata)


# class wrapper for ImageNet that have the modified __getitem__
class ImageNet_Wrapper(torchvision.datasets.ImageNet):
    def __init__(self, root, downlaod, transform):
        super().__init__(root=root, transform=transform)
        # We need to download the archives externally (eg. from my GitHub)
        if downlaod:
            os.makedirs(root)
            wget.download('https://github.com/0429charlie/ImageNet_metadata/raw/master/ILSVRC2012_devkit_t12.tar.gz', root)
        self.map = None

    def updatemap(self, newmap, mymetadata):
        # Given newmap to bind to, update the self.map so that data is retrieve correctly and return the updated mymetadata
        inv_newmap = {v: k for k, v in newmap.items()}
        class_index = len(newmap)
        newdic = {}
        # For each old data entry
        for i in mymetadata:
            # the old data is already in the provided new data so no change in meta-data and mapping
            if mymetadata[i] in inv_newmap:
                newdic[i] = inv_newmap[mymetadata[i]]
            else:
                # the old data is not in the provided new data. Add it to the end of the new data and recod the new mapping
                newmap[class_index] = mymetadata[i]
                newdic[i] = class_index
                class_index = class_index+1
        self.map = newdic
        return newmap

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if (self.map!=None):
            if target in self.map:
                target = self.map[target]
        return img, target


# class wrapper for CIFAR10 that have the modified __getitem__
class CIFAR10_Wrapper(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.map = None

    def updatemap(self, newmap, mymetadata):
        # Given newmap to bind to, update the self.map so that data is retrieve correctly and return the updated mymetadata
        inv_newmap = {v: k for k, v in newmap.items()}
        class_index = len(newmap)
        newdic = {}
        # For each old data entry
        for i in mymetadata:
            # the old data is already in the provided new data so no change in meta-data and mapping
            if mymetadata[i] in inv_newmap:
                newdic[i] = inv_newmap[mymetadata[i]]
            else:
                # the old data is not in the provided new data. Add it to the end of the new data and recod the new mapping
                newmap[class_index] = mymetadata[i]
                newdic[i] = class_index
                class_index = class_index+1
        self.map = newdic
        return newmap

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if (self.map!=None):
            if target in self.map:
                target = self.map[target]
        return img, target


# class wrapper for CIFAR100 that have the modified __getitem__
class CIFAR100_Wrapper(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.map = None

    def updatemap(self, newmap, mymetadata):
        # Given newmap to bind to, update the self.map so that data is retrieve correctly and return the updated mymetadata
        inv_newmap = {v: k for k, v in newmap.items()}
        class_index = len(newmap)
        newdic = {}
        # For each old data entry
        for i in mymetadata:
            # the old data is already in the provided new data so no change in meta-data and mapping
            if mymetadata[i] in inv_newmap:
                newdic[i] = inv_newmap[mymetadata[i]]
            else:
                # the old data is not in the provided new data. Add it to the end of the new data and recod the new mapping
                newmap[class_index] = mymetadata[i]
                newdic[i] = class_index
                class_index = class_index+1
        self.map = newdic
        return newmap

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if (self.map!=None):
            if target in self.map:
                target = self.map[target]
        return img, target
# See if there is otherway that we don't have to create wrapper for each dataset


# Wrapper class for torchvision dataset
class TorchVisionDataSet:
    def __init__(self, dataset_name, train, download, transform):
        # dataset_name is one of the available dataset in torch vision. It should be one of dataset in the following
        # https://pytorch.org/docs/stable/torchvision/datasets.html
        self.dataset_name = dataset_name
        self.train = train
        self.download = download
        self.transform = transform
        # update the dataset according to the specific dataset from torch vision
        if (self.dataset_name == "ImageNet"):
            #ds = torchvision.datasets.ImageNet(root="./ImageNet")
            ds = ImageNet_Wrapper("./ImageNet", download=self.download, transform=self.transform)
            # check this below...actually, we need to download the archives manually to download the dataset
            self.metadata = ds.load_meta_file("./ImageNet")
            self.ds = ds
        elif (self.dataset_name == "CIFAR10"):
            ds = CIFAR10_Wrapper(root="./CIFAR10", train=self.train, download=self.download,
                                              transform=self.transform)
            d = self.unpickle("CIFAR10/cifar-10-batches-py/batches.meta")
            i = 0
            dic = {}
            for class_name in d[b'label_names']:
                dic[i] = class_name.decode("utf-8")
                i = i + 1
            self.metadata = dic
            self.ds = ds
        elif (self.dataset_name == "CIFAR100"):
            ds = torchvision.datasets.CIFAR100(root="./CIFAR100", train=self.train, download=self.download,
                                               transform=self.transform)
            d = self.unpickle("CIFAR100/cifar-100-batches-py/batches.meta")
            i = 0
            dic = {}
            for class_name in d[b'label_names']:
                dic[i] = class_name.decode("utf-8")
                i = i + 1
            self.metadata = dic
            self.ds = ds
        else:
            print("Please input one of ImageNet, CIFAR10, CIFAR100 for dataset_name")
            self.metadata = None
            self.ds = None
        # Let add more in future

    # Helper we need to get meta data for CIFAR data set
    # Please visit https://www.cs.toronto.edu/~kriz/cifar.html for more
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def getdataset(self):
        return self.ds

    def getmatadata(self):
        # return the mata data according to the specific dataset from torch vision
        # the meta data is in the form of dictionary. dic[index] = class_name
        return self.metadata

    def updatemap(self, bindto):
        self.metadata = self.ds.updatemap(bindto, self.metadata)

#--------we good till here


def train_model(net, criterion, optimizer, dataloader, num_epochs=25):
    # First record the best model and best accuracy, which is current model with accuracy of 0 assuming it is randomly initialized
    best_model = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    # Iterate through each epoch
    for i in range(num_epochs):
        net.train()
        # record the accuracy for each epoch
        running_corrects = 0
        total = len(dataloader)
        # Iterate through data randomly got from the dataloader
        for j, data in enumerate(dataloader):
            # Note that here input is in the form of (Batch_size X 3 X Image_length X Image_width)
            # target, on the other hand, is in the form of (Batch_size X 1)
            inputs, target = data

            # if use gpu....input.cuda(), net.cuda(), target.cuda()

            optimizer.zero_grad()
            output = net.forward(inputs)
            # Here, the output is in the form of (Batch_size X class_number)
            loss = criterion(output, target)
            # Then get get loss of tensor with single number (no dimension)
            loss.backward()
            optimizer.step()

            # Record the number of correct output
            # output is in the form of (Batch_size X class_number). So torch.max(output,1) gives the maximum in the
            # second dimension which is the dimension of class_number. This mean _ is the maximum percentage and preds
            # is its index which in this case is the int corresponding to the class. So _ and preds are in the shapes
            # of (Batch_size X 1)
            _, preds = torch.max(output, 1)
            # Note that preds is in format of (Batch_size X 1) and target.data is also in form of (Batch_size X 1)
            running_corrects = running_corrects+torch.sum(preds == target.data)

        # Calculate the accuracy and update the best model accordingly
        if (best_acc<(running_corrects/total)):
            best_acc = (running_corrects/total)
            best_model = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model)
    return net


def evaluate_model(net, criterion, dataloader, num_epochs=25):
    # We have to record all loss and accuracy from all epochs
    total_loss = []
    total_acc = []
    # Iterate through each epoch
    for i in range(num_epochs):
        net.eval()
        # record the loss and accuracy for each epoch
        running_loss = 0.0
        running_corrects = 0
        total = len(dataloader)
        # Iterate through data randomly got from the dataloader
        for j, data in enumerate(dataloader):
            # Note that here input is in the form of (Batch_size X 3 X Image_length X Image_width)
            # target, on the other hand, is in the form of (Batch_size X 1)
            inputs, target = data

            # if use gpu....input.cuda(), net.cuda(), target.cuda()

            output = net.forward(inputs)
            # Here, the output is in the form of (Batch_size X class_number)
            loss = criterion(output, target)
            # Then get get loss of tensor with single number (no dimension)

            # Record the loss and accuracy for this epoch
            running_loss = running_loss+loss.item()
            # output is in the form of (Batch_size X class_number). So torch.max(output,1) gives the maximum in the
            # second dimension which is the dimension of class_number. This mean _ is the maximum percentage and preds
            # is its index which in this case is the int corresponding to the class. So _ and preds are on the shapes
            # of (Batch_size X 1)
            _, preds = torch.max(output, 1)
            # Note that preds is in format of (Batch_size X 1) and target.data is also in form of (Batch_size X 1)
            running_corrects = running_corrects+torch.sum(preds == target.data)

        # Record the loss and accuracy for this epoch
        total_acc.append(running_corrects/total)
        total_loss.append(running_loss)
    return total_acc, total_loss

# Function that create the model based on the provided dataset
def create_model(datasets,mydata,underline_model,criterion,optimizer,lr,momentum):
    # Assume datasets is the list of underline data name
    # Assume mydata is the location of the meta data of the provided data
    train_ds_class = []
    if (mydata!=None):
        # Create dataset class and its underline dataset
        train_ds_class.append(MyDataset(mydata,None))
    if (datasets!=None):
        for d in datasets:
            train_ds_class.append(TorchVisionDataSet(d,True,True,None))
    # record the dataset to use for ConCateDataSet()
    train_ds = [train_ds_class[0].getdataset()]
    # Update the meta data and ways to retrieve data for each dataset
    metadata = train_ds_class[0].getmatadata()
    for idx in range(1,len(datasets)):
        train_ds_class[idx].updatemap(metadata)
        train_ds.append(train_ds_class[idx].getdataset())
        metadata = train_ds_class[idx].getmatadata()
    # Now, from the given data, we get the metadata of this model and the dataloader that combined all the dataset
    concated_dataset = Dataset.ConcatDataset(train_ds)
    train_loader = DataLoader(concated_dataset, batch_size=4, shuffle=True, num_workers=2)
    #Now, let define the network given the dataset to train on
    net = Net(len(metadata),underline_model)
    # Next, let choose the criterion to use to evaluate the loss
    # criterion can be choosed from https://pytorch.org/docs/master/nn.html#loss-functions
    c = None
    if criterion=="CrossEntropyLoss":
        c = nn.CrossEntropyLoss()
    # Then let choose the optimizer for training
    # optimizer can be choosed from https://pytorch.org/docs/master/optim.html#
    optm = None
    if optimizer=="SGD":
        optm = optim.SGD(net.parameters(),lr,momentum)
    # Finally, we can train the network

    # Evaluate the network

    # Save the network

    pass



if __name__ == '__main__':
    # Running locally, this bind to the url: http://localhost:80/
    app.run(host='0.0.0.0', port=80)
