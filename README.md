CNN_Pytorch
===
Python server that provides functions to construct CNN for image recognition

Provided function
---
1. create_model(datasets,mydata,underline_model,criterion,optimizer,lr,momentum)<br>
This function takes in the list of underline data name (ImageNet, CIFAR10, CIFAR100, etc.) and the location of the meta data of the provided data. Based on those, it creates a concatenated dataset using the costumed dataset (MyDataset & TorchVisionDataset - please see the explanation for those classes in the google doc provided at the end of this section). Then the model/network suitable for the dataset is created and trained and saved in the working directory. The meta-data of the comined dataset is also saved as text file is the form of [model_name]_metadata.txt
2. sss<br>
sss
3. 

For more detailed explanation and other functions/classes, please visit https://docs.google.com/document/d/19V1e8tMSWst83J8rNUQ_FdllOzKaGpc1Q1ED1zSVsfg/edit?usp=sharing 

More
---

