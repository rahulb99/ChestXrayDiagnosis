import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class ChestXrayDataSet(Dataset):
    def __init__(self, train_or_valid = "train", transform=None):
        data_path = './output'
        self.train_or_valid = train_or_valid
        if train_or_valid == "train":
            loaded = True #if loaded, load intermediate files, saving saves lots of time!
            if not loaded: 
                self.X = np.uint8(np.load(data_path + "train_X_small2.npy")) #(77872, 256, 256, 1)
                with open(data_path + "train_y_onehot2.pkl", "rb") as f: 
                    self.y = pickle.load(f) # (77872, 8)
                sub_bool = (self.y.sum(axis=1)!=0) # true for each positive example
                self.y = self.y[sub_bool,:] #only positive results
                self.X = self.X[sub_bool,:]
                np.save("output/train_X_smallest2.npy", self.X)
                np.save("output/train_y_smallest2.npy", self.y)
                self.X = np.uint8(self.X * 255)
                
            else:
                self.X = np.load("output/train_X_smallest2.npy")
                self.y = np.load("output/train_y_smallest2.npy")
                self.X = np.uint8(self.X * 255)

        else:
            self.X = np.uint8(np.load(data_path + "valid_X_small.npy")*255)
            with open(data_path + "valid_y_onehot.pkl", "rb") as f:
                self.y = pickle.load(f)
        
        #label weight for each class, depends on frequency. This generally leads to marginally better results
        self.label_weight_pos = (len(self.y)-self.y.sum(axis=0))/len(self.y) #negatives / total
        self.label_weight_neg = (self.y.sum(axis=0))/len(self.y) # positives / total
        self.transform = transform
    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        current_X = np.tile(self.X[index],3) #add second and third channels
        label = self.y[index]
        label_inverse = 1- label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        if self.transform is not None:
            image = self.transform(current_X)
        return image, torch.from_numpy(label).type(torch.FloatTensor), torch.from_numpy(weight).type(torch.FloatTensor)
    def __len__(self):
        return len(self.y)
    
class ResNet101(nn.Module): # DenseNet121 yields better results and faster training
    """Model modified.
    The architecture of our model is the same as standard ResNet101
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet101(x)
        return x
    
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
BATCH_SIZE = 32

def main():
    print("begin...")
    train_dataset = ChestXrayDataSet(train_or_valid="train",
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]) #from imagenet
                                        ]))
    print("train dataset loaded")
    def repeater(data_loader): #trainloader can run indefinitely
        for loader in repeat(data_loader):
            for data in loader:
                yield data          
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    train_loader = repeater(train_dataloader)
    

    valid_dataset = ChestXrayDataSet(train_or_valid="valid",
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    print("valid dataset loaded")
    
    cudnn.benchmark = True
    N_CLASSES = 8

    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005, betas=(0.9, 0.999))
    total_length = 4 * len(train_dataset) #hyperparameter of 4 times the original train dataset length through data augmentation
    
    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch:",epoch)
        running_loss = 0.0
        print("total train examples: " + str(total_length))
        print("total batches: " + str(int(total_length/BATCH_SIZE)))
        
        for i, (inputs, labels, weights) in tqdm(enumerate(train_loader), total=int(total_length/BATCH_SIZE)):
            if i * BATCH_SIZE > total_length:
                break
            optimizer.zero_grad()
            inputs_sub, labels_sub = Variable(inputs.cuda()), Variable(labels.cuda())
            weights_sub = Variable(weights.cuda())

            outputs = model(inputs_sub)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights_sub)
            loss = criterion(outputs, labels_sub)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()

        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()

        print("validating...")
        valid_length = len(valid_dataset)
        print("total valid examples: " + str(valid_length))
        print("total batches: " + str(int(valid_length/BATCH_SIZE)))
        for i, (inputs, target, weight) in tqdm(enumerate(valid_loader), total=int(valid_length/BATCH_SIZE)):
            if i * BATCH_SIZE > valid_length:
                break
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            with torch.no_grad():
                input_var = Variable(inputs.view(-1, 3, 224, 224).cuda())
                output = model(input_var)
                output = torch.sigmoid(output)
            pred = torch.cat((pred, output.data), 0)
            
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration',
                                'Mass','Nodule', 'Pneumonia', 'Pneumothorax']

        def compute_AUCs(gt, pred):
            AUROCs = []
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for i in range(N_CLASSES):
                AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            return AUROCs

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

        model.train()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / 715 ))
        torch.save(model.state_dict(),'DenseNet_weighted_split'+str(epoch+1)+'_'+str(AUROC_avg)+'.pkl')

    print('Finished Training')

def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


if __name__ == '__main__':
    main()

