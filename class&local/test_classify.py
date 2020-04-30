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
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

#This python script finds the AUC of the model on the test set, mainly for quality assurance

data_path = "/home/ubuntu/project/data/postproc/"
model_path = "DenseNet_weighted3_0.779168123174.pkl"
curve_path = "/home/ubuntu/project/CheXNet-with-localization/curves_dense/"
classify = False

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
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ChestXrayDataSet(Dataset):
    def __init__(self, transform=None):

        self.X = np.uint8(np.load(data_path + "test_X_small.npy")*255)
        with open(data_path + "test_y_onehot.pkl", "rb") as f:
            self.y = pickle.load(f)
        
        self.label_weight_pos = (len(self.y)-self.y.sum(axis=0))/len(self.y) #label weight for each class, depends on frequency
        self.label_weight_neg = (self.y.sum(axis=0))/len(self.y)
        self.transform = transform
        

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        current_X = np.tile(self.X[index],3) 
        label = self.y[index]
        label_inverse = 1- label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        if self.transform is not None:
            image = self.transform(current_X)
        return image, torch.from_numpy(label).type(torch.FloatTensor), torch.from_numpy(weight).type(torch.FloatTensor)
    def __len__(self):
        return len(self.y)

print("beginning...")
class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
BATCH_SIZE = 32
N_CLASSES = 8

test_dataset = ChestXrayDataSet(transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    
print("test dataset loaded")

cudnn.benchmark = True

# initialize and load the model
model = DenseNet121(N_CLASSES).cuda()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(model_path))
print(model_path)
print("model loaded")
model.eval()

# initialize the ground truth and output tensor
gt = torch.FloatTensor().cuda() #of shape (# of batches * batch_size, 8)
pred = torch.FloatTensor().cuda()

print("testing...")
test_length = len(test_dataset)
print("total test examples: " + str(test_length))
print("total batches: " + str(int(test_length/BATCH_SIZE)))

for i, (inputs, target, weight) in tqdm(enumerate(test_loader), total=int(test_length/BATCH_SIZE)):
    target = target.cuda()
    gt = torch.cat((gt, target), 0)
    with torch.no_grad():
        input_var = Variable(inputs.view(-1, 3, 224, 224).cuda())
        output = model(input_var)
        output = torch.sigmoid(output)
    pred = torch.cat((pred, output.data), 0)
            
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration',
                                'Mass','Nodule', 'Pneumonia', 'Pneumothorax']
def compute_stats(gt, pred):
    AUROCs = []
    roc_curves = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        roc_curves.append(roc_curve(gt_np[:, i], pred_np[:, i]))
    return AUROCs, roc_curves

AUROCs, roc_curves = compute_stats(gt, pred)
AUROC_avg = np.array(AUROCs).mean()
print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
for i in range(N_CLASSES):
    print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

for i in range(N_CLASSES):
    fpr, tpr, thresholds = roc_curves[i]
    plt.plot([0,1], [0,1], linestyle = "--")
    plt.plot(fpr, tpr, label = "model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC CURVE: " + CLASS_NAMES[i])
    plt.savefig(curve_path + CLASS_NAMES[i] + ".png")
    plt.clf()
    




    
    
    
    