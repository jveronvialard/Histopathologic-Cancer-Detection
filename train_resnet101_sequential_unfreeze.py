import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
from resnet_binary_classifier import Resnet_Binary_Classifier, Resnet_Binary_Classifier_Sequential_Unfreeze
import os

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



ROOT_DIR = os.getcwd().replace("\\", "/")
LEARNING_RATE = 5e-3
NUM_EPOCHS = 30
BATCH_SIZE = 256
EVALUATE_EVERY = 50000
#SIZE_TRAIN_DATASET = 64 # To train on small dataset
#SIZE_VAL_DATASET = 64
SIZE_TRAIN_DATASET, SIZE_VAL_DATASET = None, None
L2_WD = 1e-5
#LR_SCHEDULER_T_MAX = 10
#LR_SCHEDULER_ETA_MIN = 1e-5

# TENSORBOARD SETUP
save_dir = "./save/"
SAVE_MODEL_PATH = save_dir + "train_resnet_sequential_unfreeze_best" + np.datetime64("now").astype(str).replace('-', '').replace(':', '')
name = "train_resnet_sequential_unfreeze_" + np.datetime64("now").astype(str).replace('-', '').replace(':', '')
writer = SummaryWriter(save_dir+name)

# DEVICE SETUP
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = "torch.cuda.FloatTensor"
else:
    device = torch.device('cpu')
    dtype = "torch.FloatTensor"

print('using device:', device)

def main():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
    train_dataset = PCamDataset(csv_file="train_labels.csv", 
                                root_dir=ROOT_DIR+"/data/", 
                                transform=data_transform)
    val_dataset = PCamDataset(csv_file="dev_labels.csv", 
                              root_dir=ROOT_DIR+"/data/", 
                              transform=data_transform)
    
    if SIZE_TRAIN_DATASET is not None:
        train_dataset = torch.utils.data.Subset(dataset=train_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(train_dataset)), 
                            size=SIZE_TRAIN_DATASET, 
                            replace=False))
    if SIZE_VAL_DATASET is not None:
        val_dataset = torch.utils.data.Subset(dataset=val_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(val_dataset)), 
                            size=SIZE_VAL_DATASET, 
                            replace=False))
        
    loader_train = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

    
    loader_val = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)
    
    model = Resnet_Binary_Classifier_Sequential_Unfreeze()
    
    if device.type == 'cuda':
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WD)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                 T_max=LR_SCHEDULER_T_MAX, 
    #                                                 eta_min=LR_SCHEDULER_ETA_MIN)
    
    epoch = 0
    n_iter = 0
    best_val_loss = float("inf")
    while epoch != NUM_EPOCHS:
        epoch += 1
        print("Starting epoch {epoch}".format(epoch=epoch))
        with tqdm(total=train_dataset.__len__()) as progress_bar:
            for t, (x, y) in enumerate(loader_train):
                n_iter += BATCH_SIZE
                x = x.to(device=device)
                y = y.to(device=device)
                model.train()
                scores = model(x)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # TENSORBOARD
                # no smoothing: running_loss
                loss_val  = loss.item()
                walltime = (epoch-1)*train_dataset.__len__() + t*BATCH_SIZE
                writer.add_scalar('train/BCE', loss_val, walltime)
                
                # Update progress bar
                progress_bar.update(BATCH_SIZE)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)
                
                if n_iter >= EVALUATE_EVERY:
                    n_iter = 0
                    model.eval() # set model to evaluation mode
                    with torch.no_grad():
                        loss_num, loss_den = 0., 0.
                        for t, (x, y) in enumerate(loader_val):
                            x = x.to(device=device)  # move to device, e.g. GPU
                            y = y.to(device=device)
                            scores = model(x)
                            loss = criterion(scores, y)
                            loss_num += x.size(0)*loss.item()
                            loss_den += x.size(0)
                        loss_val = loss_num/loss_den
                        writer.add_scalar('dev/BCE', loss_val, walltime)
                        if loss_val < best_val_loss:
                            best_val_loss = loss_val
                            torch.save(model, SAVE_MODEL_PATH)
                            print('Save best model at iteration {} with dev loss: {}'.format(walltime, best_val_loss))                    
        
                
        model.eval() # set model to evaluation mode
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for x, y in loader_train:
                x = x.to(device=device)  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                preds = (scores > 0).type(dtype)
                preds = preds.to(device)
                num_correct += (preds==y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got accuracy %f: %d / %d on training set' % (acc, num_correct, num_samples))
                        
            num_correct = 0
            num_samples = 0
            y_prob = np.zeros(val_dataset.__len__())
            y_val = np.zeros(val_dataset.__len__())
            for t, (x, y) in enumerate(loader_val):
                x = x.to(device=device)  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                y_prob[t*BATCH_SIZE: min((t+1)*BATCH_SIZE, val_dataset.__len__())] = torch.sigmoid(scores).cpu().numpy().flatten()
                y_val[t*BATCH_SIZE: min((t+1)*BATCH_SIZE, val_dataset.__len__())] = y.cpu().numpy().flatten()
                preds = (scores > 0).type(dtype)
                preds = preds.to(device)
                num_correct += (preds==y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got accuracy %f: %d / %d on val set' % (acc, num_correct, num_samples))
            
            
            y_val = y_val
            y_prob = y_prob
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            lw = 2
            ax.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC')
            ax.legend(loc="lower right")

            writer.add_figure("ROC", fig, global_step=epoch)
            
    
        #scheduler.step()
        


if __name__ == "__main__":
    main()
