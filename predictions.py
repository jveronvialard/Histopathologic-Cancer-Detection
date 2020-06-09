import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
import os

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn



ROOT_DIR = os.getcwd().replace("\\", "/")
BATCH_SIZE = 128

save_dir = "./save/"
SAVE_MODEL_PATH = save_dir + "train_resnet_sequential_unfreeze_best20200604T175157"

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
    test_dataset = PCamDataset(csv_file="test_labels.csv", 
                                root_dir=ROOT_DIR+"/data/", 
                                transform=data_transform)
        
    loader_test = DataLoader(test_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
    
    len_dataset = test_dataset.__len__()

    if device.type == 'cpu':
        model = torch.load(SAVE_MODEL_PATH, map_location=torch.device('cpu'))
    else:
        model = torch.load(SAVE_MODEL_PATH)
    
    if device.type == 'cuda':
        model.cuda()
    """
    y_test = np.zeros(len_dataset)
    y_hat = np.zeros(len_dataset)

    model.eval()
    with torch.no_grad():
        for t, (x, y) in tqdm(enumerate(loader_test)):
            #print("Iteration {}/{}".format(t, len_dataset//BATCH_SIZE))
            y_test[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] = y.numpy().reshape(len(y))
            y_hat[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] = model(x).numpy().reshape(len(y))
    y_hat = 1/(1+np.exp(-y_hat))
    np.save("y_test", y_test)
    np.save("y_hat_resnet101_unfreeze", y_hat)
    """
    
    y_hat = np.load("y_hat_resnet101_unfreeze.npy")
    y_test = np.load("y_test.npy")
    
    print("[y_hat]")
    plt.hist(y_hat)
    plt.title("Distribution of y_hat")
    plt.savefig("y_hat")
    plt.show()    
    
    indices_0 = np.where(y_test==0)
    indices_1 = np.where(y_test==1)
    
    plt.hist(y_hat[indices_0], bins=100, color='green', label='No tumor')
    plt.hist(y_hat[indices_1], bins=100, color='red', label='Tumor')
    plt.title("Distribution of predictions for Classes 0 and 1")
    plt.legend()
    plt.savefig("y_hat_0_1")
    plt.show()
    
    print("[ROC]")
    fpr, tpr, ROC_thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig("ROC curve")
    plt.show()
    distances = fpr**2 + (1-tpr)**2
    ROC_optimal_threshold = ROC_thresholds[distances.argmin()]
    print("The optimal threshold for ROC optimization is %0.3f" %ROC_optimal_threshold)
    y_pred_ROC = (y_hat>ROC_optimal_threshold).astype(float)
    FPR = fpr[distances.argmin()]
    FNR = np.sum(1-y_pred_ROC[indices_1[0]])/len(indices_1[0])
    print("The false negative rate is %0.4f" %FNR)
    print("The false positive rate is %0.4f" %FPR)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_ROC)
    matrix = matrix/matrix.sum(axis=1).reshape((2, 1))
    sn.heatmap(matrix, annot=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    print("[PRECISION-RECALL]")
    precision, recall, PR_thresholds = precision_recall_curve(y_test, y_hat)
    avg_precision_recall_score = average_precision_score(y_test, y_hat)
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='Precision Recall Curve (area = %0.4f)' % avg_precision_recall_score)
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig("PR curve")
    plt.show()
    distances = (1-precision)**2 + (1-recall)**2
    PR_optimal_threshold = PR_thresholds[distances.argmin()]
    print("The optimal threshold for precision-recall optimization is %0.3f" %PR_optimal_threshold)
    y_pred_PR = (y_hat>PR_optimal_threshold).astype(float)
    FPR = np.sum(y_pred_PR[indices_0[0]])/len(indices_0[0])
    FNR = np.sum(1-y_pred_PR[indices_1[0]])/len(indices_1[0])
    print("The false negative rate is %0.4f" %FNR)
    print("The false positive rate is %0.4f" %FPR)
    
if __name__ == "__main__":
    main()
    #pass

