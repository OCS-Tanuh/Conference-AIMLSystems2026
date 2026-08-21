import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import time
import numpy as np
import csv
import os
import random
import copy
from sklearn.metrics import confusion_matrix
from transformers import MobileViTV2ForImageClassification
from torchmetrics.classification import AUROC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def set_seed(seed: int = 32) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Declare a list of 10 random seeds to be used for training
#seeds = [A list of 10 random seeds]

def get_metrics(cnf):
    spec = cnf[0,0]/(cnf[0,0]+cnf[0,1])
    sen = cnf[1,1]/(cnf[1,0]+cnf[1,1])
    return sen, spec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 2
batch_size = 64
num_epochs = 50

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        eps = 1e-8
        sgn = torch.sign(inputs)
        inputs = torch.clamp(abs(inputs), eps, 1-eps)
        inputs = sgn * inputs
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets.float())
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss)
        for k in range(len(targets)):
            loss[k] = loss[k] * self.alpha[int(targets[k])]
        return loss.mean()



def get_undersampled_loader(x_train, y_train, batch_size, epoch):
    rus = RandomUnderSampler(random_state=epoch)
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_train_res, y_train_res = rus.fit_resample(x_train_reshaped, y_train)
    x_train_res = x_train_res.reshape(x_train_res.shape[0], 3, 256, 256)
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train_res), torch.from_numpy(y_train_res))
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dl

def get_oversampled_loader(x_train, y_train, batch_size, epoch):
    ros = RandomOverSampler(random_state=epoch)
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_train_res, y_train_res = ros.fit_resample(x_train_reshaped, y_train)
    x_train_res = x_train_res.reshape(x_train_res.shape[0], 3, 256, 256)
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train_res), torch.from_numpy(y_train_res))
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dl



def train_model(model_ft, criterion, optimizer_ft, inst, num_epochs, t_schedular):
    best_loss = 100.0
    prev_prod = 0.0
    best_sen = 0.0
    best_val_auc = 0.0
    t_acc = 0.0
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        t_l=0
        tac=0
        ts=0
        model_ft.train()
        train_preds_ls = []
        train_labels_ls = []
        train_pred_scores = []
        for images, labels in train_dl:
            images = images.float().to(device)
            images = images[:,[2,1,0],:,:]
            labels = labels.float().to(device)
            outputs = model_ft(images).logits.squeeze()
            loss = criterion(outputs, labels)
            outputs = nn.Sigmoid()(outputs)
            t_l += loss.item()
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            tac += torch.sum(torch.round(outputs)==labels).item()
            ts += labels.shape[0]
            train_preds_ls = np.append(train_preds_ls, torch.round(outputs).cpu().detach().numpy())
            train_labels_ls = np.append(train_labels_ls, labels.cpu().detach().numpy())
            train_pred_scores = np.append(train_pred_scores, outputs.cpu().detach().numpy())
        tl = t_l/len(train_dl)
        t_acc= tac/ts
        cnf_tr = confusion_matrix(train_labels_ls, train_preds_ls)
        train_auc = AUROC(task="binary")(torch.tensor(train_pred_scores), torch.tensor(train_labels_ls)).item()
        sen_tr, spec_tr = get_metrics(cnf_tr)
        print('Train: Loss: {:.4f} Acc: {:.4f}'.format(tl,t_acc), 'Sensitivity: {:.4f} Specificity: {:.4f}'.format(sen_tr, spec_tr), 'Train AUC: {:.4f}'.format(train_auc))
        
        #validation
        acc=0
        v_l=0
        tot=0
        val_preds_ls = []
        val_labels_ls = []
        val_pred_scores = []
        model_ft.eval()
        for images,labels in valid_dl:
            images = images.float().to(device)
            images = images[:,[2,1,0],:,:]
            labels = labels.float().to(device)
            with torch.no_grad():
                outputs = model_ft(images).logits.squeeze()
                val_loss = criterion(outputs, labels)
                outputs = nn.Sigmoid()(outputs)
                val_preds = torch.round(outputs)
                ca=(torch.sum(val_preds == labels)).item()
                tot+=labels.size(0)
                acc+=ca
                v_l+=val_loss.item()
                val_preds_ls = np.append(val_preds_ls, val_preds.cpu().detach().numpy())
                val_pred_scores = np.append(val_pred_scores, outputs.cpu().detach().numpy())
                val_labels_ls = np.append(val_labels_ls, labels.cpu().detach().numpy())
                best_train_auc = train_auc
        cnf = confusion_matrix(val_labels_ls, val_preds_ls)
        sen, spec = get_metrics(cnf)
        vl=v_l/len(valid_dl)                
        va=acc/tot
        prod = sen*spec
        val_auc = AUROC(task="binary")(torch.tensor(val_pred_scores), torch.tensor(val_labels_ls)).item()
        t_schedular.step(prod)
        print('Val: Loss: {:.4f} Acc: {:.4f}  Sensitivity: {:.4f} Specificity: {:.4f}, AUC: {:.4f}'.format(vl,va,sen,spec,val_auc))
        print("sum val sen & spec: ", sen+spec) 
        print(t_schedular._last_lr)
        
        if prod > prev_prod and tl < best_loss:
            best_epch = epoch
            best_val_auc = val_auc
            prev_prod = prod
            best_sen = sen
            best_spec = spec
            tr_acc = t_acc
            best_tr_sen = sen_tr
            best_tr_spec = spec_tr
            best_loss = tl
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            vl_loss = vl
            vl_acc = va
            best_train_auc = train_auc
            torch.save(best_model_wts, f'path/to/save/model/weights/_{inst}.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return best_model_wts, best_loss, tr_acc, best_tr_sen, vl_loss, vl_acc, best_sen, best_spec, best_tr_spec, best_val_auc, best_train_auc, best_epch

def test_model(model_ft, model_wts, criterion_ft):
    acc=0
    v_l=0
    tot=0
    test_preds= []
    test_labels= []
    test_pred_scores = []
    model_ft.load_state_dict(model_wts)
    model_ft.eval()
    for images,labels in test_dl:
        images = images.float().to(device)
        images = images[:,[2,1,0],:,:]
        labels = labels.float().to(device)
        with torch.no_grad():
            outputs = model_ft(images).logits.squeeze()
            if outputs.shape == ():
                outputs = torch.unsqueeze(outputs, 0)
            val_loss = criterion_ft(outputs, labels)
            outputs = nn.Sigmoid()(outputs)
            te_pred = torch.round(outputs)
            ca=(torch.sum(te_pred == labels)).item()
            tot+=labels.size(0)
            acc+=ca
            v_l+=val_loss.item()
            test_preds = np.append(test_preds, te_pred.cpu().detach().numpy())
            test_labels = np.append(test_labels, labels.cpu().detach().numpy())
            test_pred_scores = np.append(test_pred_scores, outputs.cpu().detach().numpy())
    tl=v_l/len(test_dl)                
    ta=acc/tot
    print('Test: Loss: {:.4f} Acc: {:.4f} '.format(tl,ta)) 
    mat = confusion_matrix(test_labels, test_preds)
    test_auc = AUROC(task="binary")(torch.tensor(test_pred_scores), torch.tensor(test_labels)).item()
    sen, spec = get_metrics(mat)
    return ta, sen, spec, tl, test_auc


with h5py.File(r'name_of_the_h5_file_class_0.h5', 'r') as hf:
    x_train0 = hf['x_train'][:]
    y_train0 = hf['y_train'][:]
    x_val = hf['x_val'][:]
    y_val = hf['y_val'][:]
    x_test = hf['x_test'][:]
    y_test = hf['y_test'][:]
    hf.close()

with h5py.File(r'name_of_the_h5_file_class_1.h5', 'r') as hf:
    x_train1 = hf['x_train'][:]
    y_train1 = hf['y_train'][:]
    hf.close()


x_train = np.concatenate((x_train0, x_train1), axis=0)
y_train = np.concatenate((y_train0, y_train1), axis=0)

val_data = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
valid_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size,  shuffle=True)

test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
test_dl = torch.utils.data.DataLoader(test_data, batch_size = batch_size,  shuffle=True)

train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size,  shuffle=True)

class_counts = np.bincount(y_train)
total_samples = len(y_train)

class_weights = []
for count in class_counts:
    weight = 1 / (count / total_samples)
    class_weights.append(weight)

csv_file = 'path/to/to/store/results.csv'
csv_columns = ['Accuracy', 'Sensitivity', 'Specificity', 'Loss', 'Val Sensitivity', 'Val Accuracy', 'Train Sensitivity', 'Train Accuracy', 'Train Loss', 'Val Loss',
            'Val Specificity', 'Train Specificity', 'Val AUC', 'Test AUC', 'Train AUC', 'Best Train Epoch']
with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    csvfile.close()

since = time.time()
for i in range(10):
    set_seed(seeds[i])
    model = MobileViTV2ForImageClassification.from_pretrained('apple/mobilevitv2-1.0-imagenet1k-256')
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    model = model.to(device)      
    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=0.001)
    pos_weight = torch.tensor(class_weights[1]/class_weights[0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-7)

    model_wts, tr_loss, tr_acc, tr_sen, vl_loss, vl_acc, vl_sen, vl_spec, tr_spec, val_auc, train_auc, best_epch = train_model(model, criterion, optimizer, i, num_epochs, schedular)
    ta, sen, spec, tl, test_auc = test_model(model, model_wts, criterion)
    print(f'Test acc: {ta}, Test sen: {sen}, Test spec: {spec}, Test loss: {tl}, Val sen: {vl_sen}, Train sen: {tr_sen}, Train loss: {tr_loss}, Val loss: {vl_loss}','\n',
            f'Train acc: {tr_acc}, Val acc: {vl_acc}', f'Train spec: {tr_spec}, Val spec: {vl_spec}', f'Val AUC: {val_auc}', f'Test AUC: {test_auc}', '\n',
            f'Train AUC: {train_auc}', f'Best Train Epoch: {best_epch}')
    
    dict_data = {'Accuracy': ta, 'Sensitivity': sen, 'Specificity': spec, 'Loss': tl, 'Val Sensitivity': vl_sen, 'Val Accuracy': vl_acc, 
                    'Train Accuracy': tr_acc, 'Train Sensitivity': tr_sen, 'Train Loss': tr_loss, 'Val Loss': vl_loss,
                    'Val Specificity': vl_spec, 'Train Specificity': tr_spec, 'Val AUC': val_auc, 'Test AUC': test_auc, 'Train AUC': train_auc, 'Best Train Epoch': best_epch}
    with open(csv_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(dict_data)
        csvfile.close() 
    print(f'Iteration {i} done')
    torch.cuda.empty_cache()
    del model, model_wts, criterion, optimizer
time_elapsed = time.time() - since
print('Total time is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
