import torch

from torchscripts.classifiers import *
from torchscripts.utils import *
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score,confusion_matrix,balanced_accuracy_score,precision_recall_fscore_support,f1_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
from collections import Counter



def fairness_metrics(dataf, base, colstocompare, newcolnames):
    
    #tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    dataf['Sensitivity'] = dataf['tp'] / (dataf['tp']+dataf['fn'])
    dataf['Specificity'] = dataf['tn'] / (dataf['tn']+dataf['fp'])
    

    for i,coltocompare in enumerate(colstocompare):
        base_val = dataf.loc[dataf['Train'] == base, coltocompare].iloc[0]
        dataf[newcolnames[i]] = dataf[coltocompare].div(base_val).round(2)
    return dataf

def train_classifier(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data, labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs= model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / len(train_loader.dataset)
    return train_loss / len(train_loader), accuracy

def evaluate_classifier(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device, dtype=torch.long)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    accuracy = correct / len(val_loader.dataset)
    return val_loss / len(val_loader), accuracy
def train_best_image(model, device, criterion, optimizer, train_loader, valid_loader, num_epochs, patience, filename):
    best_val_accuracy = 0
    num_epochs_no_improvement = 0
    
    # Initialize lists to store losses and accuracies
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_classifier(model , train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate_classifier(model , valid_loader, criterion, device)
        
        # Append the losses and accuracies to their respective lists
        if epoch % 1 == 0:
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            num_epochs_no_improvement = 0
            save_best_model(model, epoch, val_accuracy, filename)
        
        best_val_accuracies.append(best_val_accuracy)
        num_epochs_no_improvement += 1
        if num_epochs_no_improvement >= patience:
            print("Early stopping triggered. No improvement in validation accuracy for {} consecutive epochs.".format(patience))
            break
    
    # Return the lists of losses, accuracies, and best_val_accuracies
    return train_losses, train_accuracies, val_losses, val_accuracies, best_val_accuracies
def save_best_model(model, epoch, accuracy, filename='best_model.pth'):
    print("Saving the best model (Epoch: {}, Validation Accuracy: {:.4f})...".format(epoch, accuracy))
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)





#metrics functions
def collect_imetrics(model, device, test_tensor,true_labels, kernel, filter,model_name):
        model_totest = model
        model_totest.to(device)
        model_totest.eval()
        with torch.no_grad():
            test_tensor = test_tensor.to(device)
            outputs = model_totest(test_tensor)
            predicted_probs = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_labels = np.argmax(predicted_probs, axis=1)
            #predicted_labels = predicted_labels.detach().cpu().numpy()
        
                                    
        model_names= np.tile(model_name, len(predicted_labels))
        
        acc =  accuracy_score(true_labels, predicted_labels)
        #balanced accuracy
        ba = balanced_accuracy_score(true_labels, predicted_labels )
        #binarize labels
        true_labels_binarized = label_binarize(true_labels, classes=np.arange(8))
        auc_score_ovr = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovr")
        auc_score_ovo = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovo")
        precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, predicted_labels)
        macro_average_f1_score = np.mean(f1_score)
        accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC(OVO)':auc_score_ovo, "AUC(OVR)":auc_score_ovr,'Macro-F1':macro_average_f1_score ,'Filter':filter,'Kernel':kernel }
        print('Model:', model_name, 'Filter', filter,'kernel:', kernel,'\tAcc:',"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'Macro-F1:', "{:.2f}".format(macro_average_f1_score)  ,'\tAUC(OVR):', "{:.2f}".format(auc_score_ovr),'\tAUC(OVO):', "{:.2f}".format(auc_score_ovo))
        acc_df = pd.DataFrame(data=accs, index=[0])
        return acc_df

def collect_tabmetrics(model, device, test_loader, kernel, filter,model_name):
        
        model_totest = model
        model_totest.to(device)
        model_totest.eval()
        true_labels, predicted_labels, predicted_probs = predict_labels(model_totest, test_loader, device)                                         
        model_names= np.tile(model_name, len(predicted_labels))

        
        acc =  accuracy_score(true_labels, predicted_labels)
        #balanced accuracy
        ba = balanced_accuracy_score(true_labels, predicted_labels )
        #binarize labels
        true_labels_binarized = label_binarize(true_labels, classes=np.arange(8))
        predicted_probs = np.array(predicted_probs)
        auc_score_ovr = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovr")
        auc_score_ovo = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovo")
        precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, predicted_labels)
        macro_average_f1_score = np.mean(f1_score)
        accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC(OVO)':auc_score_ovo, "AUC(OVR)":auc_score_ovr,'Macro-F1':macro_average_f1_score ,'Filter':filter,'Kernel':kernel }
        print('Model:', model_name,'Filter', filter,'kernel:', kernel, '\tAcc:',"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'Macro-F1:', "{:.2f}".format(macro_average_f1_score)  ,'\tAUC(OVR):', "{:.2f}".format(auc_score_ovr),'\tAUC(OVO):', "{:.2f}".format(auc_score_ovo))
        acc_df = pd.DataFrame(data=accs, index=[0])
        return acc_df

def predict_labels(model, loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device, dtype=torch.long)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            predicted_probs.extend(probabilities.cpu().numpy())


    return true_labels, predicted_labels, predicted_probs
class TabularDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].unsqueeze(0).unsqueeze(0)
        y = self.labels[idx]
        return x, y

def collect_dlmetrics(model, device, test_tensor,true_labels,model_name):
        model_totest = model
        model_totest.to(device)
        model_totest.eval()
        with torch.no_grad():
            test_tensor = test_tensor.to(device)
            outputs = model_totest(test_tensor)
            predicted_probs = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_labels = np.argmax(predicted_probs, axis=1)
            #predicted_labels = predicted_labels.detach().cpu().numpy()
        
                                    
        
        acc =  accuracy_score(true_labels, predicted_labels)
        #balanced accuracy
        ba = balanced_accuracy_score(true_labels, predicted_labels )
        #binarize labels
        true_labels_binarized = label_binarize(true_labels, classes=np.arange(8))
        auc_score_ovr = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovr")
        auc_score_ovo = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovo")
        precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, predicted_labels)
        macro_average_f1_score = np.mean(f1_score)
        accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC(OVO)':auc_score_ovo, "AUC(OVR)":auc_score_ovr,'Macro-F1':macro_average_f1_score }
        print('Model:', model_name,"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'Macro-F1:', "{:.2f}".format(macro_average_f1_score)  ,'\tAUC(OVR):', "{:.2f}".format(auc_score_ovr),'\tAUC(OVO):', "{:.2f}".format(auc_score_ovo))
        acc_df = pd.DataFrame(data=accs, index=[0])
        return acc_df
def predict_taxonnlabels(model, loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        for data, labels in loader:
            data = {k: v.to(device) for k, v in data.items()}
            labels = labels.to(device, dtype=torch.long)    
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            predicted_probs.extend(probabilities.cpu().numpy())


    return true_labels, predicted_labels, predicted_probs
def collect_taxonn(model, device, test_loader,model_name):
        
        model_totest = model
        model_totest.to(device)
        model_totest.eval()
        true_labels, predicted_labels, predicted_probs = predict_taxonnlabels(model_totest, test_loader, device)                                         
        model_names= np.tile(model_name, len(predicted_labels))

        
        acc =  accuracy_score(true_labels, predicted_labels)
        #balanced accuracy
        ba = balanced_accuracy_score(true_labels, predicted_labels )
        #binarize labels
        true_labels_binarized = label_binarize(true_labels, classes=np.arange(8))
        predicted_probs = np.array(predicted_probs)
        auc_score_ovr = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovr")
        auc_score_ovo = roc_auc_score(true_labels_binarized, predicted_probs, multi_class="ovo")
        precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, predicted_labels)
        macro_average_f1_score = np.mean(f1_score)
        accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC(OVO)':auc_score_ovo, "AUC(OVR)":auc_score_ovr,'Macro-F1':macro_average_f1_score  }
        print('Model:', model_name, '\tAcc:',"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'Macro-F1:', "{:.2f}".format(macro_average_f1_score)  ,'\tAUC(OVR):', "{:.2f}".format(auc_score_ovr),'\tAUC(OVO):', "{:.2f}".format(auc_score_ovo))
        acc_df = pd.DataFrame(data=accs, index=[0])
        return acc_df
class PhylumDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.labels = y
        self.groups = self.get_top_groups()
        self.grouped_data = self.group_data()
        self.image_transformers = {}
        self.input_sizes = {}
        for group in self.groups:
            if group == "other":
                group_cols = [col for col in self.X.columns if col.split('_')[1] not in self.groups[:-1]]
                #print(group_cols)
            else:
                group_cols = [col for col in self.X.columns if col.split('_')[1] == group]
           # print(group_cols.unique)
            group_data = self.X[group_cols]
            group_data = group_data.drop(group_data.index[-1])
            self.grouped_data[group] = group_data


    def get_top_groups(self):
        group_counts = Counter(self.X.iloc[-1])
        #print(group_counts)
        top_groups = [group[0] for group in group_counts.most_common(4)]
        #print(top_groups)
        return top_groups

    def group_data(self):
        grouped_data = {group: None for group in self.groups}
        other_indices = []

        for group in self.groups[:-1]:
            group_indices = self.X.columns[self.X.iloc[-1] == group].tolist()
            grouped_data[group] = self.X[group_indices].iloc[:-1].values.astype('float32')
            other_indices.extend(group_indices)

        other_indices = set(self.X.columns) - set(other_indices)
        grouped_data['other'] = self.X[list(other_indices)].iloc[:-1].values.astype('float32')

        return grouped_data

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
            tensor_data = {}
            self.label = torch.from_numpy(self.labels)
            label = torch.tensor(self.labels[index], dtype=torch.long)
            for group in self.groups:
                img_data = self.grouped_data[group].values[index].astype(np.float32)
                tensor_data[group] = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            return tensor_data, label
def train_taxonn(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):

        data = {k: v.to(device) for k, v in data.items()}
        labels = labels.to(device, dtype=torch.long)    

        optimizer.zero_grad()

        outputs= model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / len(train_loader.dataset)
    return train_loss / len(train_loader), accuracy
def evaluate_taxonn(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = {k: v.to(device) for k, v in data.items()}
            labels = labels.to(device, dtype=torch.long)    
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    accuracy = correct / len(val_loader.dataset)
    return val_loss / len(val_loader), accuracy
def train_best_taxonn(model, num_labs, dataset, device,criterion,optimizer,train_loader, valid_loader, num_epochs, patience):
    best_val_accuracy = 0  # Best validation accuracy so far
    num_epochs_no_improvement = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_taxonn(model , train_loader, optimizer, criterion, device)
        val_loss,val_accuracy = evaluate_taxonn(model , valid_loader, criterion, device)
        if epoch  == (num_epochs + 1) :
            print("Epoch: {} Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}".format(epoch, train_loss, train_accuracy*100, val_accuracy*100))
        # Check if the current validation accuracy is better than the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            num_epochs_no_improvement = 0
            if epoch == (num_epochs + 1):
                save_best_model(model, epoch, val_accuracy, "taxonn.pth")  # Save the best model
            else:
                save_best_model(model, epoch, val_accuracy,"taxonn.pth")  # Save the best model
        else:
            num_epochs_no_improvement += 1
        # Early stopping
        if num_epochs_no_improvement >= patience:
            print("Early stopping triggered. No improvement in validation accuracy for {} consecutive epochs.".format(patience))
            break

    encoders = {}
    for groups in dataset.groups:
        input_size = dataset.grouped_data[groups].shape[1]
        encoders[groups] = TaxoNNsub(input_size).to(device)
    best_model =TaxoNN(encoders,num_labs)
    best_model.load_state_dict(torch.load('taxonn.pth'))
    return(best_model)

def collect_dlmetrics_bin(model, device, test_tensor,true_labels,model_name):
    model_totest = model
    model_totest.to(device)
    model_totest.eval()
    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        outputs = model_totest(test_tensor)
        predicted_probs = F.softmax(outputs, dim=1).cpu().numpy()
        predicted_labels = np.argmax(predicted_probs, axis=1)
    acc =  accuracy_score(true_labels, predicted_labels)
    ba = balanced_accuracy_score(true_labels, predicted_labels )
    auc_score = roc_auc_score(true_labels, predicted_probs[:, 1]) # get probabilities of positive class
    f1 = f1_score(true_labels, predicted_labels)
    accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC':auc_score,'F1':f1 }
    print('Model:', model_name,'\tAcc:',"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'F1:', "{:.2f}".format(f1), '\tAUC:', "{:.2f}".format(auc_score))
    acc_df = pd.DataFrame(data=accs, index=[0])
    return acc_df

def collect_taxonn_bin(model, device, test_loader,model_name):
    model_totest = model
    model_totest.to(device)
    model_totest.eval()
    true_labels, predicted_labels, predicted_probs = predict_taxonnlabels(model_totest, test_loader, device)
    acc =  accuracy_score(true_labels, predicted_labels)
    ba = balanced_accuracy_score(true_labels, predicted_labels )
    auc_score = roc_auc_score(true_labels, np.array(predicted_probs)[:, 1]) # get probabilities of positive class
    f1 = f1_score(true_labels, predicted_labels)
    accs = {'Train': model_name,'Acc':acc,'Bacc':ba,'AUC':auc_score,'F1':f1 }
    print('Model:', model_name, '\tAcc:',"{:.2f}".format(acc), '\tBAcc:', "{:.2f}".format(ba),'F1:', "{:.2f}".format(f1), '\tAUC:', "{:.2f}".format(auc_score))
    acc_df = pd.DataFrame(data=accs, index=[0])
    return acc_df