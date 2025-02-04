import pandas as pd
import networkx as nx
import numpy as np
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
from torch.utils.data import random_split
from torch_geometric.data import Data
from models.GIN import GNN_GIN_Model
import time
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt




class Dataset:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:",self.device)
        self.data = None


    def load_dataset(self,folder,splits=[.35,.15,.50],split_type="normal",split_seed=0):  
        self.folder = folder
        accounts_df = pd.read_csv(folder+"/account_attributes.csv")
        transactions_df = pd.read_csv(folder+"/transactions.csv")

        nodes_df = accounts_df
        edges_df = transactions_df

        fp =np.round(nodes_df[nodes_df.node_isSar == 1.0].shape[0] / nodes_df.shape[0],4)*100

        print("loading dataset",folder,"|","length:",nodes_df.shape[0],"| fraud percentage (%):",fp)

        x_np = nodes_df.to_numpy()
        X = x_np[:,0:-2]
        
        self.feature_labels = nodes_df.columns.tolist()
        
        # Define your graph
        X = torch.tensor(X)  # (n x features)
        edge_index =  torch.stack([torch.tensor(edges_df.orig_acct.to_numpy()),torch.tensor(edges_df.bene_acct.to_numpy())],dim=-1).T  # Define your edge index
        edge_weight = torch.nn.functional.normalize(torch.tensor(edges_df.base_amt.to_numpy()),dim=0).long()
        y =  torch.tensor(nodes_df.node_isSar.to_numpy().astype(int),dtype=torch.long) # target values

        #normalization methods
        # method 1
        X = torch.nn.functional.normalize(X,dim=0).to(torch.float32)
        #dim 0 is vertical (features)
        #dim 1 is horizontal (nodes)

        train_size = int(splits[0] * len(y))  # 60% of the dataset for training
        val_size = int(splits[1] * len(y))    # 20% of the dataset for validation
        test_size = len(y) - train_size - val_size  # Remaining 20% for testing

        torch.manual_seed(split_seed)
        train_dataset, val_dataset, test_dataset = random_split(y, [train_size, val_size, test_size])

        new_trainset_indices = []
        if split_type == "imbalanced_method":
            a = np.array(train_dataset.indices)
            b = y[a]
            pos_samples_in_trainset = a[b==1]
            c = a[b==0]
            scalar = 5
            neg_samples_in_trainset = np.random.choice(len(c), len(pos_samples_in_trainset)*scalar, replace=False)
            new_trainset_indices = np.concatenate((pos_samples_in_trainset, neg_samples_in_trainset))
            train_dataset.indices = new_trainset_indices
        else:
            pass

        # Create masks for train, validation, and test sets
        self.train_mask = torch.zeros(len(y), dtype=torch.bool)
        self.val_mask = torch.zeros(len(y), dtype=torch.bool)
        self.test_mask = torch.zeros(len(y), dtype=torch.bool)

        self.train_mask[train_dataset.indices] = True
        self.val_mask[val_dataset.indices] = True
        self.test_mask[test_dataset.indices] = True
        
        # Load your data into PyTorch Geometric's Data class
        self.data = Data(X=X, edge_index=edge_index, edge_weight=edge_weight, y=y,train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask)
        self.data.to(self.device)


class Model:
    def __init__(self, data,save_model=False,save_model_path="default.pth", gridsearch_flag=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:",self.device)
        self.data = data
        self.optimizer = None
        self.criterion = None
        self.model = None
        self.lr = 0.005
        self.beta = 0.23
        self.w2 = [1,len(data.y==0)/len(data.y==1)]
        self.save_model = save_model
        self.save_model_path = save_model_path
        self.gridsearch_flag = gridsearch_flag
        self.gridsearch_results = {
            'test_f1':0,
            'test_auc':0,
            'test_precision':0,
            'test_recall':0,
            'test_epoch':0
        }

        if self.gridsearch_flag == False:
            self.writer = SummaryWriter()


    def load_model(self, model_name, K=1, F=8, additional_params={}):
        self.dataset_num_features = self.data.X.size()[1]
        self.dataset_num_classes = 2
        self.K = K
        self.F = F
        self.additional_params = additional_params
        self.model_name = model_name

        if model_name == "GIN":
            self.model = GNN_GIN_Model(hidden_size=self.F, input_size=self.dataset_num_features, output_size=self.dataset_num_classes, num_layers=self.K)
        elif model_name == "GAMLNet":
            print(self.additional_params)

            self.model = GNN_GAMLNET_Model(hidden_size1=self.additional_params['F1'],
                 hidden_size2=self.additional_params['F2'],
                 input_size1=len(self.additional_params['gin_feature_indices']), 
                 input_size2=self.dataset_num_features,
                 output_size=self.dataset_num_classes,
                 num_layers1=self.additional_params['K1'],
                 num_layers2=self.additional_params['K2'],
                 gin_feature_indices=self.additional_params['gin_feature_indices'])
        else:
            print("INVALID MODEL OPTION")
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.beta,1.0]).to(self.device))
        self.model.to(self.device)


    def train_model(self,epochs):
        print("model training starting...")
        st = time.time()
        best_f1_val_score = 0
        for epoch in range(0, epochs):
            self.model.train()
            self.optimizer.zero_grad()
            # We now give as input also the graph connectivity
            out = self.model(self.data.X, self.data.edge_index)
            
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        

            if epoch%10 == 0 and self.gridsearch_flag == False:
                train_acc, train_out, train_pred = self.test(self.data.train_mask,self.data)
                precision, recall, f1_score, _ = precision_recall_fscore_support(self.data.y[self.data.train_mask].cpu(), train_pred.cpu(), average='binary')
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
                self.writer.add_scalar('Accuracy/train',train_acc, epoch)
                self.writer.add_scalar('F1/train',f1_score, epoch)
                self.writer.add_scalar('Precision/train',precision, epoch)
                self.writer.add_scalar('Recall/train',recall, epoch)

            #loss = sigmoid_focal_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            if epoch%10 == 0:
                # validation

                val_acc, val_out, val_pred = self.test(self.data.val_mask,self.data)
                val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
                precision, recall, f1_score, _ = precision_recall_fscore_support(self.data.y[self.data.val_mask].cpu(), val_pred.cpu(), average='binary')
                
                if self.gridsearch_flag == False:
                    self.writer.add_scalar('Accuracy/validation',val_acc, epoch)
                    self.writer.add_scalar('Loss/validation',val_loss, epoch)
                    self.writer.add_scalar('F1/validation',f1_score, epoch)
                    self.writer.add_scalar('Precision/validation',precision, epoch)
                    self.writer.add_scalar('Recall/validation',recall, epoch)
            
                if f1_score > best_f1_val_score:
                    best_f1_val_score = f1_score
                    if self.save_model == True:
                        torch.save(self.model.state_dict(), self.save_model_path)
                    if self.gridsearch_flag == True: #THIS IS USUALLY USED IN COMBINATION WITH THE GRIDSEARCH FILE. this is only a component of the gridsearch
                        test_acc, test_out, test_pred = self.test(self.data.test_mask,self.data)
                        precision, recall, f1_score, _ = precision_recall_fscore_support(self.data.y[self.data.test_mask].cpu(), test_pred.cpu(), average='binary')
                        y_true = self.data.y[self.data.test_mask].cpu()
                        y_pred = test_pred.detach().cpu().numpy()
                        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                        roc_auc = auc(fpr, tpr)
                        self.gridsearch_results['test_f1'] = f1_score
                        self.gridsearch_results['test_precision'] = precision
                        self.gridsearch_results['test_recall'] = recall
                        self.gridsearch_results['test_auc'] = roc_auc
                        self.gridsearch_results['test_epoch'] = epoch
                        self.run_results = self.gridsearch_results #simply show the run results outside of the full gridsearch setting
                '''
                if f1_score > self.best_f1_from_gs:
                    tpfp = self.check_topology_performance('v2/128K_05_v2_2', self.data)
                    results = {
                    'tp':tpfp[0],
                    'fp':tpfp[1],
                    }
                    print('########################################')
                    print('########################################')
                    print('########################################')
                    print('########################################')
                    print(results)
                    print(f1_score)
                    print('########################################')
                    print('########################################')
                    print('########################################')
                    print('########################################')
                '''        

        et = time.time()
        elapsed_time = np.round(et - st,2)
        print("model training done. elapsed time:", elapsed_time, "seconds")
        

    def test(self,mask,data):
        self.model.eval()
        out = self.model(data.X, data.edge_index)
        #out = model(data.X, data.edge_index,data.edge_weight)
        pred = out.argmax(dim=1)
        test_correct = pred[mask] == data.y[mask]
        test_acc = int(test_correct.sum()) / int(mask.sum())
        test_out = out[mask]
        test_pred = pred[mask]
        return test_acc, test_out, test_pred

    def test_model(self,data):
        test_acc, test_out, test_pred = self.test(data.test_mask,data)
        precision, recall, f1_score, _ = precision_recall_fscore_support(data.y[data.test_mask].cpu(), test_pred.cpu(), average='binary')

        print('test acc:',np.round(test_acc*100,2))
        print('test precision:',np.round(precision*100,2))
        print('test recall:',np.round(recall*100,2))
        print('test f1-score:',np.round(f1_score*100,2))

    def check_topology_performance(self, folder, data):
        ### Here we colour nodes

        # col 1
        # 1 -> in training/val set
        # 2 -> true positive
        # 3 -> true negative
        # 4 -> false positive
        # 5 -> false negative
        test_acc, test_out, test_pred = self.test(data.test_mask,data)
        test_pred = test_pred.detach().cpu()
        y_true = data.y[data.test_mask].cpu()
        precision, recall, f1_score, _ = precision_recall_fscore_support(data.y[data.test_mask].cpu(), test_pred.cpu(), average='binary')
        
        accounts_df = pd.read_csv(folder+"/account_attributes.csv")

        indicies = accounts_df.index

        out = np.array(test_pred)  # Convert 'out' to a NumPy array for easier indexing
        true_labels = np.array(y_true)  # Convert 'true_labels' to a NumPy array for easier indexing

        # True Positives (TP)
        tp_indices = np.where((out == 1) & (true_labels == 1))[0]
        #print("true positives =",len(tp_indices))
        tp_indices = indicies[data.test_mask.cpu().numpy()][tp_indices]

        # True Negatives (TN)
        tn_indices = np.where((out == 0) & (true_labels == 0))[0]
        #print("true negatives =",len(tn_indices))
        tn_indices = indicies[data.test_mask.cpu().numpy()][tn_indices]

        # False Positives (FP)
        fp_indices = np.where((out == 1) & (true_labels == 0))[0]
        #print("false positives =",len(fp_indices))
        fp_indices = indicies[data.test_mask.cpu().numpy()][fp_indices]

        # False Negatives (FN)
        fn_indices = np.where((out == 0) & (true_labels == 1))[0]
        #print("false negatives =",len(fn_indices))
        fn_indices = indicies[data.test_mask.cpu().numpy()][fn_indices]

        '''
        accounts_df['trained_color'] = "#000000"
        accounts_df['trained_color'][train_mask.numpy()] = "#343B46"
        accounts_df['trained_color'][val_mask.numpy()] = "#343B46"
        accounts_df['trained_color'][tp_indices] = "#eb4034"
        accounts_df['trained_color'][tn_indices] = "#1ad623"
        accounts_df['trained_color'][fp_indices] = "#26e0be"
        accounts_df['trained_color'][fn_indices] = "#fae01e"
        '''

        accounts_df['trained_color'] = 1
        accounts_df['trained_color'][data.train_mask.cpu().numpy()] = 2
        accounts_df['trained_color'][data.val_mask.cpu().numpy()] = 2
        accounts_df['trained_color'][tp_indices] = 3
        accounts_df['trained_color'][tn_indices] = 4
        accounts_df['trained_color'][fp_indices] = 5
        accounts_df['trained_color'][fn_indices] = 6

        #accounts_df['id'] = indicies
        accounts_df['id'] = (accounts_df['id']).astype(int)

        #accounts_df.to_csv('../datasets/60K_01_sar_count/account_attributes_trained_vis_SAGE.csv',index=0)

        #np.sum(accounts_df['trained_color'] == 0), len(train_dataset), len(val_dataset), len(test_dataset)

        sar_df = pd.read_csv(folder+"/alert_accounts.csv")

        ids_cycle = sar_df["acct_id"][sar_df["alert_type"]=='cycle'].to_numpy() #super important to convert to numpy array!!!!! keeping as pandas series messes up all indexing!!!
        ids_fan_in = sar_df["acct_id"][sar_df["alert_type"]=='fan_in'].to_numpy()
        ids_fan_out = sar_df["acct_id"][sar_df["alert_type"]=='fan_out'].to_numpy()
        ids_gather_scatter = sar_df["acct_id"][sar_df["alert_type"]=='gather_scatter'].to_numpy()
        ids_scatter_gather = sar_df["acct_id"][sar_df["alert_type"]=='scatter_gather'].to_numpy()
        ids_bipartite = sar_df["acct_id"][sar_df["alert_type"]=='bipartite'].to_numpy()
        ids_stack = sar_df["acct_id"][sar_df["alert_type"]=='stack'].to_numpy()
        set(sar_df["alert_type"])

        #print('ids_cycle',ids_cycle)

        ids_test_set = accounts_df['id'][data.test_mask.cpu().numpy()].astype(int).to_numpy()
        labels_test_set = data.y[data.test_mask].cpu().numpy()
        predictions_test_set = test_pred.numpy()
        correct_pos_pred = np.logical_and(labels_test_set, predictions_test_set) # is can only equal 1 if both label and precition were 1

        pred_cycle = correct_pos_pred[[e in ids_cycle for e in ids_test_set]]
        pred_fan_in = correct_pos_pred[[e in ids_fan_in for e in ids_test_set]]
        pred_fan_out = correct_pos_pred[[e in ids_fan_out for e in ids_test_set]]
        pred_gather_scatter = correct_pos_pred[[e in ids_gather_scatter for e in ids_test_set]]
        pred_scatter_gather = correct_pos_pred[[e in ids_scatter_gather for e in ids_test_set]]
        pred_bipartite = correct_pos_pred[[e in ids_bipartite for e in ids_test_set]]
        pred_stack = correct_pos_pred[[e in ids_stack for e in ids_test_set]]

        topologies = ('Cycle', 'Fan In', 'Fan Out', 'Gather Scatter', 'Scatter Gather', 'Bipartite', 'Stack')

        tp = np.round(np.array([np.sum(pred_cycle)/len(pred_cycle), np.sum(pred_fan_in)/len(pred_fan_in), np.sum(pred_fan_out)/len(pred_fan_out), np.sum(pred_gather_scatter)/len(pred_gather_scatter), np.sum(pred_scatter_gather)/len(pred_scatter_gather), np.sum(pred_bipartite)/len(pred_bipartite), np.sum(pred_stack)/len(pred_stack)]),2)
        fp = np.ones(len(tp)) - tp
        topology_counts = {
            'True Positive': tp,
            'False Positive': fp,
        }
        width = 0.6  # the width of the bars: can also be len(X) sequence
        #print(1)
            
        fig, ax = plt.subplots()
        #print(2)
        bottom = np.zeros(7)
        #print(3)
        for top, top_count in topology_counts.items():
            print(topologies, top_count)
            p = ax.bar(topologies, top_count, width, label=top, bottom=bottom)
            bottom += top_count

            ax.bar_label(p, label_type='center')
        #print(4)
        ax.set_title('Model Performance by Topology Type')
        #print(5)
        ax.legend()
        #print(6)

        plt.show()
        #plt.savefig('topologies_perf.png')
        
        return [tp,fp]