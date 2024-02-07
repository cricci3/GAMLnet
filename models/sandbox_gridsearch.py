import model_helper as mh
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import time
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import json
from datetime import datetime

def rf_gridsearch(ds):
    param_grid = {'bootstrap': [True, False],
        'max_depth': [10, 30, 50, 70, 90, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [20, 50, 100]}
    
    trainX = ds.data.x[ds.data.train_mask + ds.data.val_mask].detach().cpu().numpy()
    trainY = ds.data.y[ds.data.train_mask + ds.data.val_mask].detach().cpu().numpy()

    testX = ds.data.x[ds.data.test_mask].detach().cpu().numpy()
    testY = ds.data.y[ds.data.test_mask].detach().cpu().numpy()

    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(trainX, trainY)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    #clf = RandomForestClassifier(bootstrap=best_params['bootstrap'],max_depth=best_params['max_depth'],max_features=best_params['max_features'],min_samples_leaf=best_params['min_samples_leaf'],min_samples_split=best_params['min_samples_split'],n_estimators=best_params['n_estimators'])
    #st = time.time()
    #clf.fit(trainX, trainY)
    #et = time.time()
    #elapsed_time = et - st
    #best_model = clf
    
    test_pred = best_model.predict(testX)
    y_pred_prob = best_model.predict_proba(testX)
    y_pred_prob = y_pred_prob[:,1]

    precision, recall, f1_score, _ = precision_recall_fscore_support(testY, test_pred, average='binary')                     
    fpr, tpr, thresholds = roc_curve(testY, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    results = {
        'test_f1':f1_score,
        'test_auc':roc_auc,
        'test_precision':precision,
        'test_recall':recall,
        'bootstrap':best_params['bootstrap'],
        'max_depth':best_params['max_depth'],
        'n_estimators':best_params['n_estimators'],
        'max_features':best_params['max_features'],
        'min_samples_leaf':best_params['min_samples_leaf'],
        'min_samples_split':best_params['min_samples_split']
    }
    return results


def xg_gridsearch(ds):
    trainX = ds.data.x[ds.data.train_mask + ds.data.val_mask].detach().cpu().numpy()
    trainY = ds.data.y[ds.data.train_mask + ds.data.val_mask].detach().cpu().numpy()

    testX = ds.data.x[ds.data.test_mask].detach().cpu().numpy()
    testY = ds.data.y[ds.data.test_mask].detach().cpu().numpy()

    clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    params = {
        "gamma": [0,0.2,0.4,0.6],
        'max_depth': [10, 30, 50, 70, 90, None],
        'n_estimators': [20, 50, 100]}

    grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=3, n_jobs=-1)
    grid_search.fit(trainX, trainY)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    #clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42,gamma=best_params["gamma"], max_depth=best_params["max_depth"],n_estimators=best_params["n_estimators"])
    #st = time.time()
    #clf.fit(trainX, trainY)
    #et = time.time()
    #elapsed_time = et - st

    test_pred = best_model.predict(testX)
    test_pred = np.round(test_pred)
    print(np.sum(test_pred))

    y_pred_prob = best_model.predict_proba(testX)
    y_pred_prob = y_pred_prob[:,1]

    #test_acc, test_out, test_pred  = model_gs.test(data.test_mask)
    precision, recall, f1_score, _ = precision_recall_fscore_support(testY, test_pred, average='binary')                     
    fpr, tpr, thresholds = roc_curve(testY, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    results = {
        'test_f1':f1_score,
        'test_auc':roc_auc,
        'test_precision':precision,
        'test_recall':recall,
        'max_depth':best_params['max_depth'],
        'n_estimators':best_params['n_estimators'],
        'gamma':best_params['gamma']
    }
    return results

def gnn_gridsearch(ds_split_seeds, dataset_folder, method, gnn_params, ds_name, results_path, gs_results):
    epochs = 5000
    results = {
        'test_f1':None,
        'test_auc':None,
        'test_precision':None,
        'test_recall':None,
        'lr':None,
        'w':None,
        'K':None,
        'F':None,
        'K1':None,
        'K2':None,
        'F1':None,
        'F2':None,
        'epoch':None
    }
    best_average_test_f1 = 0
    count = 1
    for lr in gnn_params['lr']:
        for w in gnn_params['w']:
            if method == "GINSAGE":
                
                for K1 in gnn_params['K1']:
                    for K2 in gnn_params['K2']:
                        for F1 in gnn_params['F1']:
                            for F2 in gnn_params['F2']:
                                f1_scores_for_group = []
                                best_epochs_for_group = []
                                auc_scores_for_group = []
                                precision_scores_for_group = []
                                recall_scores_for_group = []
                                for seed in ds_split_seeds:
                                    print(method, count, 'out of', len(gnn_params['lr'])*len(gnn_params['w'])*len(gnn_params['K1'])*len(gnn_params['K2'])*len(gnn_params['F1'])*len(gnn_params['F2'])*len(ds_split_seeds))
                                    
                                    ds = mh.Dataset()
                                    ds.load_dataset(folder=dataset_folder+ds_name,splits=[0.5,0.2,0.3],split_type="normal",split_seed=seed)
                                    gin_feature_indices = [ds.feature_labels.index(item) for item in ['node_deg_out_unique', 'node_deg_in_unique']]
                                    
                                    model = mh.Model(ds.data, gridsearch_flag=True)
                                    
                                    additional_params = {'K1':K1, 'K2':K2, 'F1':F1, 'F2':F2, 'gin_feature_indices':gin_feature_indices}
                                    model.w = w
                                    model.lr = lr
                                    model.load_model("GINSAGE",additional_params=additional_params)
                                    model.train_model(epochs=epochs)
                                    f1_scores_for_group.append(model.gridsearch_results['test_f1'])
                                    best_epochs_for_group.append(model.gridsearch_results['test_epoch'])
                                    auc_scores_for_group.append(model.gridsearch_results['test_auc'])
                                    precision_scores_for_group.append(model.gridsearch_results['test_precision'])
                                    recall_scores_for_group.append(model.gridsearch_results['test_recall'])
                                    count+=1

                                if np.mean(f1_scores_for_group) > best_average_test_f1:
                                    best_average_test_f1 = np.mean(f1_scores_for_group)
                                    results = {
                                    'mean_f1':np.mean(f1_scores_for_group),
                                    'test_f1':f1_scores_for_group,
                                    'test_auc':auc_scores_for_group,
                                    'test_precision':precision_scores_for_group,
                                    'test_recall':recall_scores_for_group,
                                    'epoch':best_epochs_for_group,
                                    'lr':lr,
                                    'w':w,
                                    'K':None,
                                    'F':None,
                                    'K1':K1,
                                    'K2':K2,
                                    'F1':F1,
                                    'F2':F2
                                    }
                                for ele in results:
                                    gs_results[ds_name][method][ele] = results[ele]
                                    print(ele,results[ele])
                                with open(results_path, 'w') as file:
                                    json.dump(gs_results, file, indent=2)
                                    
                                
            else:
                for K in gnn_params['K']:
                    for F in gnn_params['F']:
                        f1_scores_for_group = []
                        best_epochs_for_group = []
                        auc_scores_for_group = []
                        precision_scores_for_group = []
                        recall_scores_for_group = []
                        for seed in ds_split_seeds:
                            print(method, count, 'out of', len(gnn_params['lr'])*len(gnn_params['w'])*len(gnn_params['K'])*len(gnn_params['F'])*len(ds_split_seeds))


                            ds = mh.Dataset()
                            ds.load_dataset(folder=dataset_folder+ds_name,splits=[0.5,0.2,0.3],split_type="normal",split_seed=seed)

                            model = mh.Model(ds.data, gridsearch_flag=True)
                            model.w = w
                            model.lr = lr
                            model.load_model(method,K=K,F=F)
                            model.train_model(epochs=epochs)
                            f1_scores_for_group.append(model.gridsearch_results['test_f1'])
                            best_epochs_for_group.append(model.gridsearch_results['test_epoch'])
                            auc_scores_for_group.append(model.gridsearch_results['test_auc'])
                            precision_scores_for_group.append(model.gridsearch_results['test_precision'])
                            recall_scores_for_group.append(model.gridsearch_results['test_recall'])
                            count+=1

                        if np.mean(f1_scores_for_group) > best_average_test_f1:
                            best_average_test_f1 = np.mean(f1_scores_for_group)
                            results = {
                            'mean_f1':np.mean(f1_scores_for_group),
                            'test_f1':f1_scores_for_group,
                            'test_auc':auc_scores_for_group,
                            'test_precision':precision_scores_for_group,
                            'test_recall':recall_scores_for_group,
                            'epoch':best_epochs_for_group,
                            'lr':lr,
                            'w':w,
                            'K':K,
                            'F':F,
                            'K1':None,
                            'K2':None,
                            'F1':None,
                            'F2':None
                            }
                        for ele in results:
                            gs_results[ds_name][method][ele] = results[ele]
                            print(ele,results[ele])
                        with open(results_path, 'w') as file:
                            json.dump(gs_results, file, indent=2)
                            
                        

    return results


gnn_params = {
    'lr':[0.001,0.005,0.01,0.05],
    'w':[0.4],
    'K':[2,3,4,5,6],
    'F':[8,16,32,64],
    'K1':[1,2,3],
    'K2':[4,5,6],
    'F1':[8,16],
    'F2':[16,32,64]
}
gnn_params = {
    'lr':[0.01, 0.005],
    'w':[0.44],
    'K':[2,3],
    'F':[32,16,8],
    'K1':[2,3],
    'K2':[4,5,6],
    'F1':[8,16,32],
    'F2':[16,32,64]
}
#'w':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],

gnn_params = {
    'lr':[0.005],
    'w':[.44],
    'K':[5],
    'F':[16],
    'K1':[3],
    'K2':[5],
    'F1':[8],
    'F2':[32]
}

#loop format
#loop through datasets
#loop through models
#gridsearch model

dataset_folder = 'cscs/'
#dataset_folder = 'thesis_datasets/'
datasets = ['16K_5_v2']
methods = ['SAGE','GINSAGE','XG','RF']
#methods = ['GIN','XG','RF']
methods = ['GIN']
#methods = ['GINSAGE']
#methods = ['RF','XG']

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H-%M-%S")
result_string = f"{current_date}_{current_time}"

#results_path = 'gridsearch_results.json'
results_path = 'gridsearch_results/gridsearch_results_'+result_string+'.json'


gs_results = {}
try:
    # Try to open the existing JSON file
    with open(results_path, 'r') as file:
        gs_results = json.load(file)
        print("Existing JSON gridsearch results file loaded.")
except FileNotFoundError:
    # If the file doesn't exist, create an empty dictionary
    gs_results = {}
    print("JSON gridsearch results file not found. Creating a new one.")
    with open(results_path, 'w') as file:
        json.dump(gs_results, file, indent=2)

for ds_name in datasets:
    split_seeds = [0,111,222]
    ds = mh.Dataset()
    ds.load_dataset(folder=dataset_folder+ds_name,splits=[0.5,0.2,0.3],split_type="normal")
    print('Running Dataset',ds_name)
    if ds_name not in gs_results:
        gs_results[ds_name] = {}

    for method in methods:
        model_path = './best_models/'+dataset_folder+ds_name+'_'+method+'.pth'
        print('    Running method',method)
        #grid search
        
        gs_results[ds_name][method] = {}
        
        results = {}
        if method in ['GIN','SAGE','MPNN','GCN','GAT','GINSAGE']:
            model = mh.Model(ds.data, gridsearch_flag=True)
            results = gnn_gridsearch(split_seeds, dataset_folder, method, gnn_params, ds_name, results_path, gs_results)
            '''
            try:
                results = gnn_gridsearch(model, ds, method, gnn_params)
            except:
                print(method,'on dataset',ds_name,'has crashed. Continuing to next seach.')
            '''
        elif method == 'XG':
            results = xg_gridsearch(ds)
        elif method == 'RF':
            results = rf_gridsearch(ds)
        else:
            print('method',method,'does not exist!')
        
        for ele in results:
            gs_results[ds_name][method][ele] = results[ele]
            print(ele,results[ele])

        with open(results_path, 'w') as file:
            json.dump(gs_results, file, indent=2)






