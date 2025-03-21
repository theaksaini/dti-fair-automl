import tpot
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
from tpot.search_spaces.nodes import *
from tpot.search_spaces.pipelines import *
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from functools import partial
import pickle
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import networkx as nx
from tpot import GraphPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time
import os
import utils
import random
import traceback


class CustonmKFold:
    def __init__(self, groups=None, n_splits=3):
        self.n_splits = n_splits
        self.groups = groups

    def split(self, X, y):
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits) 
        return sgkf.split(np.array(X), y, self.groups)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
def loop_through_tasks(experiments, data_file, base_save_folder, num_runs, ga_params):
    assert os.path.isdir(base_save_folder), f"Folder to save results does not exist: {base_save_folder}"
    
    for r in range(num_runs):
        for e, exp in enumerate(experiments):

            save_folder = f"{base_save_folder}/{r}_{exp}"
            time.sleep(random.random()*5)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            else:
                continue

            print("working on ")
            print(save_folder)

            print("loading data")
            super_seed = e*1000+r
            print("Super Seed : ", super_seed)

            X_train, X_test, y_train, y_test, X_train_groups, X_test_groups = utils.retrieve_data(data_file)
            sens_features = ['PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
                                 'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
                                 'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
                                 'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE']
                
            print("starting ml")
                                
            try:  
                print("Starting the fitting process. ")
                if exp=='Regular Feature Selection':
                    cv = StratifiedGroupKFold(n_splits=10) 
                    objective_names = ['balanced_accuracy'] + ['fpr'] + ['subgroup_fnr_loss']
                    objective_weights = [1.0] + [-1.0] + [-1.0]
                    objective_scorer = partial(utils.custom_objectives, X=X_train, X_prime=X_train.loc[:,sens_features], y= y_train,cv=cv, objectives=objective_names, groups=X_train_groups)
                    objective_scorer.__name__ = 'compressed'
                    

                elif exp=='Fairness Aware Feature Selection':
                    cv = StratifiedGroupKFold(n_splits=10) 
                    objective_names = ['balanced_accuracy'] + ['fpr'] 
                    objective_weights = [1.0] + [-1.0] 
                    objective_scorer = partial(utils.custom_objectives, X=X_train,X_prime=X_train.loc[:,sens_features],y= y_train,cv=cv, objectives=objective_names, groups=X_train_groups)
                    objective_scorer.__name__ = 'compressed'

                else:
                    raise ValueError(f"Experiment {exp} not recognized. Please use one of the following: {experiments}")
                    
                # Define the pipeline
                est = tpot.TPOTEstimator(search_space = utils.get_pipeline_space(n_features=X_train.shape[1], seed=super_seed),
                              scorers=[],
                              scorers_weights=[],
                              cv= sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=super_seed), # not used
                              other_objective_functions = [objective_scorer],
                              other_objective_functions_weights =  objective_weights,
                              objective_function_names =  objective_names,
                              generations=ga_params['generations'], 
                              population_size=ga_params['population_size'],
                              random_state=super_seed, verbose=5,
                              max_time_mins=None,
                              classification=True,
                              n_jobs=ga_params['n_jobs'],
                              max_eval_time_mins=10)

                # Execute the pipeline
                est.fit(X_train, y_train)
                print("Ending the fitting process. ")

                train_score = utils.evaluate_objective_functions(est, X_train,X_train.loc[:, sens_features], y_train, ['balanced_accuracy', 'fpr', 'subgroup_fnr_loss'])
                test_score = utils.evaluate_objective_functions(est, X_test, X_test.loc[:, sens_features],y_test, ['balanced_accuracy', 'fpr', 'subgroup_fnr_loss'])

                print("Ending the scoring process. ")

                this_score = {}
                train_score = {f"train_{k}": v for k, v in train_score.items()}
                this_score.update(train_score)
                this_score.update(test_score)

                this_score["pipeline"] = est.fitted_pipeline_
                this_score["exp_name"] = exp
                this_score["seed"] = super_seed
                this_score["run"] = r

                
                with open(f"{save_folder}/scores.pkl", "wb") as f:
                    pickle.dump(this_score, f)

                return           
                
            except Exception as e:
                    trace =  traceback.format_exc()
                    pipeline_failure_dict = {"exp_name": exp,  "error": str(e), "trace": trace}
                    print("failed on ")
                    print(save_folder)
                    print(e)
                    print(trace)

                    with open(f"{save_folder}/failed.pkl", "wb") as f:
                        pickle.dump(pipeline_failure_dict, f)

                    return
        
    print("all finished")

def test_code():
    X_train, X_test, y_train, y_test, X_train_groups, X_test_groups = utils.retrieve_data('data_rus_100p.pkl')
    print(type(y_test))
    rf = RandomForestClassifier(n_estimators=100, random_state=123)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:,1]
    y_pred = y_prob > 0.55
    sens_features = ['PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
                                 'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
                                 'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
                                 'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE']
                
    score = utils.evaluate_objective_functions(rf, X_test, X_test.loc[:, sens_features],y_test, ['balanced_accuracy', 'fpr', 'subgroup_fnr_loss'])
    print(score)
    print(X_test.head())
    print(y_test.head())



if __name__ == '__main__':
    # Define the experiments to run
    experiments = ['Regular Feature Selection', 'Fairness Aware Feature Selection']
    base_save_folder = 'results_local'
    data_dir = 'data'
    num_runs = 1
    ga_params_local = {'population_size': 8, 'generations': 4, 'n_jobs':4}
    ga_params_remote = {'population_size': 100, 'generations': 50, 'n_jobs':28}


    loop_through_tasks(experiments, 'data_rus_100p.pkl', base_save_folder, 5, ga_params_local)
    # loop_through_tasks(experiments, base_save_folder, 20, ga_params_remote)

