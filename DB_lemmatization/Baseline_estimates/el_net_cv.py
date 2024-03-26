#!/usr/bin/env python

#SBATCH --partition=g100_usr_interactive
#SBATCH --time=04:00:00            
#SBATCH --job-name=elnet_cv              
#SBATCH --output=elnet_cv_%j.txt


#### You make first 5 splits into test and train, after on train set with 5 fold CV you find the best parameters and take test score results. 
#### After take average of 5 test scores

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import  KFold
from joblib import dump
import pickle
import matplotlib.pyplot as plt

tfidf = pd.read_csv('/g100_work/IscrC_mental/data/tfidf/results_concatenate/tfidf_concatenate.csv')

with open('/g100_work/IscrC_mental/data/tfidf/ridge/elnet_cv_embed/filtered_n_grams.pkl', 'rb') as file:
    filtered = pickle.load(file)
filtered = list(filtered)
filtered.append('group_id')

tfidf = tfidf[filtered]

print(f'{len(tfidf)} df length', flush=True)


file_path = '/g100_work/IscrC_mental/data/survey/survey_labels/survey_labels.parquet'
predictions = pd.read_parquet(file_path, engine='pyarrow')

filtered_predictions = predictions[~(
    (predictions['group_id'].astype(str).str[0:2] == '99') |
    (predictions['group_id'].astype(str).str[2:4] == '99') |
    (predictions['group_id'].astype(str).str[4:6] == '99')|
    (predictions['group_id'].astype(str).str[6:8] == '99')
)]

labels_pivoted = filtered_predictions.pivot(index='group_id', columns='question', values='label')
labels_pivoted.reset_index(drop=False, inplace=True)
survey = labels_pivoted

questions = {}
for i in range(len(survey.columns[1:])):
    questions[i] = survey.columns[1:][i]

print(len(questions))
output_results = pd.DataFrame(columns =['question', 'l2_cv', 'mse_cv'])
output_results['question'] = questions.values()
column_names_to_itterate = ['elnet_r2','elnet_mse', 'elnet_alpha', 'elnet_l1' ]
for i in range(5):
    for col in column_names_to_itterate:
        output_results[f'{col}_{i}']=np.nan

# Define the number of folds for cross-validation
num_folds = 5

# Initialize KFold object
kf = KFold(n_splits=num_folds)

print('all downloaded', flush = True)

colors = ['blue', 'orange', 'green', 'red', 'purple']

def process_question(question, tfidf, survey, kf):
    df = tfidf.merge(survey.loc[:, [question, 'group_id']], on='group_id', how ='inner')
    df = df[df[question]>0]
    X = df.drop([question, 'group_id'], axis = 1)
    y = df.loc[:, [question]]
    
    # Initialize lists to store test scores and MSE for each fold
    elnet_test_scores = []
    elnet_mse_scores = []

    # Perform k-fold cross-validation
    index_for_output_table = 0
    plt.figure()
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]  

        elastic_net = ElasticNet()
        param_grid = {'alpha': uniform(loc=0, scale=20), 'l1_ratio': l1_ratio_values}
        random_search = RandomizedSearchCV(elastic_net, param_distributions=param_grid, n_iter=100, cv=5)
        random_search.fit(X_train_scaled, y_train)

        best_alpha = random_search.best_params_['alpha']
        best_l1_ratio = random_search.best_params_['l1_ratio']

        elastic_net_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
        elastic_net_model.fit(X_train_scaled, y_train) 
        test_score = elastic_net_model.score(X_test_scaled, y_test)
        elnet_test_scores.append(test_score)   
        y_pred = elastic_net_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        elnet_mse_scores.append(mse)

        dump(elastic_net_model, f'/g100_work/IscrC_mental/data/tfidf/ridge/elnet_cv_embed_plots/rlen_concatenate_{question}_{index_for_output_table}.joblib')
        
        output_results.loc[output_results['question'] == question, f'elnet_r2_{index_for_output_table}'] = test_score
        output_results.loc[output_results['question'] == question, f'elnet_mse_{index_for_output_table}'] = mse
        output_results.loc[output_results['question'] == question, f'elnet_alpha_{index_for_output_table}'] = best_alpha
        output_results.loc[output_results['question'] == question, f'elnet_l1_{index_for_output_table}'] = best_l1_ratio
        
      
        plt.scatter(y_test, y_pred, color=colors[fold_index], label=f'test fold {fold_index+1}') 
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"{question}_{index_for_output_table}")
        plt.legend()

        plt.savefig(f"/g100_work/IscrC_mental/data/tfidf/ridge/elnet_cv_embed_plots/scatter_plot_{question}.png")   

        plt.show() 
        index_for_output_table += 1
    
    avg_test_score = np.mean(elnet_test_scores)
    avg_mse = np.mean(elnet_mse_scores)

    output_results.loc[output_results['question'] == question, 'l2_cv'] = avg_test_score
    output_results.loc[output_results['question'] == question, 'mse_cv'] = avg_mse

    output_line = output_results[output_results['question'] == question]
    output_line.to_csv(f'/g100_work/IscrC_mental/data/tfidf/ridge/elnet_cv_embed_plots/output_{question}_elnet.csv', index=False)

    print(f'elnet results for {question}')



for i in tqdm(range(len(survey.columns[1:]))):
    question = questions[i]
    process_question(question, tfidf, survey, kf)
