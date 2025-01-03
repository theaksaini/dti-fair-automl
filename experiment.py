import tpot2
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
from tpot2.search_spaces.nodes import *
from tpot2.search_spaces.pipelines import *

def process_data():
    # Load data as pandas dataframe
    data = pd.read_csv('DTI_dummy_Data_11_5_24.csv')
    
    # Drop empty columns
    data = data.dropna(axis=1, how='all')

    # Drop entries with missing values
    data = data.dropna()

    # Drop columns named 'pat_enc_csn_id' and 'actual_class'
    data = data.drop(columns=['pat_enc_csn_id', 'actual_class'])

    # Set 'actual_yn' as y and the rest as X
    y = data['actual_yn']
    X = data.drop(columns=['actual_yn'])

    #Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# pipeline search space: selector(optional) -> transformer(optional) -> regressor/classifier(required)
def get_pipeline_space(n_features, seed):
    return tpot2.search_spaces.pipelines.SequentialPipeline([
        GeneticFeatureSelectorNode(n_features=n_features),
        tpot2.config.get_search_space(["transformers","Passthrough"], n_classes=2, random_state=seed),
        tpot2.config.get_search_space('RandomForestClassifier', n_classes=2, random_state=seed)])


def execute_experiment():
    # Load data
    X_train, X_test, y_train, y_test = process_data()
    scorer = sklearn.metrics.get_scorer('roc_auc_ovo')
    random_state = 123

    # Define the pipeline
    est = tpot2.TPOTEstimator(search_space = get_pipeline_space(n_features=X_train.shape[1], seed=random_state),
                              scorers=["roc_auc_ovr", tpot2.objectives.complexity_scorer],
                              scorers_weights=[1.0, -1.0],
                              generations=5, population_size=20, cv=5, random_state=random_state, verbose=2,
                              classification=True)

    # Execute the pipeline
    est.fit(X_train, y_train)


    print(scorer(est, X_test, y_test)) 




if __name__ == '__main__':
    execute_experiment()