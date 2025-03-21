import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random

def process_data(orig_data_file, processed_data_file, seed=None, prop_to_process = 1.0):
    """
    Process the input data.
    """
    rng = np.random.default_rng(seed)


    chunksize = 10 ** 5
    chunk = pd.read_csv(orig_data_file,
                    chunksize=chunksize, on_bad_lines = 'skip')
    data = pd.concat(chunk) # get execution time

    data = data.drop('Unnamed: 0', axis=1)

    # Target variable
    target = 'EVENT_YN'
    features = [
               # General patient info 
               'AGE','SEX','PATIENT_RACES_S','N_RACE','ETHNICITY','ETHNIC_RACE_S',
               
               # Clinical info
               'DEPARTMENT_NAME','LOS',
               
               # Lab work
               #'LABLKV_BANDS', # missing 98%
               #'LABLKV_BANDSPCT', # missing 78%
               'LABLKV_GLUCOSE','LABLKV_PLATELET','LABLKV_SODIUM','LABLKV_BUN','LABLKV_HEMATOCRIT',
               'LABLKV_HEMOGLOBIN','LABLKV_POTASSIUM','LABLKV_WBC',
               #'GLUCOSE-POC', # missing 55%
               #'LABLKV_LACTATE', # missing 77%
    
               # Flowchart values 
               'FLOLKV_SOFA','FLOLKV_PULSE','FLOLKV_SPO2','FLOLKV_GLASGOW','FLOLKV_BRADEN','FLOLKV_RESP',
               'FLOLKV_BP_SYS','FLOLKV_BP_DIA','FLOLKV_TEMP',
               'FLOLKV_NEUROLOGICAL','FLOLKV_DEVICE','FLOLKV_PULSE_CHG','FLOLKV_GLASGOW_CHG',
               'BPALKV_HIGH_RISK','MEDLKV_HIGH_RISK'
               ]
    # Get dummy variables
    dummy_cols = ['SEX','PATIENT_RACES_S','ETHNICITY','ETHNIC_RACE_S','DEPARTMENT_NAME','FLOLKV_NEUROLOGICAL',
              'FLOLKV_DEVICE','FLOLKV_PULSE_CHG','FLOLKV_GLASGOW_CHG','BPALKV_HIGH_RISK','MEDLKV_HIGH_RISK']
    data = pd.get_dummies(data, columns=dummy_cols)
    data = data.drop(columns=['BPALKV_HIGH_RISK_No','MEDLKV_HIGH_RISK_No']) # Completely redundant with _Yes columns
    features = ['AGE','N_RACE','LOS',
            'LABLKV_GLUCOSE','LABLKV_PLATELET','LABLKV_SODIUM','LABLKV_BUN',
            'LABLKV_HEMATOCRIT','LABLKV_HEMOGLOBIN','LABLKV_POTASSIUM','LABLKV_WBC',
            'FLOLKV_SOFA','FLOLKV_PULSE','FLOLKV_SPO2','FLOLKV_GLASGOW','FLOLKV_BRADEN',
            'FLOLKV_RESP','FLOLKV_BP_SYS','FLOLKV_BP_DIA','FLOLKV_TEMP',
            'SEX_FEMALE','SEX_MALE','SEX_NON-BINARY','SEX_OTHER','SEX_UNKNOWN',
            'PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
            'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
            'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
            'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE',
            'ETHNICITY_HISPANIC','ETHNICITY_NON-HISPANIC','ETHNICITY_UNKNOWN',
            'ETHNIC_RACE_S_AMERICAN INDIAN OR ALASKA NATIVE','ETHNIC_RACE_S_ASIAN',
            'ETHNIC_RACE_S_BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_HISPANIC ASIAN',
            'ETHNIC_RACE_S_HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_HISPANIC OTHER',
            'ETHNIC_RACE_S_HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_HISPANIC WHITE',
            'ETHNIC_RACE_S_MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_NON-HISPANIC ASIAN',
            'ETHNIC_RACE_S_NON-HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_NON-HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_NON-HISPANIC OTHER',
            'ETHNIC_RACE_S_NON-HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_NON-HISPANIC WHITE',
            'ETHNIC_RACE_S_OTHER',
            'ETHNIC_RACE_S_UNKNOWN',
            'ETHNIC_RACE_S_WHITE',
            'DEPARTMENT_NAME_3-ACU (3-PACU)',
            'DEPARTMENT_NAME_3-LDR',
            'DEPARTMENT_NAME_3-N',
            'DEPARTMENT_NAME_3-N MFCU',
            'DEPARTMENT_NAME_3-SE',
            'DEPARTMENT_NAME_3-SE OB',
            'DEPARTMENT_NAME_3-SW',
            'DEPARTMENT_NAME_3-SW OB',
            'DEPARTMENT_NAME_3N-UNIV',
            'DEPARTMENT_NAME_3SPT',
            'DEPARTMENT_NAME_4-NE',
            'DEPARTMENT_NAME_4-NW',
            'DEPARTMENT_NAME_4-SE',
            'DEPARTMENT_NAME_4-SW',
            'DEPARTMENT_NAME_4S-MON',
            'DEPARTMENT_NAME_4S-PICU',
            'DEPARTMENT_NAME_5-ACU (5-PACU)',
            'DEPARTMENT_NAME_5-NE',
            'DEPARTMENT_NAME_5-NW',
            'DEPARTMENT_NAME_5-SE',
            'DEPARTMENT_NAME_5-SW',
            'DEPARTMENT_NAME_6-ACU (6-PACU)',
            'DEPARTMENT_NAME_6-ICU',
            'DEPARTMENT_NAME_6-NE',
            'DEPARTMENT_NAME_6-NW',
            'DEPARTMENT_NAME_6-SE',
            'DEPARTMENT_NAME_6-SW',
            'DEPARTMENT_NAME_7-ACU (7-PACU)',
            'DEPARTMENT_NAME_7-NE',
            'DEPARTMENT_NAME_7-NW',
            'DEPARTMENT_NAME_7-SE',
            'DEPARTMENT_NAME_7-SW',
            'DEPARTMENT_NAME_7GI-ACU (GI LAB PACU)',
            'DEPARTMENT_NAME_8-ACU (8-PACU)',
            'DEPARTMENT_NAME_8-NE',
            'DEPARTMENT_NAME_8-NW',
            'DEPARTMENT_NAME_8-SE',
            'DEPARTMENT_NAME_8-SW',
            'DEPARTMENT_NAME_AHSP 5-ACU',
            'DEPARTMENT_NAME_AHSP 5-OSU',
            'FLOLKV_NEUROLOGICAL_WDL',
            'FLOLKV_NEUROLOGICAL_X',
            'FLOLKV_DEVICE_O2',
            'FLOLKV_DEVICE_assisted',
            'FLOLKV_DEVICE_room air',
            'FLOLKV_PULSE_CHG_decline',
            'FLOLKV_PULSE_CHG_increase',
            'FLOLKV_PULSE_CHG_no change',
            'FLOLKV_GLASGOW_CHG_decline',
            'FLOLKV_GLASGOW_CHG_increase',
            'FLOLKV_GLASGOW_CHG_no change',
            'BPALKV_HIGH_RISK_Yes',
            'MEDLKV_HIGH_RISK_Yes']
    
    pat_encs = data.PAT_ENC_CSN_ID.unique().tolist()
    num_pat_encs = len(pat_encs)
    
    if prop_to_process < 1.0:
        # randomly sample a subset of the data
        pat_encs = rng.choice(pat_encs,int(num_pat_encs*prop_to_process), replace=False).tolist() 


    # Sample a subset of the data for training
    sample_ids = rng.choice(pat_encs,int(len(pat_encs)/2), replace=False).tolist()
    sample_df = data[data['PAT_ENC_CSN_ID'].isin(sample_ids)]

    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))
    print('Data includes hospital admissions between ',sample_df.HOSP_ADMSN_TIME.min(),' and ',sample_df.HOSP_ADMSN_TIME.max())
    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age = round(np.median(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')

    # Build test set
    hold_out_df = data[~data['PAT_ENC_CSN_ID'].isin(sample_ids)]
    print('Validation data includes hospital admissions between ',hold_out_df.HOSP_ADMSN_TIME.min(),' and ',hold_out_df.HOSP_ADMSN_TIME.max())
    print('Number of unique patient encounters: ',len(hold_out_df.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age= round(np.median(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(hold_out_df.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')

    group_column = 'PAT_ENC_CSN_ID'
    
    df = sample_df[[group_column]+[target]+features]
    df = df.dropna() # drop any instances with missing values

    X_train_groups = np.array(df[group_column])
    y = np.array(df[target])
    X = df.drop(columns=['PAT_ENC_CSN_ID',target])

    # test model on remainder of data (hold-out sample)
    hold_out_df = hold_out_df[[group_column]+[target]+features]
    hold_out_df = hold_out_df.dropna() # drop any instances with missing values

    X_test_groups = np.array(hold_out_df[group_column])
    y_test = hold_out_df[target]
    X_test = hold_out_df.drop(columns=['PAT_ENC_CSN_ID',target])

    # Save X, y, X_test, y_test as pickle files
    data= {'X_train': X, 'y_train': y, 'X_test': X_test, 'y_test': y_test, 'X_train_groups': X_train_groups, 'X_test_groups': X_test_groups}
    with open(processed_data_file, 'wb') as f:
        pickle.dump(data, f)

def stratified_grouped_train_test_split(df, group_by, stratify_cols, test_size, random_state=None):
    # Drop duplicates to avoid repeated measures
    unique_df = df.drop_duplicates(group_by)

    # Create a stratify column that combines the stratify_cols
    unique_df['stratify_col'] = unique_df[stratify_cols].astype(str).agg('-'.join, axis=1)
    print(unique_df['stratify_col'])
    print(unique_df['stratify_col'].value_counts())

    # Extract all samples of the the class where counts==1
    unique_counts = unique_df['stratify_col'].value_counts()
    value_with_one_sample = unique_counts[unique_counts == 1].index.tolist()
    rare_samples = unique_df[unique_df['stratify_col'].isin(value_with_one_sample)]
    remaining_df = unique_df[~unique_df['stratify_col'].isin(value_with_one_sample)]
    print("Number of rare samples: ", len(rare_samples))

    print(remaining_df['stratify_col'])
    print(remaining_df['stratify_col'].value_counts())

    # Split the unique persons into train and test sets
    train_ids, test_ids = train_test_split(
        remaining_df[group_by], test_size=test_size, random_state=random_state,
        stratify=remaining_df['stratify_col']
    )

    # Create the train and test sets by filtering the original dataframe
    rare_samples = rare_samples.drop(columns=['stratify_col'])
    train_df = df[df[group_by].isin(train_ids)]
    train_df = pd.concat([train_df, rare_samples])
    train_df = train_df.reset_index(drop=True)

    test_df = df[df[group_by].isin(test_ids)]
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def GroupedRandomUnderSample(data, target, group_by, random_state):
    pos_ids = data[group_by][data[target]==1].unique().tolist()
    neg_ids = data[group_by][data[target]==0].unique().tolist()
    
    random.seed(random_state)
    n = len(pos_ids)
    sample_neg_ids = random.sample(neg_ids,n)
    sample_ids = pos_ids + sample_neg_ids

    rus_df = data[data[group_by].isin(sample_ids)].reset_index(drop=True)
    return rus_df

def process_data_rus(orig_data_file, processed_data_file, seed=None, prop_to_process = 1.0):
    """
    Process the input data. Employ random undersampling to balance the dataset.
    """
    rng = np.random.default_rng(seed)

    chunksize = 10 ** 5
    chunk = pd.read_csv(orig_data_file,
                    chunksize=chunksize, on_bad_lines = 'skip')
    data = pd.concat(chunk) # get execution time

    data = data.drop('Unnamed: 0', axis=1)

    # Target variable
    target = 'EVENT_YN'
    group_column = 'PAT_ENC_CSN_ID'
    original_features = [
               # General patient info 
               'AGE','SEX','PATIENT_RACES_S','N_RACE','ETHNICITY','ETHNIC_RACE_S',
               
               # Clinical info
               'DEPARTMENT_NAME','LOS',
               
               # Lab work
               #'LABLKV_BANDS', # missing 98%
               #'LABLKV_BANDSPCT', # missing 78%
               'LABLKV_GLUCOSE','LABLKV_PLATELET','LABLKV_SODIUM','LABLKV_BUN','LABLKV_HEMATOCRIT',
               'LABLKV_HEMOGLOBIN','LABLKV_POTASSIUM','LABLKV_WBC',
               #'GLUCOSE-POC', # missing 55%
               #'LABLKV_LACTATE', # missing 77%
    
               # Flowchart values 
               'FLOLKV_SOFA','FLOLKV_PULSE','FLOLKV_SPO2','FLOLKV_GLASGOW','FLOLKV_BRADEN','FLOLKV_RESP',
               'FLOLKV_BP_SYS','FLOLKV_BP_DIA','FLOLKV_TEMP',
               'FLOLKV_NEUROLOGICAL','FLOLKV_DEVICE','FLOLKV_PULSE_CHG','FLOLKV_GLASGOW_CHG',
               'BPALKV_HIGH_RISK','MEDLKV_HIGH_RISK'
               ]
    # Get dummy variables
    dummy_cols = ['SEX','PATIENT_RACES_S','ETHNICITY','ETHNIC_RACE_S','DEPARTMENT_NAME','FLOLKV_NEUROLOGICAL',
              'FLOLKV_DEVICE','FLOLKV_PULSE_CHG','FLOLKV_GLASGOW_CHG','BPALKV_HIGH_RISK','MEDLKV_HIGH_RISK']
    data = pd.get_dummies(data, columns=dummy_cols)
    data = data.drop(columns=['BPALKV_HIGH_RISK_No','MEDLKV_HIGH_RISK_No']) # Completely redundant with _Yes columns
    features = ['AGE','N_RACE','LOS',
            'LABLKV_GLUCOSE','LABLKV_PLATELET','LABLKV_SODIUM','LABLKV_BUN',
            'LABLKV_HEMATOCRIT','LABLKV_HEMOGLOBIN','LABLKV_POTASSIUM','LABLKV_WBC',
            'FLOLKV_SOFA','FLOLKV_PULSE','FLOLKV_SPO2','FLOLKV_GLASGOW','FLOLKV_BRADEN',
            'FLOLKV_RESP','FLOLKV_BP_SYS','FLOLKV_BP_DIA','FLOLKV_TEMP',
            'SEX_FEMALE','SEX_MALE','SEX_NON-BINARY','SEX_OTHER','SEX_UNKNOWN',
            'PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
            'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
            'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
            'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE',
            'ETHNICITY_HISPANIC','ETHNICITY_NON-HISPANIC','ETHNICITY_UNKNOWN',
            'ETHNIC_RACE_S_AMERICAN INDIAN OR ALASKA NATIVE','ETHNIC_RACE_S_ASIAN',
            'ETHNIC_RACE_S_BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_HISPANIC ASIAN',
            'ETHNIC_RACE_S_HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_HISPANIC OTHER',
            'ETHNIC_RACE_S_HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_HISPANIC WHITE',
            'ETHNIC_RACE_S_MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_NON-HISPANIC ASIAN',
            'ETHNIC_RACE_S_NON-HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_NON-HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_NON-HISPANIC OTHER',
            'ETHNIC_RACE_S_NON-HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_NON-HISPANIC WHITE',
            'ETHNIC_RACE_S_OTHER',
            'ETHNIC_RACE_S_UNKNOWN',
            'ETHNIC_RACE_S_WHITE',
            'DEPARTMENT_NAME_3-ACU (3-PACU)',
            'DEPARTMENT_NAME_3-LDR',
            'DEPARTMENT_NAME_3-N',
            'DEPARTMENT_NAME_3-N MFCU',
            'DEPARTMENT_NAME_3-SE',
            'DEPARTMENT_NAME_3-SE OB',
            'DEPARTMENT_NAME_3-SW',
            'DEPARTMENT_NAME_3-SW OB',
            'DEPARTMENT_NAME_3N-UNIV',
            'DEPARTMENT_NAME_3SPT',
            'DEPARTMENT_NAME_4-NE',
            'DEPARTMENT_NAME_4-NW',
            'DEPARTMENT_NAME_4-SE',
            'DEPARTMENT_NAME_4-SW',
            'DEPARTMENT_NAME_4S-MON',
            'DEPARTMENT_NAME_4S-PICU',
            'DEPARTMENT_NAME_5-ACU (5-PACU)',
            'DEPARTMENT_NAME_5-NE',
            'DEPARTMENT_NAME_5-NW',
            'DEPARTMENT_NAME_5-SE',
            'DEPARTMENT_NAME_5-SW',
            'DEPARTMENT_NAME_6-ACU (6-PACU)',
            'DEPARTMENT_NAME_6-ICU',
            'DEPARTMENT_NAME_6-NE',
            'DEPARTMENT_NAME_6-NW',
            'DEPARTMENT_NAME_6-SE',
            'DEPARTMENT_NAME_6-SW',
            'DEPARTMENT_NAME_7-ACU (7-PACU)',
            'DEPARTMENT_NAME_7-NE',
            'DEPARTMENT_NAME_7-NW',
            'DEPARTMENT_NAME_7-SE',
            'DEPARTMENT_NAME_7-SW',
            'DEPARTMENT_NAME_7GI-ACU (GI LAB PACU)',
            'DEPARTMENT_NAME_8-ACU (8-PACU)',
            'DEPARTMENT_NAME_8-NE',
            'DEPARTMENT_NAME_8-NW',
            'DEPARTMENT_NAME_8-SE',
            'DEPARTMENT_NAME_8-SW',
            'DEPARTMENT_NAME_AHSP 5-ACU',
            'DEPARTMENT_NAME_AHSP 5-OSU',
            'FLOLKV_NEUROLOGICAL_WDL',
            'FLOLKV_NEUROLOGICAL_X',
            'FLOLKV_DEVICE_O2',
            'FLOLKV_DEVICE_assisted',
            'FLOLKV_DEVICE_room air',
            'FLOLKV_PULSE_CHG_decline',
            'FLOLKV_PULSE_CHG_increase',
            'FLOLKV_PULSE_CHG_no change',
            'FLOLKV_GLASGOW_CHG_decline',
            'FLOLKV_GLASGOW_CHG_increase',
            'FLOLKV_GLASGOW_CHG_no change',
            'BPALKV_HIGH_RISK_Yes',
            'MEDLKV_HIGH_RISK_Yes']
    
    pat_encs = data.PAT_ENC_CSN_ID.unique().tolist()
    num_pat_encs = len(pat_encs)
    
    if prop_to_process < 1.0:
        # randomly sample a subset of the data
        pat_encs = rng.choice(pat_encs,int(num_pat_encs*prop_to_process), replace=False).tolist() 

    # Sample a subset of the data for training
    sample_ids = rng.choice(pat_encs,int(len(pat_encs)/2), replace=False).tolist()
    sample_df = data[data['PAT_ENC_CSN_ID'].isin(sample_ids)]

    print("Training data characteristics before Random Undersampling:")
    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))
    print('Data includes hospital admissions between ',sample_df.HOSP_ADMSN_TIME.min(),' and ',sample_df.HOSP_ADMSN_TIME.max())
    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age = round(np.median(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')

    sensitive_features = ['PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
                                 'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
                                 'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
                                 'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE']
    train_split, val_split = stratified_grouped_train_test_split(df=sample_df, group_by=group_column, stratify_cols=[target]+sensitive_features, test_size=0.20, random_state=0)
    train_df_rus = GroupedRandomUnderSample(data=train_split, target=target, group_by=group_column, random_state=0)

    X_train = train_df_rus[features]
    X_train_groups = np.array(train_df_rus[group_column])
    y_train = np.array(train_df_rus[target])

    # validation split (use this for finetuning)                     
    X_val = val_split[features]
    y_val = val_split[target]

    print("Training data characteristics after Random Undersampling:")
    print('Number of unique patient encounters: ',len(train_df_rus.PAT_ENC_CSN_ID.unique()))
    print('Data includes hospital admissions between ',train_df_rus.HOSP_ADMSN_TIME.min(),' and ',train_df_rus.HOSP_ADMSN_TIME.max())
    print('Number of unique patient encounters: ',len(train_df_rus.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(train_df_rus.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(train_df_rus.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age = round(np.median(train_df_rus.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(train_df_rus.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')

    # Build test set
    hold_out_df = data[~data['PAT_ENC_CSN_ID'].isin(sample_ids)]
    print('Test data includes hospital admissions between ',hold_out_df.HOSP_ADMSN_TIME.min(),' and ',hold_out_df.HOSP_ADMSN_TIME.max())
    print('Number of unique patient encounters: ',len(hold_out_df.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age= round(np.median(hold_out_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(hold_out_df.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')

    # test model on remainder of data (hold-out sample)
    hold_out_df = hold_out_df[[group_column]+[target]+features]
    hold_out_df = hold_out_df.dropna() # drop any instances with missing values

    X_test_groups = np.array(hold_out_df[group_column])
    y_test = hold_out_df[target]
    X_test = hold_out_df.drop(columns=['PAT_ENC_CSN_ID',target])


    # Save X, y, X_test, y_test as pickle files
    data= {'X_train': X_train, 'y_train': y_train, 
           'X_test': X_test, 'y_test': y_test,
           'X_val': X_val, 'y_val': y_val,
           'X_train_groups': X_train_groups, 'X_test_groups': X_test_groups}
    with open(processed_data_file, 'wb') as f:
        pickle.dump(data, f)



def preprocess_data(orig_data_file, processed_data_file, seed=None):
    """
    Process the input data. Employ random undersampling to balance the dataset.
    """
    rng = np.random.default_rng(seed)

    chunksize = 10 ** 5
    chunk = pd.read_csv(orig_data_file,
                    chunksize=chunksize, on_bad_lines = 'skip')
    data = pd.concat(chunk) # get execution time

    print("Original data features: ", list(data.columns))

    data = data.drop('Unnamed: 0', axis=1)

    # Target variable
    target = 'EVENT_YN'
    group_column = 'PAT_ENC_CSN_ID'
  
    # Get dummy variables
    dummy_cols = ['SEX','PATIENT_RACES_S','ETHNICITY','ETHNIC_RACE_S','DEPARTMENT_NAME','FLOLKV_NEUROLOGICAL',
              'FLOLKV_DEVICE','FLOLKV_PULSE_CHG','FLOLKV_GLASGOW_CHG','BPALKV_HIGH_RISK','MEDLKV_HIGH_RISK']
    data = pd.get_dummies(data, columns=dummy_cols)
    features = ['AGE','N_RACE','LOS',
            'LABLKV_GLUCOSE','LABLKV_PLATELET','LABLKV_SODIUM','LABLKV_BUN',
            'LABLKV_HEMATOCRIT','LABLKV_HEMOGLOBIN','LABLKV_POTASSIUM','LABLKV_WBC',
            'FLOLKV_SOFA','FLOLKV_PULSE','FLOLKV_SPO2','FLOLKV_GLASGOW','FLOLKV_BRADEN',
            'FLOLKV_RESP','FLOLKV_BP_SYS','FLOLKV_BP_DIA','FLOLKV_TEMP',
            'SEX_FEMALE','SEX_MALE','SEX_NON-BINARY','SEX_OTHER','SEX_UNKNOWN',
            'PATIENT_RACES_S_AMERICAN INDIAN OR ALASKA NATIVE','PATIENT_RACES_S_ASIAN',
            'PATIENT_RACES_S_BLACK OR AFRICAN AMERICAN','PATIENT_RACES_S_MULTIRACIAL',
            'PATIENT_RACES_S_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','PATIENT_RACES_S_OTHER',
            'PATIENT_RACES_S_UNKNOWN','PATIENT_RACES_S_WHITE',
            'ETHNICITY_HISPANIC','ETHNICITY_NON-HISPANIC','ETHNICITY_UNKNOWN',
            'ETHNIC_RACE_S_AMERICAN INDIAN OR ALASKA NATIVE','ETHNIC_RACE_S_ASIAN',
            'ETHNIC_RACE_S_BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_HISPANIC ASIAN',
            'ETHNIC_RACE_S_HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_HISPANIC OTHER',
            'ETHNIC_RACE_S_HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_HISPANIC WHITE',
            'ETHNIC_RACE_S_MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC AMERICAN INDIAN OR ALASKA NATIVE',
            'ETHNIC_RACE_S_NON-HISPANIC ASIAN',
            'ETHNIC_RACE_S_NON-HISPANIC BLACK OR AFRICAN AMERICAN',
            'ETHNIC_RACE_S_NON-HISPANIC MULTIRACIAL',
            'ETHNIC_RACE_S_NON-HISPANIC NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'ETHNIC_RACE_S_NON-HISPANIC OTHER',
            'ETHNIC_RACE_S_NON-HISPANIC UNKNOWN',
            'ETHNIC_RACE_S_NON-HISPANIC WHITE',
            'ETHNIC_RACE_S_OTHER',
            'ETHNIC_RACE_S_UNKNOWN',
            'ETHNIC_RACE_S_WHITE',
            'DEPARTMENT_NAME_3-ACU (3-PACU)',
            'DEPARTMENT_NAME_3-LDR',
            'DEPARTMENT_NAME_3-N',
            'DEPARTMENT_NAME_3-N MFCU',
            'DEPARTMENT_NAME_3-SE',
            'DEPARTMENT_NAME_3-SE OB',
            'DEPARTMENT_NAME_3-SW',
            'DEPARTMENT_NAME_3-SW OB',
            'DEPARTMENT_NAME_3N-UNIV',
            'DEPARTMENT_NAME_3SPT',
            'DEPARTMENT_NAME_4-NE',
            'DEPARTMENT_NAME_4-NW',
            'DEPARTMENT_NAME_4-SE',
            'DEPARTMENT_NAME_4-SW',
            'DEPARTMENT_NAME_4S-MON',
            'DEPARTMENT_NAME_4S-PICU',
            'DEPARTMENT_NAME_5-ACU (5-PACU)',
            'DEPARTMENT_NAME_5-NE',
            'DEPARTMENT_NAME_5-NW',
            'DEPARTMENT_NAME_5-SE',
            'DEPARTMENT_NAME_5-SW',
            'DEPARTMENT_NAME_6-ACU (6-PACU)',
            'DEPARTMENT_NAME_6-ICU',
            'DEPARTMENT_NAME_6-NE',
            'DEPARTMENT_NAME_6-NW',
            'DEPARTMENT_NAME_6-SE',
            'DEPARTMENT_NAME_6-SW',
            'DEPARTMENT_NAME_7-ACU (7-PACU)',
            'DEPARTMENT_NAME_7-NE',
            'DEPARTMENT_NAME_7-NW',
            'DEPARTMENT_NAME_7-SE',
            'DEPARTMENT_NAME_7-SW',
            'DEPARTMENT_NAME_7GI-ACU (GI LAB PACU)',
            'DEPARTMENT_NAME_8-ACU (8-PACU)',
            'DEPARTMENT_NAME_8-NE',
            'DEPARTMENT_NAME_8-NW',
            'DEPARTMENT_NAME_8-SE',
            'DEPARTMENT_NAME_8-SW',
            'DEPARTMENT_NAME_AHSP 5-ACU',
            'DEPARTMENT_NAME_AHSP 5-OSU',
            'FLOLKV_NEUROLOGICAL_WDL',
            'FLOLKV_NEUROLOGICAL_X',
            'FLOLKV_DEVICE_O2',
            'FLOLKV_DEVICE_assisted',
            'FLOLKV_DEVICE_room air',
            'FLOLKV_PULSE_CHG_decline',
            'FLOLKV_PULSE_CHG_increase',
            'FLOLKV_PULSE_CHG_no change',
            'FLOLKV_GLASGOW_CHG_decline',
            'FLOLKV_GLASGOW_CHG_increase',
            'FLOLKV_GLASGOW_CHG_no change',
            'BPALKV_HIGH_RISK_Yes',
            'MEDLKV_HIGH_RISK_Yes']
    
    pat_encs = data.PAT_ENC_CSN_ID.unique().tolist()

    sample_df = data[data['PAT_ENC_CSN_ID'].isin(pat_encs)]

    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))
    print('Number of unique patient encounters: ',len(sample_df.PAT_ENC_CSN_ID.unique()))

    avg_age = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    std_age = round(np.std(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Average age of patients was: ',avg_age,' with a standard deviation of ',std_age)
    med_age = round(np.median(sample_df.groupby('PAT_ENC_CSN_ID')['AGE'].max()),2)
    print('Median age of patients was: ',med_age)
    adv_events = round(np.mean(sample_df.groupby('PAT_ENC_CSN_ID')['EVENT_YN'].max()),2)*100
    print('Adverse events occurred in approximately ',adv_events,'% of encounters')
    
    print("Features: ", list(sample_df.columns))
    X_train = sample_df.loc[:,features]
    y_train = np.array(sample_df[target])

    # Save X, y, X_test, y_test as pickle files
    data= {'X_train': X_train, 'y_train': y_train}
    with open(processed_data_file, 'wb') as f:
        pickle.dump(data, f)




if __name__ == "__main__":
    process_data_rus('DTI_Data.csv', 'data_rus_100p.pkl', seed=123, prop_to_process = 1.0)