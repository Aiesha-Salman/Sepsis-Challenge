#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_curve
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')


    # ================================
    # Feature selection (column dropping)
    # ================================
    # For example, the team may select a subset of variables. Put your feature selection code if needed. Here, we simply use all raw columns.

    selected_variables = ['agecalc_adm','height_cm_adm','hr_bpm_adm','glucose_mmolpl_adm','diasbp_mmhg_adm',
    'weight_kg_adm','bcseye_adm','sysbp_mmhg_adm','rr_brpm_app_adm','lactate_mmolpl_adm',
    'hematocrit_gpdl_adm','bcsverbal_adm','temp_c_adm','spo2site2_pc_oxi_adm',
    'exclbreastfed_adm','feedingstatus_adm','watersource_adm','deadchildren_adm',
    'spo2site1_pc_oxi_adm','deliveryloc_adm','priorweekantimal_adm','malariastatuspos_adm',
    'vaccdpt_adm','waterpure_adm','sex_adm','sqi2_perc_oxi_adm','respdistress_adm','muac_mm_adm',
    'momage_adm','birthdetail_adm___5','birthdetail_adm___4','vaccpneumoc_adm','birthdetail_adm___1',
    'birthdetail_adm___2','birthdetail_adm___3','sqi1_perc_oxi_adm','oxygenavail_adm',
    'symptoms_adm___3','priorweekabx_adm','bcsmotor_adm','bcgscar_adm','symptoms_adm___9']

    print(selected_variables)

    data = data[selected_variables]

    # Save the selected features to file.
    with open(os.path.join(model_folder, 'selected_variables.txt'), 'w') as f:
        f.write("\n".join(selected_variables))

    # ================================
    # Preprocessing: dummy encoding     
    # # ================================
    data = pd.get_dummies(data)
    dummy_columns = list(data.columns)
    #Saving the dummy-encoded column names for later alignment during inference.
    with open(os.path.join(model_folder, 'dummy_columns.txt'), 'w') as f:
        f.write("\n".join(dummy_columns))
        
    # Split off a validation fold for threshold optimization
    X_train_d, X_val_d, y_train, y_val = train_test_split(
        data, label, test_size=0.2, stratify=label, random_state=42
    )
    if verbose >= 2:
        print(f"Train/val split: {len(y_train)} train, {len(y_val)} val.")

    # Fit imputer on training fold
    imputer = SimpleImputer().fit(X_train_d)
    X_train_imp = imputer.transform(X_train_d)
    X_val_imp = imputer.transform(X_val_d)

# Step 1: Show original class distribution
    if verbose >= 1:
        print('Original class distribution:')
        print(pd.Series(y_train.ravel()).value_counts())

# Step 2: Apply Borderline-SMOTE with sampling_strategy=0.3
    bl_smote = BorderlineSMOTE(sampling_strategy=0.35, random_state=789)
    X_resampled, y_resampled = bl_smote.fit_resample(X_train_imp, y_train.ravel())

# Step 3: Show new class distribution
    if verbose >= 1:
        print('After Borderline-SMOTE class distribution (25% minority):')
        print(pd.Series(y_resampled).value_counts())

    # Train classifier on training fold
    
    neg = np.sum(y_resampled == 0)
    pos = np.sum(y_resampled == 1)
    scale_pos_weight = neg / pos

    et = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
    )
    calibrated_et = CalibratedClassifierCV(et, method='isotonic', cv=3)
    calibrated_et.fit(X_resampled, y_resampled)

# Validation predictions and threshold optimization
    val_probs = calibrated_et.predict_proba(X_val_imp)[:, 1]
    threshold = find_threshold_for_sensitivity(y_val, val_probs, min_sens=0.8)
    with open(os.path.join(model_folder, 'threshold.txt'), 'w') as f:
        f.write(str(threshold))
    if verbose >= 1:
        print(f"Optimized threshold (min_sens=0.8): {threshold:.8f}")

# Retrain imputer & model on full data
    imputer_full = SimpleImputer().fit(data)
    X_full_imp = imputer_full.transform(data)

# Retrain final Extra Trees on all data (no need to calibrate here unless you want to)
    prediction_model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ).fit(X_full_imp, label.ravel())


    #prediction_model = RandomForestClassifier(
    #    n_estimators=123,
     #   max_leaf_nodes=456,
      #  random_state=789
   # ).fit(X_full_imp, label.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, prediction_model, selected_variables, dummy_columns, threshold)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')
    # Attempt to load the selected features from 'selected_variables.txt'
    try:
        with open(os.path.join(model_folder, 'selected_variables.txt'), 'r') as f:
            selected_variables = f.read().splitlines()
        if verbose:
            print("Loaded selected features from 'selected_variables.txt'")
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'selected_variables.txt'. Using all features. Error:", e)
        selected_variables = None

    # Load the dummy-encoded columns (used during training)
    try:
        with open(os.path.join(model_folder, 'dummy_columns.txt'), 'r') as f:
            dummy_columns = f.read().splitlines()
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'dummy_columns.txt'.", e)
        dummy_columns = None

    # Load the saved model.
    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    # Save the selected features into the model dictionary under a standardized key.
    model['selected_variables'] = selected_variables
    model['columns'] = dummy_columns
    return model

def find_threshold_for_sensitivity(y, p, thr=None, min_sens=0.8):
    if thr is not None:
        return thr
    fpr,tpr,ths=roc_curve(y,p)
    valid=ths[tpr>=min_sens]
    return float(valid.max()) if len(valid)>0 else 0.5

def run_challenge_model(model, data_folder, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']
    dummy_columns = model['dummy_columns']
    selected_variable = model['selected_variables']
    threshold = model.get('threshold', 0.5)

    
    # Load test data. If selected_variables is None, all columns are loaded.
    patient_ids, data, _ = load_challenge_testdata(data_folder, selected_variable)
    
    # Preprocess: apply dummy encoding and align with training dummy columns.
    data = pd.get_dummies(data)
    data = data.reindex(columns=dummy_columns, fill_value=0)
    
    # Impute missing data.
    data_imputed = imputer.transform(data)
    
    # Get prediction probabilities.
    prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]
    
    # want to use any other threshold, you may provide it here
    # threshold = 0.5

    # Compute binary predictions using the threshold.
    prediction_binary = (prediction_probability >= threshold).astype(int)
    
    # Write the threshold to a file called "threshold.txt" ignore if already wrote it during training.
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    
    return patient_ids, prediction_binary, prediction_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model along with the imputer, selected features, and dummy columns.
def save_challenge_model(model_folder, imputer, prediction_model, selected_variables, dummy_columns, threshold):
    d = {
        'imputer': imputer,
        'prediction_model': prediction_model,
        'selected_variables': selected_variables,
        'dummy_columns': dummy_columns,
        'threshold': threshold
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
