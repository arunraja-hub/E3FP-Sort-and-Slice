# %% [markdown]
# # Import Packages

# %%
# import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# import custom code written for this project
from modules import *
# from sort_and_slice_ecfp_featuriser import *

# import RDKit
from rdkit import Chem

# import and check funtionality of pytorch
import torch
print("Pytorch version = ", torch.__version__)
print("CUDA version = ", torch.version.cuda)
print("CUDA available = ", torch.cuda.is_available())
print("Random Pytorch test tensor = ", torch.rand(1))

# %%
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd

# import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# import custom code written for this project
from modules import *

# import RDKit
from rdkit import Chem
from rdkit.Chem import AllChem


# import and check funtionality of pytorch
import torch


from e3fp.pipeline import fprints_from_mol, fprints_from_smiles
from e3fp.fingerprint.generate import fprints_dict_from_sdf
from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint

from e3fp.fingerprint.fprinter import Fingerprinter

from collections import Counter

# %% [markdown]
# # Load and Prepare Data

# %%
# initialise dictionary used to store experimental settings, after dictionary is initialised, choose one of the five data sets below to proceed

settings_dict = {}

# %%
# LIT-PCBA Estrogen Receptor Alpha Antagonism

# settings_dict["dataset_name"] = "lit_pcba_esr_ant"
# settings_dict["task_type"] = "classification"
# settings_dict["prop_name"] = "Activity"

# %%
# PostEra SARS-CoV-2 Mpro inhibition

settings_dict["dataset_name"] = "postera_sars_cov_2_mpro"
settings_dict["task_type"] = "regression"
settings_dict["prop_name"] = "f_avg_IC50"

# %%
# Ames mutagenicity

# settings_dict["dataset_name"] = "ames_mutagenicity"
# settings_dict["task_type"] = "classification"
# settings_dict["prop_name"] = "Activity"

# # %%
# # AqSolDB solubility

# settings_dict["dataset_name"] = "aqsoldb_solubility"
# settings_dict["task_type"] = "regression"
# settings_dict["prop_name"] = "Solubility"

# # %%
# # MoleculeNet Lipophilicity

# settings_dict["dataset_name"] = "moleculenet_lipophilicity"
# settings_dict["task_type"] = "regression"
# settings_dict["prop_name"] = "exp"

# %%
# load clean data

dataframe = pd.read_csv("data/" + settings_dict["dataset_name"] + "/" + "clean_data.csv", sep = ",")
# display(dataframe)

# %%
# construct target variable y (for the SARS-CoV-2 main protease data set, we convert Ki to pKi by taking negative decadic logarithm)

if settings_dict["dataset_name"] == "postera_sars_cov_2_mpro":
    y = -np.log10(dataframe.loc[:, settings_dict["prop_name"]].values.astype(float))
else:
    y = dataframe.loc[:, settings_dict["prop_name"]].values.astype(float)

print("Mean Value (Target) = ", np.mean(y))
print("Standard Deviation (Target) = ", np.std(y))
print("Maximum Value (Target) = ", np.amax(y))
print("Minimum Value (Target) = ", np.amin(y), "\n")

print("Shape of y = ", y.shape)
print("\n y = ", y)

plt.hist(y)

# %%
# extract SMILES strings

x_smiles = np.reshape(dataframe["SMILES"].values, (len(dataframe), 1))[:,0]

print("Shape of x_smiles = ", x_smiles.shape)
print(x_smiles[0])
# display(Chem.MolFromSmiles(x_smiles[0]))

# %%
x_smiles = x_smiles
# [:100]
y = y
# [:100]

# %%
x_smiles_to_y_dict = dict(list(zip(x_smiles, y)))


# %% [markdown]
# # Prepare Data Split

# %%
# choose settings for dictionary that contains indices for data splits

settings_dict["split_type"] = "scaff"
# rand" # choose "rand" for random split and "scaff" for scaffold split
settings_dict["split_type_rand_stratified"] = False # given a random split and a classification problem, choose whether to stratify the split
settings_dict["k_splits"] = 2 # choose number of cross validation folds k_splits
settings_dict["m_reps"] = 3 # choose number of random seeds m_reps with which the cross validation scheme is repeated
settings_dict["random_state_cv"] = 42 # choose random state

# %%
# construct dictionary that contains indices for data splits

if settings_dict["split_type"] == "rand" and settings_dict["split_type_rand_stratified"] == False:
    
    data_split_dict = create_data_split_dict_random(x_smiles = x_smiles,
                                                    k_splits = settings_dict["k_splits"],
                                                    m_reps = settings_dict["m_reps"],
                                                    random_state_cv = settings_dict["random_state_cv"])

elif settings_dict["split_type"] == "rand" and settings_dict["split_type_rand_stratified"] == True:
    
    data_split_dict = create_data_split_dict_random_strat(x_smiles = x_smiles,
                                                          y = y,
                                                          k_splits = settings_dict["k_splits"],
                                                          m_reps = settings_dict["m_reps"],
                                                          random_state_cv = settings_dict["random_state_cv"])
    
elif settings_dict["split_type"] == "scaff":
    
    data_split_dict = create_data_split_dict_scaffold(x_smiles = x_smiles,
                                                      k_splits = settings_dict["k_splits"],
                                                      m_reps = settings_dict["m_reps"],
                                                      scaffold_func = "Bemis_Murcko_generic",
                                                      random_state_cv = settings_dict["random_state_cv"])

# %% [markdown]
# # Evaluate Models

# %%
# choose ECFP hyperparameters

import sys
pool_method_ = sys.argv[1]
settings_dict["ecfp_settings"] = {"mol_to_invs_function": ecfp_invariants, # ecfp_invariants or fcfp_invariants
                                  "radius": 2, # 0 or 1 or 2 or 3 ...
                                  "pool_method": pool_method_, # "hashed" or "sort_and_slice" or "filtered" or "mim"
                                  "dimension": 1024, # 256 or 512 or 1024 or 2048 or 4096 ...
                                  "use_bond_invs": True, # True or False
                                  "use_chirality": True, # True or False
                                  "use_counts": False} # True or False

# %%
# chose ml model: random forest or multilayer perceptron

settings_dict["ml_model"] = "rf" # "rf" or "mlp"

# %%
# choose rf hyperparameters

settings_dict["rf_settings"] = {"n_estimators" : 100,
                                "max_depth" : None,
                                "min_samples_leaf" : 1,
                                "min_samples_split" : 2,
                                "bootstrap" : True,
                                "max_features": "sqrt",
                                "random_state" : 42}

if settings_dict["task_type"] == "regression":

    settings_dict["rf_settings"]["criterion"] = "squared_error"

elif settings_dict["task_type"] == "classification":

    settings_dict["rf_settings"]["criterion"] = "gini"

# %%
# choose mlp hyperparameters

settings_dict["mlp_settings"] = {"architecture" : list(arch(settings_dict["ecfp_settings"]["dimension"], 1, 512, 5)),
                                "hidden_activation" : torch.nn.ReLU(),
                                "use_bias" : True,
                                "hidden_dropout_rate" : 0.25,
                                "hidden_batchnorm" : True,
                                "batch_size" : 64,
                                "dataloader_shuffle" : True,
                                "dataloader_drop_last" : True,
                                "learning_rate" : 5e-4,
                                "lr_lambda" : lambda epoch: max(0.98**epoch, 1e-2),
                                "lr_last_epoch": 0,
                                "weight_decay" : 0.1,
                                "num_epochs" : 1,
                                "optimiser" : torch.optim.AdamW,
                                "print_results_per_epochs" : None}

if settings_dict["task_type"] == "regression":

    settings_dict["mlp_settings"]["output_activation"] = torch.nn.Identity()
    settings_dict["mlp_settings"]["loss_function"] = torch.nn.MSELoss()
    settings_dict["mlp_settings"]["performance_metrics"] = "regression"

elif settings_dict["task_type"] == "classification":

    settings_dict["mlp_settings"]["output_activation"] = torch.nn.Sigmoid()
    settings_dict["mlp_settings"]["loss_function"] = torch.nn.BCELoss()
    settings_dict["mlp_settings"]["performance_metrics"] = "classification"

# %%
# # ECFP

# # dictionary to save results over the k_splits-fold cross validation with m_reps random seeds
# scores_dict = {}

# # ind_train and ind_test contain the indices for the data split corresponding to the k-th fold with the m-th random seed
# for ((m, k), (ind_train, ind_test)) in data_split_dict.items():
    
#     # create ecfp featuriser
#     featuriser = create_ecfp_featuriser(ecfp_settings = settings_dict["ecfp_settings"], 
#                                         x_smiles_train = x_smiles[ind_train], 
#                                         y_train = y[ind_train], 
#                                         discretise_y = True if settings_dict["task_type"] == "regression" else False,
#                                         base = 2, 
#                                         random_state = 42)
    
#     # create ecfp-based feature matrices
#     X_train = featuriser(x_smiles[ind_train])
#     X_test = featuriser(x_smiles[ind_test])

#     # create ml model + train ml model + make predictions on test set
#     if settings_dict["ml_model"] == "rf":
        
#         rf_model = create_rf_model(settings_dict["rf_settings"], settings_dict["task_type"])
#         rf_model.fit(X_train, y[ind_train])
#         y_test_pred = make_rf_prediction(rf_model, X_test, settings_dict["task_type"])
        
#     if settings_dict["ml_model"] == "mlp":
        
#         mlp_model = create_mlp_model(settings_dict["mlp_settings"])
#         (loss_curve_train, loss_curve_test) = train_mlp_model(mlp_model, settings_dict["mlp_settings"], X_train, y[ind_train], X_test, y[ind_test])
#         plt.plot(loss_curve_train)
#         plt.plot(loss_curve_test)
#         y_test_pred = make_mlp_prediction(mlp_model, X_test)
        
#     # record scores
#     print(m, k)
#     if settings_dict["task_type"] == "regression":
#         scores_dict[(m, k)] = regression_scores(y[ind_test], y_test_pred, #display_results = True)
#     elif settings_dict["task_type"] == "classification":
#         scores_dict[(m, k)] = binary_classification_scores(y[ind_test], y_test_pred, #display_results = True)
        
# # summarise, #display and save scores for this experiment
# summarise_#display_and_save_results_and_settings(scores_dict, settings_dict, #display_results = True)

# %%
def create_sort_and_slice_e3fp_featuriser(dataset_name,
                                          smiles_train,
                                          x_smiles_to_y_dict,
                                          vec_dimension = 1024, 
                                          break_ties_with = lambda sub_id: sub_id, 
                                          print_train_set_info = True):
   
    # if os.path.exists('e3fp_sub_ids_sorted_list/'+dataset_name+'_sub_ids_sorted_list.npy'):
    #     sub_ids_sorted_list = np.load('e3fp_sub_ids_sorted_list/'+dataset_name+'_sub_ids_sorted_list.npy')
    #     print('loaded sub_ids_sorted_list')
    # else:
    
    e3fp_generator = Fingerprinter(bits=vec_dimension, level=5, radius_multiplier=1.718, 
                    stereo=True, counts=False, include_disconnected=True, 
                    rdkit_invariants=True, exclude_floating=True, remove_duplicate_substructs=True)


# (bits=vec_dimension, level=5, radius_multiplier=1.718,
#                                    stereo=True, counts=False, include_disconnected=True,
#                                    rdkit_invariants=True, exclude_floating=True, remove_duplicate_substructs=True)


    substructures_per_mol = {}
    for smiles in smiles_train:
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        # , useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf = conf_, mol = mol)
            substructures_per_mol[smiles] = list(e3fp_generator.get_all_shells().keys())
        except:
            print('bad conformer')
            continue 

    sub_ids_to_prevs_dict = {}
    for s in smiles_train:
        if s in substructures_per_mol.keys():
            for sub_id in substructures_per_mol[s] :
                sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key = lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse = True)
    # np.save('e3fp_sub_ids_sorted_list/'+settings_dict["dataset_name"]+'_sub_ids_sorted_list.npy', np.array(sub_ids_sorted_list))
    
    
    # create auxiliary function that generates standard unit vectors in NumPy
    def standard_unit_vector(dim, k):
        
        vec = np.zeros(dim, dtype = int)
        vec[k] = 1
        
        return vec
    
    # create one-hot encoder for the first vec_dimension substructure identifiers in sub_ids_sorted_list; all other substructure identifiers are mapped to a vector of 0s
    def sub_id_one_hot_encoder(sub_id):
        
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    # create a function ecfp_featuriser that maps RDKit mol objects to vectorial ECFPs via a Sort & Slice substructure pooling operator trained on mols_train
    # count_bad_conformers = 0
    def ecfp_featuriser(s):


        mol = Chem.MolFromSmiles(s)
        mol.SetProp("_Name", s)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        # AllChem.UFFOptimizeMolecule(mol)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf = conf_, mol = mol)
            substructure_list = e3fp_generator.get_all_shells().keys()

            sub_id_list = [sub_idd for (sub_id, count) in dict(Counter(substructure_list)).items() for sub_idd in [sub_id]*count]

            ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)
        
            # print('len ecfp_vector', len(ecfp_vector))
            return (ecfp_vector, x_smiles_to_y_dict[s])
            

        except:
            print('bad conformer')
            # count_bad_conformers += 1
            return None
            
            # continue 
    
    # print information on training set
    if print_train_set_info == True:
        print('sub_ids_sorted_list', len(sub_ids_sorted_list))
        # print('num of bad conformers', count_bad_conformers)
        print("Number of compounds in molecular training set which have conformers= ", len(substructures_per_mol))
        print("Number of unique circular substructures with the specified parameters in molecular training set = ", len(sub_ids_to_prevs_dict))

    return ecfp_featuriser

# %%
def discretise(y_cont, n_bins = 2, strategy = "uniform"):
    """
    Discretise continuous array.
    """
    
    discretiser = KBinsDiscretizer(n_bins = n_bins, encode = "ordinal", strategy = strategy)
    y_disc = list(discretiser.fit_transform(np.array(y_cont).reshape(-1,1)).reshape(-1).astype(int))
    
    return y_disc

# %%
from sklearn.feature_selection import mutual_info_classif

def create_mim_e3fp_featuriser(dataset_name,
                                  smiles_train,
                                  x_smiles_to_y_dict,
                                  vec_dimension=1024,
                                  print_train_set_info=True):
    
    e3fp_generator = Fingerprinter(bits=vec_dimension, level=5, radius_multiplier=1.718,
                                   stereo=True, counts=False, include_disconnected=True,
                                   rdkit_invariants=True, exclude_floating=True, remove_duplicate_substructs=True)
    
    substructures_per_mol = {}
    for smiles in smiles_train:
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf=conf_, mol=mol)
            substructures_per_mol[smiles] = list(e3fp_generator.get_all_shells().keys())
        except:
            print('bad conformer')
            continue
    
    # Compute frequency of substructures
    sub_ids_to_prevs_dict = Counter([sub_id for s in substructures_per_mol for sub_id in substructures_per_mol[s]])
    
    # Sort by frequency
    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict.keys(), key=lambda sub_id: sub_ids_to_prevs_dict[sub_id], reverse=True)
    
    # Selecting the most relevant features using Mutual Information
    def compute_mutual_info():
        X = []
        y = []
        
        for s in smiles_train:
            if s in substructures_per_mol:
                feature_vector = np.zeros(len(sub_ids_sorted_list))
                for sub_id in substructures_per_mol[s]:
                    if sub_id in sub_ids_sorted_list:
                        feature_vector[sub_ids_sorted_list.index(sub_id)] = 1
                X.append(feature_vector)
                y.append(x_smiles_to_y_dict[s])
        
        X = np.array(X)
        y = np.array(y)
        return mutual_info_classif(X, y = discretise(y, n_bins = 2, strategy = "quantile"), discrete_features=True)
    
    mi_scores = compute_mutual_info()
    top_features = np.argsort(mi_scores)[-vec_dimension:]
    
    # Folding into lower-dimensional representation
    def sub_id_folding_encoder(sub_id):
        if sub_id in sub_ids_sorted_list:
            idx = sub_ids_sorted_list.index(sub_id)
            if idx in top_features:
                folded_idx = np.where(top_features == idx)[0][0]  # Map to reduced dimension
                vec = np.zeros(vec_dimension)
                vec[folded_idx] = 1
                return vec
        return np.zeros(vec_dimension)
    
    # Function to featurize molecules
    def fingerprint_featuriser(s):
        mol = Chem.MolFromSmiles(s)
        mol.SetProp("_Name", s)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf=conf_, mol=mol)
            substructure_list = list(e3fp_generator.get_all_shells().keys())
            folded_vector = np.sum(np.array([sub_id_folding_encoder(sub_id) for sub_id in substructure_list]), axis=0)
            return (folded_vector, x_smiles_to_y_dict[s])
        except:
            print('bad conformer')
            return None
    
    if print_train_set_info:
        print('Dataset:', dataset_name)
        print('Unique substructures:', len(sub_ids_sorted_list))
        print('Selected features:', len(top_features))
    
    return fingerprint_featuriser

def create_mim_e3fp_featuriser(dataset_name,
                                  smiles_train,
                                  x_smiles_to_y_dict,
                                  vec_dimension=1024,
                                  print_train_set_info=True):
    
    e3fp_generator = Fingerprinter(bits=vec_dimension, level=5, radius_multiplier=1.718,
                                   stereo=True, counts=False, include_disconnected=True,
                                   rdkit_invariants=True, exclude_floating=True, remove_duplicate_substructs=True)
    
    substructures_per_mol = {}
    for smiles in smiles_train:
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf=conf_, mol=mol)
            substructures_per_mol[smiles] = list(e3fp_generator.get_all_shells().keys())
        except:
            print('bad conformer')
            continue
    
    # Compute frequency of substructures
    sub_ids_to_prevs_dict = Counter([sub_id for s in substructures_per_mol for sub_id in substructures_per_mol[s]])
    
    # Sort by frequency
    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict.keys(), key=lambda sub_id: sub_ids_to_prevs_dict[sub_id], reverse=True)
    
    # Selecting the most relevant features using Mutual Information
    def compute_mutual_info():
        X = []
        y = []
        
        for s in smiles_train:
            if s in substructures_per_mol:
                feature_vector = np.zeros(len(sub_ids_sorted_list))
                for sub_id in substructures_per_mol[s]:
                    if sub_id in sub_ids_sorted_list:
                        feature_vector[sub_ids_sorted_list.index(sub_id)] = 1
                X.append(feature_vector)
                y.append(x_smiles_to_y_dict[s])
        
        X = np.array(X)
        y = np.array(y)
        return mutual_info_classif(X, y = discretise(y, n_bins = 2, strategy = "quantile"), discrete_features=True)
    
    mi_scores = compute_mutual_info()
    top_features = np.argsort(mi_scores)[-vec_dimension:]
    
    # Folding into lower-dimensional representation
    def sub_id_folding_encoder(sub_id):
        if sub_id in sub_ids_sorted_list:
            idx = sub_ids_sorted_list.index(sub_id)
            if idx in top_features:
                folded_idx = np.where(top_features == idx)[0][0]  # Map to reduced dimension
                vec = np.zeros(vec_dimension)
                vec[folded_idx] = 1
                return vec
        return np.zeros(vec_dimension)
    
    # Function to featurize molecules
    def fingerprint_featuriser(s):
        mol = Chem.MolFromSmiles(s)
        mol.SetProp("_Name", s)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=50000, useRandomCoords=True)
        
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf=conf_, mol=mol)
            substructure_list = list(e3fp_generator.get_all_shells().keys())
            folded_vector = np.sum(np.array([sub_id_folding_encoder(sub_id) for sub_id in substructure_list]), axis=0)
            return (folded_vector, x_smiles_to_y_dict[s])
        except:
            print('bad conformer')
            return None
    
    if print_train_set_info:
        print('Dataset:', dataset_name)
        print('Unique substructures:', len(sub_ids_sorted_list))
        print('Selected features:', len(top_features))
    
    return fingerprint_featuriser


# %%
settings_dict["ml_model"] = "mlp" # "rf" or "mlp"

# %%
# E3fp sns

# dictionary to save results over the k_splits-fold cross validation with m_reps random seeds
scores_dict = {}

# ind_train and ind_test contain the indices for the data split corresponding to the k-th fold with the m-th random seed
for ((m, k), (ind_train, ind_test)) in data_split_dict.items():
    
    # create ecfp featuriser
    # breakpoint()    
    if settings_dict["ecfp_settings"]["pool_method"] == "sort_and_slice":
        featuriser = create_sort_and_slice_e3fp_featuriser(dataset_name= settings_dict["dataset_name"],
                                                        smiles_train = x_smiles[ind_train],
                                                            x_smiles_to_y_dict = x_smiles_to_y_dict,
                                            vec_dimension = 1024, 
                                            print_train_set_info = True)
    elif settings_dict["ecfp_settings"]["pool_method"] == "mim":
        featuriser = create_mim_e3fp_featuriser(dataset_name= settings_dict["dataset_name"],
                                                        smiles_train = x_smiles[ind_train],
                                                            x_smiles_to_y_dict = x_smiles_to_y_dict,
                                            vec_dimension = 1024, 
                                            print_train_set_info = True)
    elif settings_dict["ecfp_settings"]["pool_method"] == "hashed":
        featuriser = create_hashed_e3fp_featuriser(dataset_name= settings_dict["dataset_name"],
                                                        smiles_train = x_smiles[ind_train],
                                                            x_smiles_to_y_dict = x_smiles_to_y_dict,
                                            vec_dimension = 1024, 
                                            print_train_set_info = True)
    

    return_train = list(map( featuriser , x_smiles[ind_train]))
    # print(return_train)
    return_train = list(filter(lambda tpl: tpl is not None, return_train))
    return_test = list(map( featuriser , x_smiles[ind_test]))
    return_test = list(filter(lambda tpl: tpl is not None, return_test))

    X_train = []
    y_train = []
    X_test = []
    y_test = []


    # print(return_train)
    # for (x,y) in return_train:
    #     if x is not None and y is not None:
    #         X_train.append(x)
    #         y_train.append(y)
    
    # for (x,y) in return_test:
    #     if x is not None and y is not None:
    #         X_test.append(x)
    #         y_test.append(y)

    X_train, y_train = zip(*return_train)
    X_test, y_test = zip(*return_test)

    X_train= np.array(X_train)
    X_test= np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    # create ml model + train ml model + make predictions on test set
    if settings_dict["ml_model"] == "rf":
        
        rf_model = create_rf_model(settings_dict["rf_settings"], settings_dict["task_type"])
        X_train= np.array(X_train)
        rf_model.fit(X_train, y_train)
        y_test_pred = make_rf_prediction(rf_model, X_test, settings_dict["task_type"])
        print('y_test_pred', y_test_pred.shape)
        
    if settings_dict["ml_model"] == "mlp":
        
        mlp_model = create_mlp_model(settings_dict["mlp_settings"])
        (loss_curve_train, loss_curve_test) = train_mlp_model(mlp_model, settings_dict["mlp_settings"], X_train, y_train, X_test, y_test)
        plt.plot(loss_curve_train)
        plt.plot(loss_curve_test)
        y_test_pred = make_mlp_prediction(mlp_model, X_test)
        
    # record scores
    print('seed: ', m, ' fold: ', k)
    if settings_dict["task_type"] == "regression":
        scores_dict[(m, k)] = regression_scores(y_test, y_test_pred, display_results = True)
    elif settings_dict["task_type"] == "classification":
        scores_dict[(m, k)] = binary_classification_scores(y_test, y_test_pred, display_results = True)
        
# summarise, display and save scores for this experiment
summarise_display_and_save_results_and_settings(scores_dict, settings_dict, display_results = True)
breakpoint()

# %%

# run full battery of experiments for one data set and data splitting type with both rfs and mlps

# first clear folder from previous experimental results
# delete_all_files_in_folder("results/" + settings_dict["dataset_name"] + "/" + settings_dict["split_type"] + "/")

for pool_method in ["sort_and_slice","hashed", "filtered", "mim"]:
    for dimension in [1024, 512, 2048, 4096]:
        for inv_func in [ecfp_invariants, fcfp_invariants]:
            for radius in [1, 2, 3]:

                print("pool_method = ", pool_method)
                print("dimension = ", dimension)
                print("inv_func = ", inv_func)
                print("radius = ", radius, "\n")

                # choose ECFP hyperparameters
                settings_dict["ecfp_settings"] = {"mol_to_invs_function": inv_func,
                                                  "radius": radius,
                                                  "pool_method": pool_method,
                                                  "dimension": dimension,
                                                  "use_bond_invs": True,
                                                  "use_chirality": True,
                                                  "use_counts": False}


                # run rf- and mlp models and save results
                scores_dict_rf = {}
                scores_dict_mlp = {}

                # ind_train and ind_test contain the indices for the data split corresponding to the k-th fold with the m-th random seed
                for ((m, k), (ind_train, ind_test)) in data_split_dict.items():
                    
                    # create ecfp featuriser
                    print("creating featuriser")
                    print(y)
                    featuriser = create_sort_and_slice_e3fp_featuriser(dataset_name= settings_dict["dataset_name"],
                                                        smiles_train = x_smiles[ind_train], 
                                                        max_radius = 2, 
                                                            pharm_atom_invs = False, 
                                                            bond_invs = True, 
                                                            chirality = False, 
                                                            sub_counts = True, 
                                                            vec_dimension = 1024, 
                                                            break_ties_with = lambda sub_id: sub_id, 
                                                            print_train_set_info = True
                    )                                          



                    return_train = list(map( featuriser , x_smiles[ind_train]))
                    return_test = list(map( featuriser , x_smiles[ind_test]))

                    print(return_train)
                    X_train = []
                    y_train = []
                    X_test = []
                    y_test = []


                    print(return_train)
                    for (x,y) in return_train:
                        if x is not None and y is not None:
                            X_train.append(x)
                            y_train.append(y)
                    
                    for (x,y) in return_test:
                        if x is not None and y is not None:
                            X_test.append(x)
                            y_test.append(y)

                    X_train= np.array(X_train)
                    X_test= np.array(X_test)
                    y_train = np.array(y_train)
                    y_test = np.array(y_test)

                    
                    # create rf model + train ml model + make predictions on test set
                    rf_model = create_rf_model(settings_dict["rf_settings"], settings_dict["task_type"])
                    rf_model.fit(X_train, y_train)
                    y_test_pred = make_rf_prediction(rf_model, X_test, settings_dict["task_type"])

                    # record rf scores
                    print(m, k, "rf")
                    if settings_dict["task_type"] == "regression":
                        scores_dict_rf[(m, k)] = regression_scores(y_test, y_test_pred, display_results = False)
                    elif settings_dict["task_type"] == "classification":
                        scores_dict_rf[(m, k)] = binary_classification_scores(y_test, y_test_pred, display_results = False)
                    
                    # create mlp model + train ml model + make predictions on test set
                    settings_dict["mlp_settings"]["architecture"][0] = dimension
                    mlp_model = create_mlp_model(settings_dict["mlp_settings"])
                    (loss_curve_train, loss_curve_test) = train_mlp_model(mlp_model, settings_dict["mlp_settings"], X_train, y_train, X_test, y_test)
                    y_test_pred = make_mlp_prediction(mlp_model, X_test)

                    # record mlp scores
                    print(m, k, "mlp")
                    if settings_dict["task_type"] == "regression":
                        scores_dict_mlp[(m, k)] = regression_scores(y_test, y_test_pred, display_results = False)
                    elif settings_dict["task_type"] == "classification":
                        scores_dict_mlp[(m, k)] = binary_classification_scores(y_test, y_test_pred, display_results = False) 
                    
                # summarise, display and save scores for this experiment
                settings_dict["ml_model"] = "rf"
                summarise_display_and_save_results_and_settings(scores_dict_rf, settings_dict, display_results = True)
                settings_dict["ml_model"] = "mlp"
                summarise_display_and_save_results_and_settings(scores_dict_mlp, settings_dict, display_results = True)
                print("\n \n \n")

# %% [markdown]
# # Visualisation of Results

# %% [markdown]
# Dataset names:
# 
#     "ames_mutagenicity",
#     "aqsoldb_solubility",
#     "lit_pcba_esr_ant",
#     "moleculenet_lipophilicity",
#     "postera_sars_cov_2_mpro".
#     
# Available classification metrics: 
# 
#     "PRC-AUC",
#     "AUROC", 
#     "Accuracy", 
#     "Balanced Accuracy", 
#     "F1-Score", 
#     "MCC", 
#     "Sensitivity", 
#     "Specificity", 
#     "Precision", 
#     "Negative Predictive Value", 
#     "Test Cases", 
#     "Negative Test Cases", 
#     "Positive Test Cases".
# 
# Available regression metrics:
# 
#     "MAE", 
#     "MedAE", 
#     "RMSE", 
#     "MaxAE", 
#     "MSE", 
#     "PearsonCorr", 
#     "R2Coeff", 
#     "Test Cases".

# %%
visualise_bar_charts(dataset_name = "moleculenet_lipophilicity", # specify dataset name
                     split_type = "scaff", # specify split type "rand" or "scaff" (for random or scaffold split)
                     metric = "MAE", # specify performance metric,
                     y_lims = None, # specify limits of y-axis (set to "None" for automatic limits)
                     y_unit = " [logD]") # specify unit for y-axis (if applicable)

# %%
visualise_box_plots(dataset_name = "moleculenet_lipophilicity", # specify dataset name,
                    metric = "MAE", # specify performance metric
                    y_unit = " [logD]", # specify unit for y-axis if applicable
                    show_legend = False, # show legend or not
                    show_x_ticks = True) # show text below subplots or not

# %%



