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

from sklearn.feature_selection import mutual_info_classif

def create_sort_and_slice_e3fp_featuriser(dataset_name,
                                          smiles_train,
                                          x_smiles_to_y_dict,
                                          max_radius = 2, 
                                          pharm_atom_invs = False, 
                                          bond_invs = True, 
                                          chirality = False, 
                                          sub_counts = True, 
                                          vec_dimension = 1024, 
                                          break_ties_with = lambda sub_id: sub_id, 
                                          print_train_set_info = True):

    
    e3fp_generator = Fingerprinter(bits=1024, level=5, radius_multiplier=1.718, 
                    stereo=True, counts=False, include_disconnected=True, 
                    rdkit_invariants=False, exclude_floating=True, remove_duplicate_substructs=True)
    # mole = mols_train[0]
    substructures_per_mol = {}
    for smiles in smiles_train:
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf = conf_, mol = mol)
            substructures_per_mol[smiles] = e3fp_generator.get_all_shells().keys()
        except:
            print('bad conformer')
            continue 

    sub_ids_to_prevs_dict = {}
    for s in smiles_train:
        if s in substructures_per_mol.keys():
            for sub_id in substructures_per_mol[s] :
                sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key = lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse = True)
    
    
    def standard_unit_vector(dim, k):
        
        vec = np.zeros(dim, dtype = int)
        vec[k] = 1
        
        return vec
    
    # create one-hot encoder for the first vec_dimension substructure identifiers in sub_ids_sorted_list; all other substructure identifiers are mapped to a vector of 0s
    def sub_id_one_hot_encoder(sub_id):
        
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    # create a function ecfp_featuriser that maps RDKit mol objects to vectorial ECFPs via a Sort & Slice substructure pooling operator trained on mols_train
    count_bad_conformers = 0
    def ecfp_featuriser(s):


        mol = Chem.MolFromSmiles(s)
        mol.SetProp("_Name", s)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True)
        # AllChem.UFFOptimizeMolecule(mol)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf = conf_, mol = mol)
            substructure_list = e3fp_generator.get_all_shells().keys()
            # create list of integer substructure identifiers contained in input mol object (multiplied by how often they are structurally contained in mol if sub_counts = True)
            if sub_counts == True:
                # breakpoint()
                
                sub_id_list = [sub_idd for (sub_id, count) in dict(Counter(substructure_list)).items() for sub_idd in [sub_id]*count]
            # else:
            #     sub_id_list = list(substructures_per_mol[s])
            
            ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)
        
            return (ecfp_vector, x_smiles_to_y_dict[s])
            

        except:
            print('bad conformer')
            count_bad_conformers += 1
            return (None, None)

    if print_train_set_info == True:
        print('dataset_name', dataset_name)
        print('sub_ids_sorted_list', len(sub_ids_sorted_list))
        print('num of bad conformers', count_bad_conformers)
        print("Number of compounds in molecular training set which have conformers= ", len(substructures_per_mol))
        print("Number of unique circular substructures with the specified parameters in molecular training set = ", len(sub_ids_to_prevs_dict))

    return ecfp_featuriser

def create_mim_e3fp_featuriser(dataset_name,
                                  smiles_train,
                                  x_smiles_to_y_dict,
                                  vec_dimension=1024,
                                  print_train_set_info=True):
    
    e3fp_generator = Fingerprinter(bits=vec_dimension, level=5, radius_multiplier=1.718,
                                   stereo=True, counts=False, include_disconnected=True,
                                   rdkit_invariants=False, exclude_floating=True, remove_duplicate_substructs=True)
    
    substructures_per_mol = {}
    for smiles in smiles_train:
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, maxAttempts=5000)
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
        return mutual_info_classif(X, y, discrete_features=True)
    
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
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True)
        
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf=conf_, mol=mol)
            substructure_list = list(e3fp_generator.get_all_shells().keys())
            folded_vector = np.sum(np.array([sub_id_folding_encoder(sub_id) for sub_id in substructure_list]), axis=0)
            return (folded_vector, x_smiles_to_y_dict[s])
        except:
            print('bad conformer')
            return (None, None)
    
    if print_train_set_info:
        print('Dataset:', dataset_name)
        print('Unique substructures:', len(sub_ids_sorted_list))
        print('Selected features:', len(top_features))
    
    return fingerprint_featuriser





if __name__ == "__main__":
    settings_dict = {}
    # MoleculeNet Lipophilicity

    settings_dict["dataset_name"] = "moleculenet_lipophilicity"
    settings_dict["task_type"] = "regression"
    settings_dict["prop_name"] = "exp"
    dataframe = pd.read_csv("data/" + settings_dict["dataset_name"] + "/" + "clean_data.csv", sep = ",")
    # smiles = dataframe["SMILES"]
    dataframe['mol'] = dataframe['SMILES'].apply(Chem.MolFromSmiles)
    # print(dataframe['SMILES'][0])
    featuriser = create_mim_e3fp_featuriser(smiles_train = dataframe['SMILES'],
                                        #   mols_train = dataframe['mol'] [:100] , 
                                          max_radius = 2, 
                                          pharm_atom_invs = False, 
                                          bond_invs = True, 
                                          chirality = False, 
                                          sub_counts = True, 
                                          vec_dimension = 1024, 
                                          break_ties_with = lambda sub_id: sub_id, 
                                          print_train_set_info = True)
    # breakpoint()
    X = list(map(featuriser, dataframe['SMILES'].tolist()))
    print(X)
    



