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

def create_sort_and_slice_e3fp_featuriser(smiles_train,
                                          max_radius = 2, 
                                          pharm_atom_invs = False, 
                                          bond_invs = True, 
                                          chirality = False, 
                                          sub_counts = True, 
                                          vec_dimension = 1024, 
                                          break_ties_with = lambda sub_id: sub_id, 
                                          print_train_set_info = True):
    """
    Creates a function "ecfp_featuriser" that maps RDKit mol objects to vectorial extended-connectivity fingerprints (ECFPs) pooled via a trained Sort & Slice operator (instead of classical hash-based folding).
    See also "Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for Extended-Connectivity Fingerprints" by Dablander, Hanser, Lambiotte and Morris (2024): https://arxiv.org/abs/2403.17954
    
    
    
    INPUTS:
    
    - mols_train (list)            ...  A list of RDKit mol objects [mol_1, mol_2, ...] that are used as the training set to calibrate the Sort & Slice substructure pooling operator.
    
    - max_radius (int)             ...  The maximal radius up to which to generate the integer ECFP-substructure identifiers. Common choices are 1, 2 or 3 (corresponding to maximal diameters of 2, 4, or 6).
    
    - pharm_atom_invs (bool)       ...  If False (= default), then the standard initial atomic invariants from RDKit (including ring membership) are used to generate the ECFPs. 
                                        If True, then instead binary pharmacophoric initial atomic invariants are used to generate a different type of ECFP also referred to as FCFPs.
    
    - bond_invs (bool)             ...  Whether or not to take into account bond invariants when generating the integer ECFP-substructure identifiers (default = True).
    
    - chirality (bool)             ...  Whether or not to take into account chirality when generating the integer ECFP-substructure identifiers (default = False).
    
    - sub_counts (bool)            ...  Whether ecfp_featuriser should generate binary vectorial fingerprints (sub_counts = False) that indicate the mere presence or absence of substructures; 
                                        or integer fingerprints (sub_counts = True) that additionally indicate how many times a substructure is found in the input compound.
    
    - vec_dimension (int)          ...  Length of the vectorial Sort & Slice ECFP. Common choices are 512, 1024, 2048 and 4096. 
                                        Only the vec_dimension most prevalent ECFP-substructures in the training set mols_train are included in the final vector representation.
    
    - break_ties_with (function)   ...  Function to map the integer ECFP-substructure identifiers to values that are used to break ties when sorting the substructure identifiers according to their prevalence in mols_train. 
                                        The default is the identity map (i.e., lambda sub_id: sub_id) which breaks ties using the (arbitrary) ordering defined by the integer substructure identifier themselves.
                                        If break_ties_with = None, then ties are broken automatically as part of the application of Python's sorted() command to sub_ids_to_prevs_dict.
    
    - print_train_set_info (bool)  ...  Whether or not to print out the number of compounds and the number of unique integer ECFP-substructure identifiers with the specified parameters in mols_train.
    
    
    OUTPUT:
    
    - ecfp_featuriser (function)   ...  A function that maps RDKit mol objects to vectorial ECFPs (1-dimensional NumPy arrays of length vec_dimension) via a Sort & Slice substructure pooling operator trained on mols_train.
    
     
     
    EXAMPLE:
    
    First select a training set of RDKit mol objects 

        mols_train = [mol_1, mol_2, ...]
    
    that should be used to calibrate the Sort & Slice operator. This training set can then be employed along with a set of desired ECFP hyperparameter settings to construct a molecular featurisation function:
    
        ecfp_featuriser = construct_sort_and_slice_ecfp_featuriser(mols_train = mols_train, 
                                                                   max_radius = 2, 
                                                                   pharm_atom_invs = False, 
                                                                   bond_invs = True, 
                                                                   chirality = False, 
                                                                   sub_counts = True, 
                                                                   vec_dimension = 1024)
                                                               
    Then ecfp_featuriser(mol) is a 1-dimensional numpy array of length vec_dimension representing the vectorial ECFP for mol pooled via a Sort & Slice operator calibrated on mols_train. 
    
    More specifically, the function ecfp_featuriser can be thought of as

    1. first generating the (multi)set of integer ECFP-substructure identifiers for mol based on the ECFP hyperparameters (max_radius, pharm_atom_invs, bond_invs, chirality, sub_counts) and then
    2. vectorising this (multi)set via a Sort & Slice operator calibrated on mols_train with output dimension vec_dimension (rather than vectorising it via classical hash-based folding).
    
    To now turn any list of RDKit mol objects mols_list into a feature matrix X whose rows correspond to vectorial Sort & Slice ECFPs one can simply run
    
        X = np.array([ecfp_featuriser(mol) for mol in mols_list])
    
    """
    
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
        # AllChem.UFFOptimizeMolecule(mol)
        try:
            conf_ = mol.GetConformer()
            e3fp_generator.run(conf = conf_, mol = mol)
            substructures_per_mol[smiles] = e3fp_generator.get_all_shells().keys()
        except:
            print('bad conformer')
            continue 
        
        # breakpoint()

    # # e3fp_generator.substructs_to_pdb(level=5, bits=None, out_dir='substructs_bits_1024', reorient=True, exact=False)
    # # run(conf = mole.GetConformer(), mol = mole,  return_substruct= True)
    # breakpoint()
    # # create a function sub_id_enumerator that maps a mol object to a dictionary whose keys are the integer substructure identifiers in mol and whose values are the associated substructure counts (i.e., how often each substructure appears in mol)
    # morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius = max_radius,
    #                                                              atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if pharm_atom_invs == True else rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True),
    #                                                              useBondTypes = bond_invs,
    #                                                              includeChirality = chirality)
    
    # sub_id_enumerator = lambda mol: morgan_generator.GetSparseCountFingerprint(mole).GetNonzeroElements()
    
    # construct dictionary that maps each integer substructure identifier sub_id in mols_train to its associated prevalence (i.e., to the total number of compounds in mols_train that contain sub_id at least once)
    sub_ids_to_prevs_dict = {}
    for s in smiles_train:
        if s in substructures_per_mol.keys():
            for sub_id in substructures_per_mol[s] :
                sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    # sub_ids_to_prevs_dict = Counter(substructures_per_mol.values())

    # breakpoint()

    # create list of integer substructure identifiers sorted by prevalence in mols_train
    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key = lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse = True)
    # breakpoint()
    
    # create auxiliary function that generates standard unit vectors in NumPy
    def standard_unit_vector(dim, k):
        
        vec = np.zeros(dim, dtype = int)
        vec[k] = 1
        
        return vec
    
    # create one-hot encoder for the first vec_dimension substructure identifiers in sub_ids_sorted_list; all other substructure identifiers are mapped to a vector of 0s
    def sub_id_one_hot_encoder(sub_id):
        
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    # create a function ecfp_featuriser that maps RDKit mol objects to vectorial ECFPs via a Sort & Slice substructure pooling operator trained on mols_train
    def ecfp_featuriser(s):

        # create list of integer substructure identifiers contained in input mol object (multiplied by how often they are structurally contained in mol if sub_counts = True)
        if sub_counts == True:
            # breakpoint()
            sub_id_list = [sub_idd for (sub_id, count) in dict(Counter(substructures_per_mol[s])).items() for sub_idd in [sub_id]*count]
        else:
            sub_id_list = list(substructures_per_mol[s])
        
        # create molecule-wide vectorial representation by summing up one-hot encoded substructure identifiers
        ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)
    
        print('len ecfp_vector', len(ecfp_vector))
        return ecfp_vector
    
    # print information on training set
    if print_train_set_info == True:
        print("Number of compounds in molecular training set which have conformers= ", len(substructures_per_mol))
        print("Number of unique circular substructures with the specified parameters in molecular training set = ", len(sub_ids_to_prevs_dict))

    return ecfp_featuriser


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
    featuriser = create_sort_and_slice_e3fp_featuriser(smiles_train = dataframe['SMILES'],
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
    



