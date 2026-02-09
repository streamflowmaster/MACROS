from rdkit import Chem
from rdkit.Chem import DataStructs
from functional_group import functional_groups,get_functional_groups
import numpy as np
def fingerprint_similarity_metric(mol1: str, mol2: str):
    """
    Calculate the similarity between two molecules.
    """
    try:
        mol1 = Chem.MolFromSmiles(mol1)
        mol2 = Chem.MolFromSmiles(mol2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    except:
        similarity = 0
    return similarity


def get_InchiKey(smi):
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key
    except:
        return None


def judge_InchiKey(key1, key2):
    if key1 is None or key2 is None:
        return 0
    return 1 if key1 == key2 else 0


def equivalent_similarity_metric(smi1, smi2):
    key1 = get_InchiKey(smi1)
    if key1 is None:
        return 0
    key2 = get_InchiKey(smi2)
    if key2 is None:
        return 0
    return judge_InchiKey(key1, key2)

def functional_group_similarity_metric(mol1: str, mol2: str):
    """
    Calculate the similarity between two molecules.
    """
    mol1_func_groups = get_functional_groups(mol1)
    mol2_func_groups = get_functional_groups(mol2)
    similarity = np.zeros(len(mol1_func_groups))
    if mol1_func_groups is None or mol2_func_groups is None:
        return similarity

    for i in range(len(mol1_func_groups)):
        if mol1_func_groups[i] == mol2_func_groups[i]:
            similarity[i] = 1
    return similarity



if __name__ == '__main__':

    # test the function
    mol1 = 'CCOO'
    mol2 = 'OOCC'
    print(fingerprint_similarity_metric(mol1, mol2))
    print(equivalent_similarity_metric(mol1, mol2))

    from rdkit import Chem
    import warnings

    warnings.filterwarnings(action="ignore")