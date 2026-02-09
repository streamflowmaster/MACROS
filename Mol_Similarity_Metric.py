from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolAlign,rdFingerprintGenerator
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import numpy as np
from typing import List, Tuple,Dict,Union

# morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# # 1. 基于分子指纹的相似性（Tanimoto 系数）
# def fingerprint_similarity(mol1, mol2):
#     # 使用 MorganGenerator 生成 Morgan 指纹
#     fp1 = morgan_gen.GetFingerprint(mol1)
#     fp2 = morgan_gen.GetFingerprint(mol2)
#
#     # 计算 Tanimoto 系数
#     tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
#     return tanimoto

def fingerprint_similarity(mol1, mol2):
    """
    Calculate the similarity between two molecules.
    """

    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity

# 2. 基于分子描述符的相似性（欧几里得距离）
def descriptor_similarity(mol1, mol2):
    descriptors = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors
    ]
    vec1 = np.array([desc(mol1) for desc in descriptors])
    vec2 = np.array([desc(mol2) for desc in descriptors])
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    max_dist = np.linalg.norm(vec1) + np.linalg.norm(vec2)
    similarity = 1 / (1 + euclidean_dist / max_dist)
    return similarity

# 3. 基于图的相似性（最大公共子图 MCS）
def mcs_similarity(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1, mol2], timeout=60)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return 0.0
    n_atoms_mcs = mcs_mol.GetNumAtoms()
    n_atoms_mol1 = mol1.GetNumAtoms()
    n_atoms_mol2 = mol2.GetNumAtoms()
    similarity = n_atoms_mcs / max(n_atoms_mol1, n_atoms_mol2)
    return similarity

# 4. 基于三维结构的相似性（形状相似性）
def shape_similarity(mol1, mol2):
    mol1_3d = Chem.Mol(mol1)
    mol2_3d = Chem.Mol(mol2)
    mol1_3d = Chem.AddHs(mol1_3d)
    mol2_3d = Chem.AddHs(mol2_3d)
    AllChem.EmbedMolecule(mol1_3d, randomSeed=42)
    AllChem.EmbedMolecule(mol2_3d, randomSeed=42)
    try:
        o3a = rdMolAlign.GetO3A(mol1_3d, mol2_3d)
        score = o3a.Score()
        max_score = min(mol1_3d.GetNumHeavyAtoms(), mol2_3d.GetNumHeavyAtoms()) * 10
        similarity = score / max_score if max_score > 0 else 0.0
        return similarity
    except:
        return 0.0

def summary_similarity(mol1,mol2):
    if mol1 == None or mol2 == None:
        return 0
    else:
        summary = (shape_similarity(mol1,mol2)+mcs_similarity(mol1,mol2)+descriptor_similarity(mol1,mol2)+fingerprint_similarity(mol1,mol2))/4
        return summary

def total_similarity(mol1,mol2):
    if mol1 == None or mol2 == None:
        return {
            'shape_similarity':0,
            'mcs_similarity':0,
            'descriptor_similarity':0,
            'fingerprint_similarity':0
        }
    else:
        return {
            'shape_similarity':shape_similarity(mol1,mol2),
            'mcs_similarity':mcs_similarity(mol1,mol2),
            'descriptor_similarity':descriptor_similarity(mol1,mol2),
            'fingerprint_similarity':fingerprint_similarity(mol1,mol2)
        }


def batch_total_similarity(mol_list1: List[Union[str, Chem.Mol]],
                           mol_list2: List[Union[str, Chem.Mol]]) -> Dict[str, List[float]]:
    """
    批量计算两组分子之间的相似度，包括形状、MCS、描述符和指纹相似度。

    Args:
        mol_list1: 第一个分子列表，可以是 SMILES 字符串或 RDKit 分子对象
        mol_list2: 第二个分子列表，长度需与 mol_list1 相同

    Returns:
        Dict[str, List[float]]: 包含四种相似度分数的字典，每种相似度为一个列表，长度为批量大小
    """
    if len(mol_list1) != len(mol_list2):
        raise ValueError(f"Mismatch in input lengths: {len(mol_list1)} vs {len(mol_list2)}")

    batch_size = len(mol_list1)
    result = {
        'shape_similarity': [0.0] * batch_size,
        'mcs_similarity': [0.0] * batch_size,
        'descriptor_similarity': [0.0] * batch_size,
        'fingerprint_similarity': [0.0] * batch_size
    }

    for i in range(batch_size):
        # 转换为 RDKit 分子对象
        mol1 = mol_list1[i]
        mol2 = mol_list2[i]

        if isinstance(mol1, str):
            mol1 = Chem.MolFromSmiles(mol1)
        if isinstance(mol2, str):
            mol2 = Chem.MolFromSmiles(mol2)

        # 如果任一分子无效，返回 0 分数
        if mol1 is None or mol2 is None:
            continue

        # 计算相似度
        try:
            result['shape_similarity'][i] = shape_similarity(mol1, mol2)
            result['mcs_similarity'][i] = mcs_similarity(mol1, mol2)
            result['descriptor_similarity'][i] = descriptor_similarity(mol1, mol2)
            result['fingerprint_similarity'][i] = fingerprint_similarity(mol1, mol2)
        except Exception as e:
            print(f"Warning: Failed to compute similarity for pair {i}: {e}")
            # 保持默认值 0.0

    return result

# 计算并输出结果
if __name__ == "__main__":
    # 示例分子（SMILES 表示）
    mol1_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    mol2_smiles = "CC(=O)NC1=CC=CC=C1C(=O)O"  # 类似物
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)

    print(f"Molecules: {mol1_smiles} and {mol2_smiles}")
    print(f"1. Fingerprint Similarity (Tanimoto): {fingerprint_similarity(mol1, mol2):.4f}")
    print(f"1. Fingerprint Similarity (Tanimoto): {fingerprint_similarity_1(mol1, mol2):.4f}")
    print(f"2. Descriptor Similarity (Euclidean): {descriptor_similarity(mol1, mol2):.4f}")
    print(f"3. MCS Similarity: {mcs_similarity(mol1, mol2):.4f}")
    print(f"4. Shape Similarity: {shape_similarity(mol1, mol2):.4f}")