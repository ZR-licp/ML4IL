import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import mordred
from mordred import Calculator, descriptors
import sys

def embed(mol):
    try:
        print(Chem.MolToSmiles(mol))
        mol_with_H = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_H)
        AllChem.MMFFOptimizeMolecule(mol_with_H)
        return mol_with_H
    except Exception as e:
        print(f"Error embedding smiles: {e}")
        return None

def try_embed(mol):
    try:
        return embed(mol)
    except rdkit.Chem.rdchem.AtomValenceException:
        return None

if __name__ == '__main__':
    path = "./dataset/Descriptor_data/"
    mols = pd.read_csv("./data/IL_smiles.csv")  # neg.csv
    mols['rdmol'] = mols['smiles'].map(lambda x: Chem.MolFromSmiles(x))
    mols = mols.drop_duplicates(subset="rdmol")
    
    mols['embedding_success'] = False    
    mols['rdmol_optimized'] = mols['rdmol'].map(try_embed)    
    mols['embedding_success'] = mols['rdmol_optimized'].notnull()    
    mols_success = mols[mols['embedding_success']]
    
    calc = Calculator(descriptors)
    df = calc.pandas(mols_success['rdmol_optimized'])
    df = df.applymap(lambda x: np.nan if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x)
    df = df.dropna(axis=1)
    non_zero_std = df.std()!= 0
    df = df[non_zero_std[non_zero_std].index]
    threshold = 0.9999
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(to_drop, axis=1)
    
    df = df.applymap(lambda x: round(x, 4) if isinstance(x, float) else x)
    
    to_save = mols_success[["smiles"]].join(df)
    
    to_save.to_csv("./data/01Des_IL.csv")
