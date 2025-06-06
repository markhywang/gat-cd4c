"""Lightweight datasets that return *only* drug or *only* protein graphs.

Used for semi‑supervised batches a la SSM‑DTA.
"""

from torch.utils.data import Dataset


class DrugOnlyDataset(Dataset):
    def __init__(self, paired_ds):
        self.unique_drugs = sorted(set(paired_ds.df['Drug'].tolist()))
        self.paired_ds = paired_ds

    def __len__(self):
        return len(self.unique_drugs)

    def __getitem__(self, idx):
        drug_smiles = self.unique_drugs[idx]
        return self.paired_ds.load_drug(drug_smiles)


class ProtOnlyDataset(Dataset):
    def __init__(self, paired_ds):
        self.unique_prots = sorted(set(paired_ds.df['Target_ID'].tolist()))
        self.paired_ds = paired_ds

    def __len__(self):
        return len(self.unique_prots)

    def __getitem__(self, idx):
        prot_id = self.unique_prots[idx]
        return self.paired_ds.load_protein(prot_id) 