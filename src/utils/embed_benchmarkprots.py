from tdc.multi_pred import DTI

def load_tdc_dti(name: str, seed: int = 42, frac=[0.8,0.1,0.1]):
    # name should be 'KIBA' or 'DAVIS'
    data = DTI(name=name)
    split = data.get_split(method='random', seed=seed, frac=frac)
    train_df, val_df, test_df = split['train'], split['valid'], split['test']

    for df in (train_df, val_df, test_df):
        df.rename(columns={
            'Drug': 'smiles',
            'Target': 'Target_ID',
            'Y':     'pChEMBL_Value'
        }, inplace=True)

    # e.g., a small script to build graphs for all unique Target_IDs in train+val+test
    from utils.embed_proteins import ProteinGraphBuilder
    import pandas as pd

    for name in ['KIBA','DAVIS']:
        data = DTI(name=name)
        all_ids = set(pd.concat([data.get_split(method='random', seed=42, frac=[.8,.1,.1])[k] 
                                for k in ['train','valid','test']])['Target'])
        builder = ProteinGraphBuilder(graph_dir='data/protein_graphs')
        for pid in all_ids:
            pdb = builder.download_af_pdb(pid)    # or your own PDB paths
            graph = builder.build(pdb)           # 24-D one-hot + coords graph
            builder.save(pid, graph)


    return train_df, val_df, test_df

    
