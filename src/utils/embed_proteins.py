"""
This module embeds proteins from the CD4C dataset.
The embeddings will be eventually be passed into the Graph Attention Network as input.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel
import torchvision
import pandas as pd


class ProteinEmbedder:
    """ This class will embed proteins using a pre-trained autotokenizer

    Instance Attributes:
        - tokenizer: Pre-trained tokenizer for tokenization
        - embedder: Pre-trained model used for embeddings
    """
    tokenizer: AutoTokenizer
    embedder: AutoModel

    def __init__(self) -> None:
        """
        Initializes ProteinEmbedder attributes
        """
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.embedder = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def embed(self, protein_sequence: str) -> torch.Tensor:
        """
        Embeds protein sequence using pre-trained tokenizers and embedders.
        Returns embedding in the form of a PyTorch tensor
        """
        token_input = self.tokenizer(protein_sequence, return_tensors="pt", truncation=False,
                                     add_special_tokens=True, padding=False)
        protein_embedding = self.embedder(**token_input).last_hidden_state.mean(dim=1).squeeze(0)
        return protein_embedding


def embed_proteins(input_filepath: str, output_filepath: str) -> None:
    """
    Reads input protein file and then embeds proteins using ProteinEmbedder instance.
    Furthermore, it prints out the finished embedding
    """
    # Read the protein data into a dataframe.
    protein_df = pd.read_csv(input_filepath)
    # Drop rows with duplicate proteins (only keeping the first row for every protein).
    protein_df = protein_df.drop_duplicates(subset='Target_ID', keep='first')
    protein_df = protein_df.set_index('Target_ID')

    embedding_size = 320
    # Initialize the empty embeddings dataframe.
    embeddings_df = pd.DataFrame(columns=[f'embedding_{x}' for x in range(embedding_size)], index=protein_df.index)
    embeddings_df.index.name = 'Target_ID'

    embedder = ProteinEmbedder()
    for protein_id in protein_df.index:
        protein_sequence = protein_df.at[protein_id, 'protein']
        protein_embedding = embedder.embed(protein_sequence).tolist()
        assert len(protein_embedding) == embedding_size
        embeddings_df.loc[protein_id] = protein_embedding
        print(f'Finished embedding {protein_id}')

    embeddings_df.to_csv(output_filepath)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': [
            'torch',
            'transformers.AutoTokenizer',
            'transformers.AutoModel',
            'torchvision',
            'pandas',
            'os'
        ],
        'disable': ['R0914', 'E1101'],  # R0914 for local variable, E1101 for attributes for imported modules
        'allowed-io': ['embed_proteins'],
        'max-line-length': 120,
    })

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    torchvision.disable_beta_transforms_warning()
    embed_proteins('../../data/filtered_cancer_all.csv', '../../data/protein_embeddings.csv')
