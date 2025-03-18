import torch
from transformers import AutoTokenizer, AutoModel
import torchvision
import pandas as pd
import os


class ProteinEmbedder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.embedder = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def embed(self, protein_sequence: str) -> torch.Tensor:
        token_input = self.tokenizer(protein_sequence, return_tensors="pt", truncation=False,
                                     add_special_tokens=True, padding=False)
        protein_embedding = self.embedder(**token_input).last_hidden_state.mean(dim=1).squeeze(0)
        return protein_embedding


def embed_proteins(input_filepath: str, output_filepath: str) -> None:
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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    torchvision.disable_beta_transforms_warning()
    embed_proteins('../assets/cancer_all.csv', '../assets/protein_embeddings.csv')
