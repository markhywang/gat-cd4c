import pandas as pd
import sys

def reformat_embeddings_csv(input_csv_path, output_csv_path, protein_id_col='Target_ID'):
    """
    Reformats a CSV with embeddings spread across multiple columns (emb_0, emb_1, ...)
    into a CSV with two columns: 'protein_id' and 'embedding' (string list).
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {input_csv_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading CSV {input_csv_path}: {e}", file=sys.stderr)
        return False

    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    if not embedding_cols:
        print(f"Error: No columns starting with 'emb_' found in {input_csv_path}. Check your CSV column names.", file=sys.stderr)
        return False

    if protein_id_col not in df.columns:
        print(f"Error: Protein ID column '{protein_id_col}' not found in {input_csv_path}.", file=sys.stderr)
        # Try to find a 'protein_id' column as an alternative
        if 'protein_id' in df.columns:
            print("Found 'protein_id' column, using that instead.", file=sys.stderr)
            protein_id_col = 'protein_id'
        else:
            return False
            
    actual_embedding_dim = len(embedding_cols)
    print(f"Detected {actual_embedding_dim} embedding columns (e.g., {embedding_cols[0]} to {embedding_cols[-1]}).")

    formatted_data = []
    for _, row in df.iterrows():
        protein_id = row[protein_id_col]
        try:
            embedding_values = [float(row[col]) for col in embedding_cols]
        except ValueError as e:
            print(f"Warning: Could not convert embedding value to float for protein {protein_id}. Error: {e}. Skipping row.", file=sys.stderr)
            continue
            
        embedding_str = str(embedding_values)
        formatted_data.append({'protein_id': protein_id, 'embedding': embedding_str})

    if not formatted_data:
        print("Error: No data was successfully reformatted. Output file will not be created.", file=sys.stderr)
        return False

    reformatted_df = pd.DataFrame(formatted_data)

    try:
        reformatted_df.to_csv(output_csv_path, index=False)
        print(f"Successfully reformatted CSV saved to: {output_csv_path}")
        print(f"Detected embedding dimension: {actual_embedding_dim}")
        print(f"Please ensure 'protein_embedding_dim' in your notebook args matches this value.")
        print(f"Also, update 'protein_embeddings_file' in your notebook to point to '{output_csv_path}'.")
        return True
    except Exception as e:
        print(f"Error saving reformatted CSV to {output_csv_path}: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python format_protein_embeddings.py <input_csv_path> <output_csv_path> [protein_id_column_name]")
        print("Example: python format_protein_embeddings.py ../data/old_embeddings.csv ../data/new_embeddings.csv Target_ID")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pid_col = 'Target_ID' # Default
    if len(sys.argv) > 3:
        pid_col = sys.argv[3]
        
    reformat_embeddings_csv(input_path, output_path, protein_id_col=pid_col) 