#!/usr/bin/env python3
"""
Link node2vec embeddings with genomic mutations from a tskit TreeSequence.

This script reads a tree sequence file and a directory of pre-computed
node embeddings. It iterates through each tree and its mutations, looks up
the corresponding embedding for the node on which a mutation occurred, and
saves the combined information into a single CSV file.

This process is crucial for downstream machine learning tasks where you want to
predict a mutation's properties (e.g., its effect, pathogenicity, or frequency)
based on the topological and temporal context of its node in the ancestral
recombination graph (ARG). The node embedding serves as this learned context
vector.

Required packages: tskit, numpy, pandas
Install them with: pip install tskit numpy pandas
"""

import os
import tskit
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm


# --- CONFIGURATION ---
# Define the paths for input and output files.
# The tskit tree sequence file containing all trees and mutations.
TS_FILE_PATH = "/home/smondal/ADNI/20250321-ADNI-trees/samples/ADNI.785_chr19.trees"

# The directory where the per-tree node embedding files (.npy) are stored.
# The script expects files named like 'chr_19_tree_0_embeddings.npy',
# 'chr_19_tree_1_embeddings.npy', etc.
EMBEDDINGS_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/embeddings/per_tree/chr_19__2/chr19"

# The final output CSV file where the linked data will be saved.
OUTPUT_CSV_PATH = "/home/smondal/ADNI/linked_embeddings_chr19.csv"



def link_embeddings_to_mutations():
    """
    Loads node embeddings and links them to specific mutations from a tree sequence.

    This is the main function that orchestrates the entire linking process.
    It performs path validation, loads the data, iterates through trees and mutations,
    collects linked data, and saves the final result to a CSV file.
    """
    # --- Setup and Validation ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Basic sanity checks to ensure all required files and directories exist.
    if not os.path.exists(TS_FILE_PATH):
        logging.error(f"FATAL: Tree sequence file not found at: {TS_FILE_PATH}")
        return

    if not os.path.isdir(EMBEDDINGS_DIR):
        logging.error(f"FATAL: Embeddings directory not found at: {EMBEDDINGS_DIR}")
        return

    logging.info(f"Loading tree sequence from: {TS_FILE_PATH}")
    try:
        ts = tskit.load(TS_FILE_PATH)
        logging.info(f"Successfully loaded tree sequence with {ts.num_trees} trees.")
    except Exception as e:
        logging.error(f"FATAL: Failed to load tree sequence. Error: {e}")
        return

    # A list to store dictionaries of linked data before converting to a DataFrame.
    linked_data = []

    # --- Main Processing Loop ---
    # `tqdm` provides a progress bar for the loop, which is useful for
    # long-running operations.
    for tree_index, tree in tqdm(enumerate(ts.trees()), total=ts.num_trees, desc="Processing trees"):
        # Construct the file path for the embeddings corresponding to the current tree.
        embeddings_file = os.path.join(EMBEDDINGS_DIR, f"chr_19_tree_{tree_index}_embeddings.npy")
        
        # Load the embeddings for the current tree.
        # This is a critical step; if the file doesn't exist, we skip to the next tree.
        if not os.path.exists(embeddings_file):
            continue

        try:
            # The .npy file contains a 2D numpy array where each row is the
            # embedding for a node. The row index corresponds to the tskit node ID.
            embeddings_array = np.load(embeddings_file)
            
            # Create a dictionary for fast lookup of embeddings by node ID.
            # The keys are string representations of node IDs for consistent hashing.
            embeddings = {str(i): embeddings_array[i] for i in range(len(embeddings_array))}

        except Exception as e:
            logging.warning(f"Skipping tree {tree_index} due to loading error: {e}")
            continue

        # Iterate through all mutations in the current tree.
        for mutation in tree.mutations():
            node_id_str = str(mutation.node)

            # Check if an embedding exists for the node this mutation is on.
            # This ensures that we only link mutations to nodes that were
            # included in the embedding generation process.
            if node_id_str in embeddings:
                site = ts.site(mutation.site)

                # We found a match! Collect all the data into a dictionary.
                linked_data.append({
                    'tree_index': tree_index,
                    'genomic_position': int(site.position),
                    'node_id': mutation.node,
                    'node_time': ts.node(mutation.node).time,
                    'ancestral_state': site.ancestral_state,
                    'derived_state': mutation.derived_state,
                    # The embedding vector is stored as a single item in the dictionary.
                    'embedding': embeddings[node_id_str]
                })

    if not linked_data:
        logging.error("Processing finished, but no linked mutations were found. Please double-check your paths and ensure the embedding files are not empty.")
        return

    # --- Save Output ---
    logging.info(f"Found {len(linked_data)} linked mutation-embedding pairs. Creating DataFrame...")

    # Convert the list of dictionaries to a Pandas DataFrame.
    df = pd.DataFrame(linked_data)

    # Explode the embedding vector into separate columns for easier use in ML frameworks.
    # This turns the 'embedding' column, which contains NumPy arrays, into
    # a set of new columns named 'emb_0', 'emb_1', ..., 'emb_127'.
    embedding_dims = df['embedding'].apply(pd.Series)
    embedding_dims = embedding_dims.rename(columns = lambda x : 'emb_' + str(x))

    # Join the new embedding columns with the original DataFrame and drop the old
    # list-like 'embedding' column.
    df_final = pd.concat([df.drop(['embedding'], axis=1), embedding_dims], axis=1)

    logging.info(f"Saving final linked data with expanded embedding columns to {OUTPUT_CSV_PATH}...")
    # Save the final DataFrame to a CSV file. `index=False` prevents Pandas from
    # writing the DataFrame index to the file.
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)

    logging.info(f"✅ Success! Analysis complete. Output saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    link_embeddings_to_mutations()
