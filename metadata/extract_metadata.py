import tskit
import pandas as pd
import json
import os
import logging
import time
from tqdm import tqdm  # For progress bars

# --- SCRIPT OVERVIEW ---
# This script is designed to efficiently extract and consolidate sample-specific
# metadata from a collection of tskit tree sequence files. For each chromosome's
# tree sequence, it processes every sample node, extracts key topological features
# (like node degree and ancestry depth), and decodes any JSON metadata
# associated with the sample's individual. The results are then compiled into a
# single, comprehensive CSV file.

# --- LOGGING CONFIGURATION ---
# Sets up a logging system that writes messages to both a file and the console.
# This ensures a persistent record of the script's execution, including
# any warnings or errors that may occur.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("metadata_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def extract_chromosome_metadata(trees_path: str) -> pd.DataFrame:
    """
    Efficiently extracts metadata for a single chromosome from a tskit file.

    This function performs the core logic for a single tree sequence. It
    calculates topological properties for each sample node and combines this
    information with any embedded JSON metadata.

    Args:
        trees_path (str): The file path to the tskit tree sequence file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted metadata for the
                      chromosome, or an empty DataFrame on error.
    """
    start_time = time.time()
    # Extract the chromosome number from the filename.
    chrom = os.path.basename(trees_path).split('_chr')[-1].split('.')[0]
    logger.info(f"Starting metadata extraction for chr{chrom}")

    try:
        # Load tree sequence with progress
        logger.info(f"Loading tree sequence: {trees_path}")
        ts = tskit.load(trees_path)
        logger.info(f"Loaded tree sequence with {ts.num_samples} samples")

        # --- EFFICIENT METADATA EXTRACTION ---
        # The following sections precompute topological features over the entire
        # tree sequence to avoid redundant calculations within the main loop.

        # Precompute edge counts, which represent the in-degree of each node.
        # This is a proxy for the number of children a node has in the ARG.
        logger.info("Precomputing edge counts...")
        edge_counts = {}
        for edge in ts.edges():
            edge_counts.setdefault(edge.child, 0)
            edge_counts[edge.child] += 1

        # Precompute ancestry depths, which is the number of edges to the root
        # in the first tree. This provides a measure of how "deep" a sample is.
        logger.info("Precomputing ancestry depths...")
        tree = ts.first()
        depths = {}
        for sample in ts.samples():
            depths[sample] = tree.depth(sample)
            
        # --- ITERATION AND RECORDING ---
        # Iterate through each sample node in the tree sequence and collect all
        # relevant metadata and computed features.
        logger.info("Iterating through samples and compiling records...")
        records = []
        for sample_id in tqdm(ts.samples(), desc=f"Processing chr{chrom}", leave=False):
            node = ts.node(sample_id)
            individual = ts.individual(node.individual) if node.individual != tskit.NULL else None

            record = {
                'chromosome': chrom,
                'sample_id': sample_id,
                'population': ts.population(node.population).metadata['name'] if node.population != tskit.NULL else None,
                'time': node.time,
                'out_degree': edge_counts.get(sample_id, 0),
                'ancestry_depth': depths.get(sample_id)
            }

            # If an individual exists and has metadata, parse and add it to the record.
            if individual and individual.metadata:
                try:
                    # Decode the metadata from bytes and parse it as JSON.
                    meta = json.loads(individual.metadata.decode('utf-8'))
                    record.update(meta)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    # Log a warning for samples with malformed metadata, but continue processing.
                    logger.warning(f"Metadata decode error for sample {sample_id}: {str(e)}")

            records.append(record)

        # --- FINALIZATION ---
        df = pd.DataFrame(records)
        elapsed = time.time() - start_time
        logger.info(f"Completed chr{chrom} in {elapsed:.2f} seconds: {len(df)} records")
        return df

    except Exception as e:
        # Catch any unexpected errors during processing and log them.
        logger.error(f"Error processing {trees_path}: {str(e)}")
        return pd.DataFrame()

# --- MAIN EXECUTION LOGIC ---
# Process all chromosomes
all_metadata = []
logger.info("===== Starting metadata extraction =====")

# Loop through human autosomes (1 to 22).
for chrom in range(1, 23):
    trees_file = f"/home/smondal/ADNI/20250321-ADNI-trees/samples/ADNI.785_chr{chrom}.trees"

    if not os.path.exists(trees_file):
        logger.error(f"File not found: {trees_file}")
        continue

    # Call the main extraction function for each chromosome.
    chrom_meta = extract_chromosome_metadata(trees_file)
    if not chrom_meta.empty:
        all_metadata.append(chrom_meta)
        logger.info(f"Processed chr{chrom}: {len(chrom_meta)} samples")
    else:
        logger.warning(f"No metadata extracted for chr{chrom}")

# --- COMBINE AND SAVE OUTPUT ---
# Concatenate all DataFrames and save the final result to a single CSV.
if all_metadata:
    combined_meta = pd.concat(all_metadata)
    output_path = "/home/smondal/ADNI/vcfs/ADNI_combined_metadata.csv"
    combined_meta.to_csv(output_path, index=False)
    logger.info(f"Saved combined metadata to {output_path}: {len(combined_meta)} records")
else:
    logger.error("No metadata extracted for any chromosome")

logger.info("===== Extraction complete =====")
