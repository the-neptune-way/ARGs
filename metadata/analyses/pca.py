```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import argparse
import os
import logging
import time
import glob
import tskit
import json
from scipy.stats import kruskal
from typing import List, Dict, Tuple
import multiprocessing as mp
from collections import defaultdict

# --- SCRIPT OVERVIEW ---
# This script performs a comprehensive genome-wide analysis by integrating
# two types of data:
# 1. Genome-wide genotype data (from VCFs).
# 2. Topological and temporal metrics from tskit tree sequences.
# The primary goal is to perform PCA on the genotype data and then visualize
# the resulting principal components (PCs) colored by the tree metrics. This
# allows for a direct comparison between genetic ancestry (captured by PCA)
# and genealogical ancestry (captured by tree metrics).
# The script uses multiprocessing to parallelize the extraction of tree metrics
# for each chromosome, significantly speeding up the data preparation phase.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genome_wide_pca.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def extract_tree_metrics(ts_path: str) -> Dict[str, dict]:
    """
    Extracts comprehensive topological and metadata metrics for a chromosome.

    This function processes a single tskit tree sequence file and calculates
    several metrics for each sample, including:
    - Average tree depth
    - In-degree (number of children)
    - Total tree length and number of nodes/edges
    - Population and individual metadata

    Args:
        ts_path (str): The file path to the tskit tree sequence.

    Returns:
        Dict[str, dict]: A dictionary where keys are sample IDs and values are
                         dictionaries of extracted metrics.
    """
    try:
        logger.info(f"Processing tree file: {os.path.basename(ts_path)}")
        ts = tskit.load(ts_path)
        chrom = os.path.basename(ts_path).split('_chr')[-1].split('.')[0]

        metrics = {}

        # Pre-calculate metrics that require iterating over the entire TS once
        # to improve efficiency.
        total_tree_length = ts.overall_tree_length
        num_trees = ts.num_trees
        num_nodes = ts.num_nodes
        num_edges = ts.num_edges

        # Process each sample
        for sample_id in ts.samples():
            # Create a unique key for the sample.
            sample_key = f"tsk_{sample_id}"
            node = ts.node(sample_id)
            individual = ts.individual(node.individual) if node.individual != tskit.NULL else None

            # Initialize the metrics dictionary for this sample.
            metrics[sample_key] = {
                'chromosome': chrom,
                'population': ts.population(node.population).metadata['name'] if node.population != tskit.NULL else None,
                'time': node.time,
                'ancestry_depth_sum': 0.0,
                'in_degree_sum': 0,
                'total_tree_length': total_tree_length,
                'num_trees': num_trees,
                'num_nodes': num_nodes,
                'num_edges': num_edges
            }

            # If individual metadata exists, parse it as JSON and add to metrics.
            if individual and individual.metadata:
                try:
                    meta = json.loads(individual.metadata.decode('utf-8'))
                    metrics[sample_key].update(meta)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Metadata decode error for sample {sample_id}: {str(e)}")
            
            # --- Per-Tree Calculations ---
            # This is a key part: we iterate through all trees in the sequence
            # to calculate average metrics for each sample.
            for tree in ts.trees():
                metrics[sample_key]['ancestry_depth_sum'] += tree.depth(sample_id)
                
                # Calculate in-degree for the current tree.
                # In-degree here refers to the number of children a node has.
                in_degree = 0
                for child in tree.children(sample_id):
                    in_degree += 1
                metrics[sample_key]['in_degree_sum'] += in_degree

        # Finalize average metrics.
        for sample_key in metrics:
            metrics[sample_key]['avg_ancestry_depth'] = metrics[sample_key]['ancestry_depth_sum'] / num_trees
            metrics[sample_key]['avg_in_degree'] = metrics[sample_key]['in_degree_sum'] / num_trees
            del metrics[sample_key]['ancestry_depth_sum']
            del metrics[sample_key]['in_degree_sum']

        logger.info(f"Finished processing {os.path.basename(ts_path)}")
        return metrics
    except Exception as e:
        logger.error(f"Error processing {ts_path}: {e}")
        return {}

def aggregate_metrics(metrics_list: List[Dict[str, dict]]) -> pd.DataFrame:
    """
    Aggregates metrics from all chromosomes into a single pandas DataFrame.

    Args:
        metrics_list (List[Dict[str, dict]]): A list of dictionaries, where
                                              each dictionary contains metrics
                                              for a single chromosome.

    Returns:
        pd.DataFrame: A DataFrame with combined, genome-wide metrics for each sample.
    """
    # Use a defaultdict to aggregate metrics by sample ID.
    aggregated = defaultdict(lambda: defaultdict(float))
    
    for chrom_metrics in metrics_list:
        for sample_key, metrics in chrom_metrics.items():
            # Sum up average metrics across chromosomes to get a genome-wide average.
            for key, value in metrics.items():
                if key in ['ancestry_depth_sum', 'in_degree_sum', 'avg_ancestry_depth', 'avg_in_degree']:
                    aggregated[sample_key][key] += value
                elif key not in ['chromosome', 'population', 'time', 'total_tree_length', 'num_trees', 'num_nodes', 'num_edges']:
                    # For non-numeric or unique metadata, just assign it.
                    aggregated[sample_key][key] = value
                
    # Normalize the aggregated metrics by the number of chromosomes.
    num_chroms = len(metrics_list)
    for sample_key in aggregated:
        if 'avg_ancestry_depth' in aggregated[sample_key]:
            aggregated[sample_key]['avg_ancestry_depth'] /= num_chroms
        if 'avg_in_degree' in aggregated[sample_key]:
            aggregated[sample_key]['avg_in_degree'] /= num_chroms

    # Create the final DataFrame from the aggregated data.
    metadata_df = pd.DataFrame.from_dict(aggregated, orient='index')
    metadata_df.index.name = 'sample_id'
    return metadata_df

def load_genotypes(vcf_files: List[str]) -> pd.DataFrame:
    """
    Loads genotype data from multiple VCF files into a single DataFrame.

    This function reads VCF files, extracts the genotype data (GT), and
    organizes it into a single DataFrame where rows are samples and
    columns are sites (SNPs).

    Args:
        vcf_files (List[str]): A list of paths to the VCF files.

    Returns:
        pd.DataFrame: A DataFrame of genotype data.
    """
    all_genotypes = []
    
    for vcf_file in tqdm(vcf_files, desc="Loading VCFs"):
        try:
            # Use `tskit.vcf_to_ts` with an in-memory ts object
            # to efficiently load genotype data without saving to disk.
            with open(vcf_file, "r") as f:
                ts = tskit.vcf_to_ts(f)

            # Extract genotype matrix.
            genotype_matrix = ts.genotype_matrix()
            sample_ids = [f"tsk_{s}" for s in ts.samples()]
            
            # Transpose to get samples as rows and sites as columns.
            df = pd.DataFrame(genotype_matrix.T, index=sample_ids)
            all_genotypes.append(df)
            
        except Exception as e:
            logger.error(f"Error loading VCF file {vcf_file}: {e}")
            continue

    if not all_genotypes:
        raise ValueError("No genotype data could be loaded from VCF files.")
    
    # Concatenate all chromosome dataframes along the columns (sites).
    genotypes = pd.concat(all_genotypes, axis=1)
    return genotypes

def preprocess_genotypes(df: pd.DataFrame, maf_threshold: float, missing_threshold: float) -> pd.DataFrame:
    """
    Preprocesses genotype data by filtering and imputing.

    Args:
        df (pd.DataFrame): The raw genotype DataFrame.
        maf_threshold (float): The minimum Minor Allele Frequency.
        missing_threshold (float): The maximum allowed fraction of missing
                                   data per site.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logger.info("Preprocessing genotypes...")
    
    # 1. Filter by Minor Allele Frequency (MAF).
    maf = df.sum(axis=0) / (2 * df.shape[0])
    valid_sites_maf = (maf >= maf_threshold) & (maf <= 1 - maf_threshold)
    df = df.loc[:, valid_sites_maf]
    logger.info(f"Filtered to {df.shape[1]} sites with MAF >= {maf_threshold}.")

    # 2. Filter by missingness.
    missing_fraction = df.isnull().sum() / df.shape[0]
    valid_sites_missing = missing_fraction <= missing_threshold
    df = df.loc[:, valid_sites_missing]
    logger.info(f"Filtered to {df.shape[1]} sites with missingness <= {missing_threshold}.")
    
    # 3. Impute missing values.
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    logger.info("Missing values imputed using median strategy.")
    
    # 4. Standardize data (center and scale).
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)
    logger.info("Genotype data standardized.")
    
    return df_scaled

def run_pca(df: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Principal Component Analysis (PCA) on the genotype data.

    Args:
        df (pd.DataFrame): The preprocessed genotype DataFrame.
        n_components (int): The number of principal components to compute.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the PCA coordinates
                                           and the feature loadings.
    """
    logger.info(f"Running PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(df)
    
    # Store results in a DataFrame for easy visualization.
    pca_df = pd.DataFrame(
        pca_coords,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Store loadings (the contribution of each SNP to each PC).
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_df, loadings

def visualize_results(pca_coords: pd.DataFrame, metadata: pd.DataFrame, output_dir: str, prefix: str):
    """
    Visualizes PCA results and links them to tskit-derived metadata.

    Args:
        pca_coords (pd.DataFrame): The DataFrame of PCA coordinates.
        metadata (pd.DataFrame): The DataFrame of aggregated tree metrics.
        output_dir (str): The directory to save the plots.
        prefix (str): A prefix for output filenames.

    Returns:
        pd.DataFrame: A final DataFrame containing PCA coordinates and metadata.
    """
    logger.info("Visualizing results...")
    
    # Combine PCA coordinates with the metadata.
    results_df = pca_coords.join(metadata, how='inner')
    
    # Define a list of metrics to visualize on the PCA plot.
    metrics_to_plot = ['avg_ancestry_depth', 'avg_in_degree', 'population', 'time']
    
    # Create plots for each metric.
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 8))
        
        # Plot PC1 vs PC2, colored by the current metric.
        if metric in ['population']:
            sns.scatterplot(
                x='PC1', y='PC2', hue=metric, data=results_df, s=50, alpha=0.7,
                palette='tab20' if len(results_df[metric].unique()) <= 20 else 'viridis'
            )
        else:
            sns.scatterplot(x='PC1', y='PC2', hue=metric, data=results_df, s=50, alpha=0.7, palette='viridis')

        plt.title(f"{prefix} PCA (PC1 vs PC2) colored by {metric}")
        plt.xlabel(f"PC1 ({pca_coords.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({pca_coords.explained_variance_ratio_[1]*100:.2f}%)")
        plt.legend(title=metric, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_PCA_PC1_PC2_{metric}.png"))
        plt.close()
        
    return results_df

def main():
    """
    Main function to run the entire genome-wide analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Genome-wide PCA and tskit metric analysis.")
    parser.add_argument("--ts-dir", required=True, help="Directory containing tskit tree sequence files.")
    parser.add_argument("--ts-pattern", default="ADNI.785_chr*.trees", help="File pattern for tskit files.")
    parser.add_argument("--vcf-dir", required=True, help="Directory containing VCF files.")
    parser.add_argument("--vcf-pattern", default="ADNI.785_chr*.vcf", help="File pattern for VCF files.")
    parser.add_argument("--output", default=".", help="Output directory for plots and data.")
    parser.add_argument("--prefix", default="GW_Analysis", help="Prefix for output filenames.")
    parser.add_argument("--components", type=int, default=10, help="Number of PCA components to compute.")
    parser.add_argument("--maf", type=float, default=0.05, help="Minor Allele Frequency threshold.")
    parser.add_argument("--missing", type=float, default=0.05, help="Missing data threshold per site.")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads for parallel processing.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    os.makedirs(args.output, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # =================================================================
        # Step 1: Load and process tree metrics in parallel
        # =================================================================
        ts_files = glob.glob(os.path.join(args.ts_dir, args.ts_pattern))
        ts_files.sort(key=lambda x: int(os.path.basename(x).split('chr')[-1].split('.')[0]))
        
        if not ts_files:
            raise ValueError(f"No tskit files found in {args.ts_dir}")
            
        logger.info(f"Found {len(ts_files)} tskit files for metric extraction")
        
        # Use a multiprocessing pool to run `extract_tree_metrics` for each file in parallel.
        with mp.Pool(processes=args.threads) as pool:
            metrics_list = pool.map(extract_tree_metrics, ts_files)
            
        # Aggregate the results from all parallel processes.
        metadata = aggregate_metrics(metrics_list)

        # =================================================================
        # Step 2: Load and process genotype data
        # =================================================================
        vcf_files = glob.glob(os.path.join(args.vcf_dir, args.vcf_pattern))
        vcf_files.sort(key=lambda x: int(os.path.basename(x).split('chr')[-1].split('.')[0]))

        if not vcf_files:
            raise ValueError(f"No VCF files found in {args.vcf_dir}")

        logger.info(f"Found {len(vcf_files)} VCF files for genotype loading")
        genotypes = load_genotypes(vcf_files)

        # Preprocess genotypes
        gt_processed = preprocess_genotypes(
            genotypes,
            maf_threshold=args.maf,
            missing_threshold=args.missing
        )

        # =================================================================
        # Step 3: Perform PCA
        # =================================================================
        pca_coords, loadings = run_pca(gt_processed, n_components=args.components)

        # =================================================================
        # Step 4: Visualize results with tree metrics
        # =================================================================
        results_df = visualize_results(pca_coords, metadata, args.output, prefix=args.prefix)
        
        # Save the final combined DataFrame for later use.
        results_df.to_csv(os.path.join(args.output, f"{args.prefix}_PCA_results.csv"))
        logger.info(f"Saved combined PCA and metadata to {os.path.join(args.output, f'{args.prefix}_PCA_results.csv')}")

        # =================================================================
        # Step 5: Ancestry path analysis
        # (This is implicitly done in the visualization step by plotting
        # tree metrics. Further statistical tests could be added here if needed.)
        # =================================================================

        elapsed = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed:.2f} seconds ({elapsed/3600:.2f} hours)")

    except Exception as e:
        logger.exception("Fatal error in analysis pipeline")
        # Reraise the exception after logging for debugging purposes.
        raise

if __name__ == "__main__":
    main()
```
