#!/usr/bin/env python3
"""
Parallel Chromosome Embedding Generator

This script is a high-performance Python pipeline designed to generate node embeddings
for individual trees within tskit tree-sequence files. It processes multiple
chromosomes in parallel, converting each tree into a graph, and then applying the
Node2Vec algorithm to learn low-dimensional vector representations of the nodes.

Pipeline Overview:

The pipeline operates in three main stages:

1.  Parallel Chromosome Processing: The main function (`main`) uses a
    `ProcessPoolExecutor` to initiate a separate process for each specified
    chromosome. This allows for concurrent processing of multiple chromosomes,
    leveraging multi-core systems.

2.  Tree-by-Tree Conversion: Each chromosome-specific worker process
    (`process_chromosome`) iterates through the trees within a tskit
    tree-sequence file. For each tree, it performs the following steps:
    * Converts the tskit `Tree` object into a NetworkX graph.
    * Generates a `Node2Vec` model.
    * Generates node embeddings for the current tree.
    * Saves the embeddings to a NumPy file.

3.  Embedding Generation: The `generate_embedding` function performs the core
    embedding task. It converts the tree to a NetworkX graph, then uses the
    `node2vec` library to generate random walks and train a Word2Vec model to
    produce the final node embeddings.

Technical Details:

1.  Multiprocessing and Resource Management:
    * `multiprocessing.set_start_method('forkserver', force=True)`: This is
        a critical technical detail for resource management. The `'forkserver'`
        start method is used to mitigate potential issues with process state
        inheritance (e.g., deadlocks caused by a forked child inheriting locks
        from a multi-threaded parent). It starts a dedicated "fork server" process,
        which is clean and single-threaded. When a new child process is needed,
        the main process requests it from the fork server. This method is generally
        safer and more robust than `'fork'` for complex applications.
    * `concurrent.futures.ProcessPoolExecutor`: This high-level API is used to
        manage the pool of worker processes for parallel chromosome processing.
    * `max_workers=args.max_concurrent`: The number of parallel chromosome processes
        is a configurable command-line argument, allowing the user to tune performance
        based on available system resources (e.g., memory and CPU cores).
    * `as_completed()`: This function is used to iterate over the results of the
        submitted tasks as they complete, rather than waiting for all tasks to finish.
        This allows for real-time logging and monitoring of chromosome-specific progress.

2.  Tree-to-Graph Conversion:
    * `tskit.Tree` to `networkx.Graph`: The script's core functionality relies on
        converting the tskit tree structure into a format compatible with the
        `node2vec` library. This is achieved by creating a `networkx.Graph` object
        and adding nodes and edges from the tskit tree.
    * `tskit.Tree.nodes()`: This method iterates through all nodes in the tree.
    * `tree.parent(node)`: This is used to find the parent of each node, which
        is necessary to define the edges of the graph. The edges are directed
        from parent to child, representing the flow of genetic ancestry.
    * `nx.DiGraph()`: A directed graph is used to represent the hierarchical
        structure of the tree.

3.  Node2Vec Algorithm Parameters:
    The Node2Vec model is configured with a specific set of parameters that control
    the random walk and embedding generation process:
    * `dimensions`: The dimensionality of the resulting node embedding vectors.
    * `walk_length`: The number of nodes in each random walk. Longer walks capture
        more global graph structure, while shorter walks focus on local neighborhoods.
    * `num_walks`: The number of random walks to generate starting from each node.
        A higher value leads to a more robust representation of the node's position
        in the graph.
    * `window`: The size of the context window for the Skip-Gram model (the
        underlying model for Node2Vec). It defines how many neighboring nodes are
        considered when learning a node's embedding.
    * `min_count`: The minimum frequency of a "word" (node) to be included in the
        vocabulary.
    * `workers`: The number of threads used by the Node2Vec algorithm for parallel
        random walk generation and model training within a single chromosome process.
        This is dynamically calculated based on available CPU cores to optimize
        performance within each parallel job.

4.  Dynamic Resource Allocation:
    * `get_optimal_workers(num_cores, num_parallel_jobs)`: This function dynamically
        determines the number of threads for the Node2Vec model. It divides the total
        available logical CPU cores by the number of concurrent chromosome jobs,
        ensuring that the total CPU usage does not exceed the system's capacity.
        This prevents over-subscription of CPU resources, which can lead to
        performance degradation.

5.  Data I/O and Persistence:
    * `numpy.save(..., allow_pickle=False)`: The embeddings are saved in the
        NumPy `.npy` format. Using `allow_pickle=False` is a security measure and
        ensures the files are a simple array dump, which is efficient and safe.
    * `os.path.join(...)`: All file paths are constructed using this method to
        ensure compatibility across different operating systems.
    * Directory Management: The script automatically creates output and log
        directories if they do not exist, ensuring a clean and robust execution
        environment.

6.  Logging and Progress Monitoring:
    * `logging` module: A custom logging system is configured to provide detailed
        information about the script's execution.
    * `logging.handlers.RotatingFileHandler`: This is used to manage log file
        sizes, automatically rotating and compressing logs when they exceed a
        certain size.
    * The logging system provides timestamps and separates log messages into
        standard output and detailed file logs.
    * Child processes are configured to use a separate log file, preventing
        race conditions and ensuring that logs from different parallel jobs are
        not intertwined.
    * Final Summary: Upon completion, the script reports key performance metrics
        such as total execution time, processing rate (trees/hour), and a
        breakdown of successful and failed chromosomes. This provides a clear
        overview of the pipeline's performance.
"""

import os
import sys
import time
import logging
import logging.handlers
import argparse
import psutil
import tskit
import networkx as nx
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from node2vec import Node2Vec
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
# Path to the directory containing the input tskit tree-sequence files.
INPUT_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/samples"

# Path to the directory where the output embedding files will be saved.
OUTPUT_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/embeddings/per_tree/per_tree_2"

# Path to the directory where log files will be stored.
LOG_DIR = "/home/smondal/logs"

# Embedding parameters for the Node2Vec model.
EMBEDDING_DIM = 128
WALK_LENGTH = 80
NUM_WALKS = 50
WINDOW_SIZE = 5
MIN_COUNT = 1
BATCH_WORDS = 4

# Processing parameters to control the pipeline's execution.
TREES_PER_LOG = 100
MIN_TREE_NODES = 2
MAX_CONCURRENT_CHROMOSOMES = 4  # Default max parallel chromosomes
MIN_WORKERS_PER_CHROM = 2       # Minimum workers per chromosome for node2vec

# =============================================================================
# Functions
# =============================================================================

def setup_logging(name_suffix=""):
    """
    Configures a robust logging system with file rotation.

    Args:
        name_suffix (str): An optional suffix for the log file name.

    Returns:
        tuple: The logger object and the path to the log file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(LOG_DIR, f"embeddings_{timestamp}{name_suffix}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler with rotation
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, log_file

def get_optimal_workers(num_cores, num_parallel_jobs):
    """
    Calculates the optimal number of workers for a single job to avoid
    over-subscribing CPU cores.

    Args:
        num_cores (int): Total number of logical CPU cores available.
        num_parallel_jobs (int): The number of concurrent processes.

    Returns:
        int: The number of workers to use per job.
    """
    if num_parallel_jobs == 0:
        return num_cores
    workers = max(MIN_WORKERS_PER_CHROM, int(num_cores / num_parallel_jobs))
    return workers

def generate_embedding(tree_id, tree_obj, chrom, workers):
    """
    Generates Node2Vec embeddings for a single tskit tree.

    Args:
        tree_id (int): The ID of the tree within the tree sequence.
        tree_obj (tskit.Tree): The tskit tree object.
        chrom (int): The chromosome number.
        workers (int): The number of workers for the Node2Vec algorithm.

    Returns:
        bool: True if embedding was successful, False otherwise.
    """
    # Check if the tree is large enough to be processed.
    if tree_obj.num_nodes <= MIN_TREE_NODES:
        return False
    
    # Convert tskit tree to a NetworkX directed graph.
    # We use a directed graph to represent the parent-child relationships.
    try:
        G = nx.DiGraph()
        for node in tree_obj.nodes():
            parent = tree_obj.parent(node)
            if parent != tskit.NULL:
                G.add_edge(parent, node)
        
        # Check if the graph is valid for Node2Vec.
        if len(G.nodes()) <= MIN_TREE_NODES:
            return False
    except Exception as e:
        logging.error(f"Error converting tree to graph for chr {chrom}, tree {tree_id}: {e}")
        return False

    # Configure and run the Node2Vec model.
    try:
        node2vec = Node2Vec(
            G,
            dimensions=EMBEDDING_DIM,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            workers=workers,
        )
        # Train the Word2Vec model with Skip-Gram architecture.
        model = node2vec.fit(
            window=WINDOW_SIZE,
            min_count=MIN_COUNT,
            batch_words=BATCH_WORDS
        )
        
        # Get the embedding vectors and save them.
        embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
        output_file_path = os.path.join(OUTPUT_DIR, f"chr_{chrom}_tree_{tree_id}_embeddings.npy")
        np.save(output_file_path, embeddings, allow_pickle=False)
        return True
    except Exception as e:
        logging.error(f"Error generating embedding for chr {chrom}, tree {tree_id}: {e}")
        return False

def process_chromosome(chrom, num_parallel_jobs):
    """
    Worker function to process all trees within a single chromosome's tree sequence file.

    Args:
        chrom (int): The chromosome number.
        num_parallel_jobs (int): The total number of parallel chromosome jobs.

    Returns:
        tuple: (chromosome number, success status, number of trees processed)
    """
    # Set up child process-specific logging.
    _, child_log_file = setup_logging(f"_chrom_{chrom}")
    
    # Calculate the number of threads for Node2Vec based on available cores and
    # the number of parallel chromosome jobs.
    total_cores = psutil.cpu_count(logical=True)
    n2v_workers = get_optimal_workers(total_cores, num_parallel_jobs)
    logging.info(f"Child process for chr {chrom} started with {n2v_workers} Node2Vec workers.")

    trees_file = os.path.join(INPUT_DIR, f"ADNI.785_chr{chrom}.trees")
    trees_processed_count = 0
    start_time = time.time()
    
    try:
        ts = tskit.load(trees_file)
        logging.info(f"Loaded chr {chrom}: {ts.num_trees} trees, {ts.num_samples} samples, {ts.num_sites} sites.")

        for tree_id, tree in enumerate(ts.trees()):
            success = generate_embedding(tree_id, tree, chrom, n2v_workers)
            if success:
                trees_processed_count += 1
            
            # Log progress periodically to track long-running jobs.
            if (tree_id + 1) % TREES_PER_LOG == 0:
                elapsed_time = time.time() - start_time
                logging.info(f"Chr {chrom}: Processed {tree_id + 1} trees in {elapsed_time:.1f}s.")
        
        logging.info(f"Chr {chrom} completed. Total trees processed: {trees_processed_count}")
        return chrom, True, trees_processed_count

    except Exception as e:
        logging.error(f"Failed to process chromosome {chrom}. Error: {e}", exc_info=True)
        return chrom, False, trees_processed_count
    
# =============================================================================
# Main Execution Block
# =============================================================================
def main():
    """
    Main function to parse arguments, set up the pipeline, and orchestrate
    parallel chromosome processing.
    """
    parser = argparse.ArgumentParser(description="Generate Node2Vec embeddings for tskit trees in parallel.")
    parser.add_argument("--chromosomes", nargs='+', type=int, default=list(range(1, 23)),
                        help="List of chromosomes to process. Defaults to 1-22.")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_CHROMOSOMES,
                        help=f"Maximum number of chromosomes to process in parallel. Defaults to {MAX_CONCURRENT_CHROMOSOMES}.")
    args = parser.parse_args()

    # Set up main logging.
    logger, main_log = setup_logging()
    logging.info("Starting Parallel Chromosome Embedding Generator.")
    logging.info(f"Input Directory: {INPUT_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Chromosomes to process: {args.chromosomes}")
    logging.info(f"Max concurrent jobs: {args.max_concurrent}")

    # Ensure output directory exists before starting jobs.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    successful_chromosomes = []
    failed_chromosomes = []
    total_trees_processed = 0

    try:
        # Create a process pool with a fixed number of workers.
        with ProcessPoolExecutor(max_workers=args.max_concurrent) as executor:
            # Submit a job for each chromosome to the executor.
            future_to_chrom = {
                executor.submit(process_chromosome, chrom, args.max_concurrent): chrom
                for chrom in args.chromosomes
            }
            # Iterate over completed jobs as they finish.
            for future in as_completed(future_to_chrom):
                chrom_num = future_to_chrom[future]
                try:
                    chrom_num, success, trees_processed = future.result()
                    if success:
                        successful_chromosomes.append(chrom_num)
                        total_trees_processed += trees_processed
                        logging.info(f" Chromosome {chrom_num} completed successfully")
                    else:
                        failed_chromosomes.append(chrom_num)
                        logging.error(f" Chromosome {chrom_num} processing failed")
                except Exception as e:
                    logging.error(f" An unexpected error occurred for chromosome {chrom_num}: {e}")
                    failed_chromosomes.append(chrom_num)

        total_time = time.time() - start_time
        trees_per_hour = total_trees_processed / (total_time / 3600) if total_time > 0 else 0
        logging.info("\n" + "="*50)
        logging.info("🏁 PROCESSING COMPLETE")
        logging.info("="*50)
        logging.info(f"Successful chromosomes: {sorted(successful_chromosomes)}")
        logging.info(f"Failed chromosomes: {sorted(failed_chromosomes)}")
        logging.info(f"Total trees processed: {total_trees_processed}")
        logging.info(f"Total time: {total_time/3600:.2f} hours")
        logging.info(f"Processing rate: {trees_per_hour:.1f} trees/hour")
        logging.info(f"Main log: {main_log}")

        if failed_chromosomes:
            logging.error(f"Failed chromosomes: {sorted(failed_chromosomes)}")
            sys.exit(1)

    except Exception as e:
        logging.critical(f" Fatal error in main process: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logging.info("Pipeline terminated")

if __name__ == "__main__":
    try:
        # Set the multiprocessing start method for stability on certain OS's.
        multiprocessing.set_start_method('forkserver', force=True)
    except (RuntimeError, AttributeError):
        logging.info("Multiprocessing start method already set or not available on this OS.")
    main()
