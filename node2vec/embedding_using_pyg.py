#!/usr/bin/env python3
"""
GPU-Accelerated Parallel Chromosome Embedding Generator using PyTorch Geometric.

This script leverages the PyTorch Geometric library to perform Node2Vec
embeddings on tskit tree sequence data. It is designed to take advantage
of GPU hardware for significant performance acceleration over traditional
CPU-based methods.

The pipeline processes multiple chromosomes in parallel, with each child
process converting trees from a single chromosome's tree sequence into a
graph representation and then generating node embeddings using a GPU-accelerated
Node2Vec implementation.

### Pipeline Overview

1.  **Parallel Chromosome Processing**: The main function (`main`) uses a
    `ProcessPoolExecutor` to initiate a separate process for each specified
    chromosome. This allows for concurrent processing of multiple chromosomes,
    optimally utilizing system resources.

2.  **Tree-to-Graph Conversion**: The `generate_embedding` function iterates
    through trees from the input `tskit` tree sequence. Each tree is converted
    into a `torch_geometric.data.Data` object, which is the native graph data
    structure for PyTorch Geometric.

3.  **GPU-Accelerated Node2Vec**: The core of the script is the
    `torch_geometric.nn.Node2Vec` class. This implementation is highly optimized
    and can perform random walk generation and the subsequent Word2Vec training
    on a CUDA-enabled GPU. This is particularly effective for large graphs and
    complex embedding tasks, as it offloads the computationally intensive parts
    of the algorithm from the CPU to the GPU.

### Technical Details

#### 1. GPU Acceleration with PyTorch and CUDA

* **`torch.device`**: The script dynamically checks for the availability of a
    CUDA-enabled GPU (`torch.cuda.is_available()`). If a GPU is found, it sets the
    device to `'cuda'`, otherwise, it defaults to `'cpu'`. This ensures the script
    is portable and can run on systems with or without a GPU.
* **Data Transfer**: The graph data (`Data` object) is explicitly moved to the
    selected device (`data.to(device)`) before the Node2Vec model is initialized
    and trained. This is a crucial step to enable GPU-accelerated computations.
* **`torch_geometric.nn.Node2Vec`**: This is the PyTorch Geometric implementation
    of the Node2Vec algorithm. It is a highly optimized, end-to-end solution for
    generating node embeddings. It handles the random walk generation and the
    training of a Skip-gram model using a single, unified interface.

#### 2. Multiprocessing and CUDA Compatibility

* **`multiprocessing.set_start_method('spawn', force=True)`**: This is a critical
    detail for compatibility between Python's `multiprocessing` and the CUDA
    runtime. The `'fork'` start method, which is the default on some Unix systems,
    can cause issues with CUDA initialization and resource management in child
    processes. The `'spawn'` method starts a fresh Python interpreter process
    for each worker, ensuring a clean state and avoiding conflicts.

#### 3. Data Representation and Conversion

* **`torch_geometric.data.Data`**: The script converts the `tskit.Tree` object
    into a `Data` object, which is a key requirement for using PyTorch Geometric.
    * The `edge_index` attribute of the `Data` object is populated with the
        parent-child relationships from the tskit tree, represented as a `(2, num_edges)`
        tensor.
    * The `num_nodes` attribute is also set to correctly define the size of the graph.
* **Numerical Precision**: The edge indices are converted to `torch.long` tensors
    to match the expected data type for graph operations in PyTorch Geometric.

#### 4. Node2Vec Hyperparameters

The script exposes several key hyperparameters for the Node2Vec algorithm:

* `embedding_dim`: The dimensionality of the output embedding vectors.
* `walk_length`: The length of each random walk.
* `num_walks`: The number of random walks to generate per node.
* `p` and `q`: These are not explicitly set in this script, but `torch_geometric`'s
    Node2Vec implementation defaults to `p=1.0` and `q=1.0`, which corresponds to
    the original Node2Vec algorithm's balance between breadth-first search (BFS)
    and depth-first search (DFS) exploration.
* `epochs`: The number of training epochs for the Word2Vec model.

#### 5. Data I/O and Persistence

* **`numpy.save(..., allow_pickle=False)`**: The final embedding vectors, which are
    PyTorch tensors, are converted to NumPy arrays and saved to `.npy` files.
    `allow_pickle=False` is used to ensure a simple, secure, and efficient binary
    format.
* **Path Management**: All file paths are constructed using `os.path.join` for
    cross-platform compatibility, and the output directories are created as needed
    using `os.makedirs`.

#### 6. Logging and Error Handling

* A robust logging system with timestamped log files is implemented to track
    the progress and status of each parallel job.
* Error handling is implemented with `try...except` blocks within the worker
    functions and the main loop. This ensures that a failure in one chromosome
    does not terminate the entire pipeline.
* The final summary reports a breakdown of successful and failed chromosomes,
    along with total processing time and rate, providing a clear overview of
    the pipeline's execution.
"""

import os
import sys
import time
import logging
import logging.handlers
import argparse
import psutil
import tskit
import torch
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_INPUT_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/samples"
DEFAULT_OUTPUT_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/embeddings/per_tree/gpu_accelerated"
DEFAULT_LOG_DIR = "/home/smondal/logs"

# EMBEDDING PARAMETERS
EMBEDDING_DIM = 128
WALK_LENGTH = 80
NUM_WALKS = 50
WINDOW_SIZE = 5
MIN_COUNT = 1
BATCH_WORDS = 4
EPOCHS = 1

# PROCESSING PARAMETERS
TREES_PER_LOG = 100
MIN_TREE_NODES = 2

# =============================================================================
# FUNCTIONS
# =============================================================================

def setup_logging(name_suffix="", log_dir=DEFAULT_LOG_DIR):
    """
    Configures a robust logging system with a timestamped log file.

    Args:
        name_suffix (str): An optional suffix for the log file name.
        log_dir (str): The directory to store log files.

    Returns:
        tuple: The logger object and the path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(log_dir, f"embeddings_gpu_{timestamp}{name_suffix}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove all existing handlers to prevent duplicate logging in child processes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

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

def generate_embedding(tree_id, tree_obj, chrom, device):
    """
    Generates Node2Vec embeddings for a single tskit tree using PyTorch Geometric
    and a specified device (CPU or GPU).

    Args:
        tree_id (int): The ID of the tree within the tree sequence.
        tree_obj (tskit.Tree): The tskit tree object.
        chrom (int): The chromosome number.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to use for computation.

    Returns:
        bool: True if embedding was successful, False otherwise.
    """
    if tree_obj.num_nodes <= MIN_TREE_NODES:
        return False
    
    # Create a PyTorch Geometric Data object from the tskit tree.
    try:
        edge_index = []
        for node in tree_obj.nodes():
            parent = tree_obj.parent(node)
            if parent != tskit.NULL:
                edge_index.append([parent, node])
        
        if not edge_index:
            return False

        # Convert the edge list to a PyTorch tensor.
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create a PyG Data object. `num_nodes` is essential for the model.
        data = Data(edge_index=edge_index_tensor, num_nodes=tree_obj.num_nodes)
        
        # Move the data to the specified device (GPU if available).
        data = data.to(device)

    except Exception as e:
        logging.error(f"Error converting tree to PyG Data for chr {chrom}, tree {tree_id}: {e}")
        return False

    # Configure and run the Node2Vec model on the selected device.
    try:
        model = Node2Vec(
            data.edge_index,
            embedding_dim=EMBEDDING_DIM,
            walk_length=WALK_LENGTH,
            context_size=WINDOW_SIZE,
            walks_per_node=NUM_WALKS,
            num_negative_samples=MIN_COUNT, # Corresponds to min_count in gensim's Word2Vec
            p=1, q=1, # Default Node2Vec parameters
        ).to(device)
        
        # Define the optimizer for training.
        loader = model.loader(batch_size=BATCH_WORDS, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train the model.
        model.train()
        for epoch in range(1, EPOCHS + 1):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
        
        # Get the final embeddings.
        model.eval()
        embeddings = model(torch.arange(tree_obj.num_nodes, device=device))
        
        # Save the embeddings as a numpy file.
        output_file_path = os.path.join(DEFAULT_OUTPUT_DIR, f"chr_{chrom}_tree_{tree_id}_embeddings.npy")
        np.save(output_file_path, embeddings.detach().cpu().numpy(), allow_pickle=False)
        
        return True
    except Exception as e:
        logging.error(f"Error generating embedding for chr {chrom}, tree {tree_id}: {e}", exc_info=True)
        return False

def process_chromosome(chrom, device):
    """
    Worker function to process all trees within a single chromosome's tree sequence file.

    Args:
        chrom (int): The chromosome number.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to use.

    Returns:
        tuple: (chromosome number, success status, number of trees processed)
    """
    # Set up child process-specific logging.
    setup_logging(f"_chrom_{chrom}", DEFAULT_LOG_DIR)
    
    logging.info(f"Child process for chr {chrom} started on device: {device}.")

    trees_file = os.path.join(DEFAULT_INPUT_DIR, f"ADNI.785_chr{chrom}.trees")
    trees_processed_count = 0
    start_time = time.time()
    
    try:
        ts = tskit.load(trees_file)
        logging.info(f"Loaded chr {chrom}: {ts.num_trees} trees, {ts.num_samples} samples, {ts.num_sites} sites.")

        for tree_id, tree in enumerate(ts.trees()):
            success = generate_embedding(tree_id, tree, chrom, device)
            if success:
                trees_processed_count += 1
            
            if (tree_id + 1) % TREES_PER_LOG == 0:
                elapsed_time = time.time() - start_time
                logging.info(f"Chr {chrom}: Processed {tree_id + 1} trees in {elapsed_time:.1f}s.")
        
        logging.info(f"Chr {chrom} completed. Total trees processed: {trees_processed_count}")
        return chrom, True, trees_processed_count

    except Exception as e:
        logging.error(f"Failed to process chromosome {chrom}. Error: {e}", exc_info=True)
        return chrom, False, trees_processed_count

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    """
    Main function to parse arguments, set up the pipeline, and orchestrate
    parallel chromosome processing.
    """
    parser = argparse.ArgumentParser(description="Generate GPU-accelerated Node2Vec embeddings for tskit trees.")
    parser.add_argument("--chromosomes", nargs='+', type=int, default=list(range(1, 23)),
                        help="List of chromosomes to process. Defaults to 1-22.")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Maximum number of chromosomes to process in parallel. Defaults to 1 to avoid GPU memory issues.")
    args = parser.parse_args()

    logger, main_log = setup_logging()
    
    # Check for CUDA availability and set device.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("CUDA GPU is available. Using GPU for embedding generation.")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        logging.warning("CUDA GPU not available. Falling back to CPU.")

    logging.info("Starting GPU-Accelerated Parallel Chromosome Embedding Generator.")
    logging.info(f"Input Directory: {DEFAULT_INPUT_DIR}")
    logging.info(f"Output Directory: {DEFAULT_OUTPUT_DIR}")
    logging.info(f"Chromosomes to process: {args.chromosomes}")
    logging.info(f"Max concurrent jobs: {args.max_concurrent}")
    logging.info(f"Device being used: {device}")

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    successful_chromosomes = []
    failed_chromosomes = []
    total_trees_processed = 0

    try:
        with ProcessPoolExecutor(max_workers=args.max_concurrent) as executor:
            future_to_chrom = {
                executor.submit(process_chromosome, chrom, device): chrom
                for chrom in args.chromosomes
            }
            for future in as_completed(future_to_chrom):
                chrom_num = future_to_chrom[future]
                try:
                    chrom_num, success, trees_processed = future.result()
                    if success:
                        successful_chromosomes.append(chrom_num)
                        total_trees_processed += trees_processed
                        logging.info(f"Chromosome {chrom_num} completed successfully.")
                    else:
                        failed_chromosomes.append(chrom_num)
                        logging.error(f"Chromosome {chrom_num} processing failed.")
                except Exception as e:
                    logging.error(f"An unexpected error occurred for chromosome {chrom_num}: {e}", exc_info=True)
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
        logging.info(f"Main log file: {main_log}")

        if failed_chromosomes:
            logging.error(f"Processing failed for chromosomes: {sorted(failed_chromosomes)}")
            sys.exit(1)

    except Exception as e:
        logging.critical(f"Fatal error in main process: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.info("Pipeline terminated.")

if __name__ == "__main__":
    # It's recommended to use 'spawn' or 'forkserver' for multiprocessing with CUDA
    # to avoid potential issues with forking.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logging.info("Multiprocessing start method already set.")

    main()
