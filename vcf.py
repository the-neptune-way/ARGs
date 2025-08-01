#!/home/smondal/miniconda3/envs/tskit-env/bin/python
"""
tskit_to_vcf_converter.py

This script is a high-performance conversion utility designed to transform
tskit tree-sequence files for human autosomes (1-22) into standard VCF files.
It is built upon the tskit library and leverages Python's multiprocessing
capabilities to achieve significant speed improvements over sequential processing.

Technical Implementation Details:

1.  Parallel Processing with `concurrent.futures.ProcessPoolExecutor`:
    - The script utilizes `concurrent.futures.ProcessPoolExecutor` to spawn
      multiple independent Python processes.
    - `max_workers` is set to 22, matching the number of human autosomes,
      to allow for simultaneous processing of all chromosomes. This is a
      CPU-bound task, making multiprocessing a highly effective strategy.
    - Each process executes the `convert_chromosome` function on a separate
      chromosome, completely isolating memory and execution state.

2.  `tskit` Library Integration:
    - The core conversion logic relies on the `tskit.load()` and
      `tskit.write_vcf()` methods.
    - `tskit.load(trees_file)`: This method deserializes a tree sequence
      from a binary `.trees` file into a `tskit.TreeSequence` object in memory.
      This object is a highly efficient data structure that stores the
      complete genetic history of the samples.
    - `tskit.write_vcf(vcf, contig_id=str(chrom))`: This method is a
      specialized function for VCF output. It iterates through all sites
      in the `TreeSequence` object, determines the genotypes for each
      sample, and writes the output in a VCF 4.2 compliant format.
      - The `contig_id` parameter is crucial for generating a valid VCF header,
        specifying the `##contig` line (e.g., `##contig=<ID=1>`).

3.  File Naming Convention and I/O:
    - The script strictly adheres to a file naming convention to automate
      the processing of multiple chromosomes.
    - Input files: `<SAMPLE_PREFIX>_chr<CHROMOSOME_NUMBER>.trees`.
    - Output files: `<SAMPLE_PREFIX>_chr<CHROMOSOME_NUMBER>.vcf`.
    - The script uses `os.path.join` for robust, cross-platform path
      construction.
    - Output directory creation: `os.makedirs(OUTPUT_DIR, exist_ok=True)`
      ensures the target directory exists without raising an error if it
      already does.

4.  Error Handling and Reporting:
    - Each `convert_chromosome` call is wrapped in a `try...except` block
      to gracefully handle potential issues with a single chromosome without
      crashing the entire script.
    - Specific exceptions caught include `FileNotFoundError` for missing
      input files and a general `Exception` for other runtime issues.
    - The script provides detailed error messages to `sys.stderr` to
      differentiate them from standard output, which typically shows
      progress updates.
    - The final summary at the end of the script aggregates the boolean
      results from each parallel process to provide a comprehensive
      report of successes and failures.
    - The script's exit code (`sys.exit(0)` for success, `sys.exit(1)` for
      failure) makes it suitable for integration into automated pipelines and
      shell scripts.

5.  Memory and Resource Management:
    - The script's memory footprint is heavily dependent on the size of the
      input `.trees` files. Each parallel process loads a full tree sequence
      into memory.
    - The `ProcessPoolExecutor` manages the lifecycle of the worker processes,
      ensuring they are properly started and shut down. This prevents resource
      leaks.

"""
import tskit
import os
import concurrent.futures
import sys

# =============================================================================
# Configuration
# =============================================================================
# Absolute path to the directory containing the input tskit tree-sequence files.
INPUT_DIR = "/home/smondal/ADNI/20250321-ADNI-trees/samples"

# Absolute path to the destination directory for the generated VCF files.
OUTPUT_DIR = "/home/smondal/ADNI/vcfs"

# The common file prefix used for all chromosome-specific files.
SAMPLE_PREFIX = "ADNI.785"

# =============================================================================
# Functions
# =============================================================================
def convert_chromosome(chrom):
    """
    Worker function for a single chromosome conversion.

    Args:
        chrom (int): The chromosome number to process (e.g., 1, 2, ...).

    Returns:
        bool: `True` upon successful VCF generation and validation, `False` otherwise.
    """
    trees_file = os.path.join(INPUT_DIR, f"{SAMPLE_PREFIX}_chr{chrom}.trees")
    vcf_file = os.path.join(OUTPUT_DIR, f"{SAMPLE_PREFIX}_chr{chrom}.vcf")

    print(f"Processing chromosome {chrom}")

    try:
        # Load the tree sequence from disk. This is a CPU-intensive and I/O-bound
        # operation. The resulting `tskit.TreeSequence` object is a complete
        # representation of the genetic data.
        ts = tskit.load(trees_file)
        print(f"  Loaded chr{chrom}: {ts.num_samples} samples, {ts.num_sites} sites")

        # Write the VCF file. `ts.write_vcf` efficiently iterates through
        # the sites in the tree sequence and outputs VCF 4.2 compliant data.
        with open(vcf_file, "w") as vcf:
            ts.write_vcf(vcf, contig_id=str(chrom))

        # Post-conversion validation: Check if the file was created and is non-empty.
        if os.path.exists(vcf_file) and os.path.getsize(vcf_file) > 0:
            size_mb = os.path.getsize(vcf_file) / 1e6
            print(f"  Saved VCF: {os.path.basename(vcf_file)} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"ERROR: Empty output for chr{chrom}", file=sys.stderr)
            return False

    except FileNotFoundError:
        print(f"ERROR: Input file not found for chr{chrom}: {trees_file}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR processing chr{chrom}: {str(e)}", file=sys.stderr)
        return False

# =============================================================================
# Main Execution Block (`if __name__ == "__main__":`)
# =============================================================================
if __name__ == "__main__":
    # Ensure the output directory exists before any worker process attempts to write.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instantiate the process pool with a fixed number of workers.
    with concurrent.futures.ProcessPoolExecutor(max_workers=22) as executor:
        chromosomes = range(1, 23)
        # `executor.map` distributes the `convert_chromosome` function calls
        # to the worker processes. It's a blocking call that waits for all
        # results to be returned before proceeding.
        results = list(executor.map(convert_chromosome, chromosomes))

    # Aggregate and report results from the parallel jobs.
    success_count = sum(results)
    print(f"\nConversion completed: {success_count}/22 chromosomes succeeded")

    # Exit with an appropriate status code for shell scripting.
    if success_count < 22:
        failed = [i + 1 for i, success in enumerate(results) if not success]
        print(f"Failed chromosomes: {failed}", file=sys.stderr)
        sys.exit(1)  # Exit code 1 indicates an error.
    else:
        print("All chromosomes converted successfully")
        sys.exit(0)  # Exit code 0 indicates success.
