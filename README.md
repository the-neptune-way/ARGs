# VCF-conversion-from-trees-files

# Tree-sequence to VCF Converter

This Python script efficiently converts a series of **tskit** tree-sequence files into Variant Call Format (VCF) files. It's designed to process human autosomal chromosomes (1-22) in parallel, making it ideal for large-scale genomic data conversion.

## How It Works

The script takes a set of input files, each representing a chromosome's tree-sequence, and converts them one by one into VCF files. It leverages Python's `concurrent.futures` module to use multiple CPU cores, allowing for faster processing.

### Key Features

  * **Parallel Processing**: Converts chromosomes 1-22 simultaneously using a process pool.
  * **Error Handling**: Catches potential errors during file loading and VCF writing and reports which chromosomes failed.
  * **Input/Output Management**: Automatically creates the output directory if it doesn't exist.
  * **Progress and Summary Reporting**: Prints progress updates for each chromosome and provides a final summary of successful and failed conversions.

## Requirements

  * **tskit**: The `tskit` library must be installed in your Python environment.
  * **Python 3**: The script requires a Python 3 environment.

-----

## Configuration

Before running the script, you must configure the following variables inside the script itself:

  * `INPUT_DIR`: The directory where your `.trees` files are located.
  * `OUTPUT_DIR`: The directory where you want the VCF files to be saved.
  * `SAMPLE_PREFIX`: The common prefix for your input files. For example, if your files are named `ADNI.785_chr1.trees`, `ADNI.785_chr2.trees`, etc., the prefix is `ADNI.785`.

-----

## Usage

1.  **Set up your environment**: Make sure you have `tskit` installed. If you're using `conda`, the environment used to run the script can be created as follows:

    ```bash
    conda create --name tskit-env python=3.10
    conda activate tskit-env
    conda install -c conda-forge tskit
    ```

2.  **Configure the script**: Open the script file and update the `INPUT_DIR`, `OUTPUT_DIR`, and `SAMPLE_PREFIX` variables with your specific paths and file names.

3.  **Run the script**: Execute the script from your terminal.

    ```bash
    python your_script_name.py
    ```

The script will begin processing the files and print its progress to the console. Once completed, it will provide a final summary of the conversion process.

If any chromosome fails to convert, the script will exit with an error code and list the failed chromosomes. If all chromosomes are successful, it will exit normally.
