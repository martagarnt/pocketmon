# Pockétmon - Binding Site Predictor

```
 ____   ___   ____ _  __ _____ _____ __  __  ___  _   _ 
|  _ \ / _ \ / ___| |/ /| ____|_   _|  \/  |/ _ \| \ | |
| |_/ | | | | |   | ' / |  _|   | | | |\/| | | | |  \| |
|  __/| |_| | |___| . \ | |___  | | | |  | | |_| | |\  |
|_|    \___/ \____|_|\_\|_____| |_| |_|  |_|\___/|_| \_|

     ░▒▓█▇▆▅▃▂▁  P O C K É T M O N  ▁▁▂▃▅▆▇█▓▒░
```

Pockétmon is a command-line tool and 3D Convolutional Neural Network that predicts ligand-binding pockets in protein structures from PDB files.

---

## Installation 

> **Recommended**: Use a Python 3.10 environment for best compatibility.

### 1. Clone the repository
```bash
git clone https://github.com/martagarnt/pocketmon.git
cd pocketmon
```

### 2. Create a virtual environment
If you're using Python 3.12+, you might encounter errors since some dependencies are not yet compatible with the latest Python versions.
In that case, we suggest creating a clean Python 3.10 environment using Conda:

#### 2.1. Create a virtual environment using Conda:

```bash
conda create -n pocketmon-env python=3.10
conda activate pocketmon-env
```

### 2.2 Create a regular virtual environment
In the case you have Python 3.10, no Conda environment needs to be created; just make sure the version is specified when calling python commands (in case you have serveral versions downloaded).

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the CLI tool
```bash
pip install .
```
> After this, the `pocketmon` command will be globally accessible in the terminal.

---

## Verifying Installation
To test if the installation was successful, run:
```bash
python3.10 -c "from pocketmon import predict; predict.print_banner('install'); print('✓ Pockétmon installed successfully and ready to catch some pockets\!')"
```

---

## Usage

```bash
pocketmon [options] --input <pdb_file_path>
```

### Options
- `--model <model_path>`: Path to the trained model weights (.pt file) `[default: best_model_refined.pt]`
- `--output <output_basename>`: Base name for output files `[default: predicted_pocket]`
- `--trust <float>`: Trust threshold for voxel classification `(0.0–1.0) [default: 0.5]`
- `-v, --verbose`: Enable verbose output
- `-h, --help`: Show help message and exit

### Required Arguments
- `--input <pdb_file_path>`: Path to the input PDB file

---

## Output
- A PDB file containing voxel pseudo-atoms representing the predicted pocket:
  - `<output_basename>_predicted_pocket.pdb`
- A text file listing predicted residues:
  - `<output_basename>_predicted_residues.txt`
  - Format: `RESNAME RESID CHAIN` (e.g., `LEU 6 A`)

---

## Trust Threshold Explained

The trust threshold sets how confident the model must be before classifying a voxel as part of the pocket:

- **Low values (e.g. 0.3)** → More voxels included, more recall
- **High values (e.g. 0.8)** → Fewer voxels, higher precision
- **Default (0.5)** → Balanced tradeoff

Tune this parameter to your needs using `--trust`.

---

## Example Usage
```bash
pocketmon --input myprotein.pdb
pocketmon -v --input 2ay2.pdb --model best_model_refined.pt --output result --trust 0.6
```

---

## Contact
For questions or support:
- Marta García · marta.garcia38@estudiant.upf.edu
- Karim Hamed  · karim.hamed01@estudiant.upf.edu
- Ivon Sánchez · ivon.sanchez01@estudiant.upf.edu
