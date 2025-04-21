# Pockétmon: Protein Binding Pocket Predictor 🔛

Pockétmon is a command-line tool and 3D Convolutional Neural Network that predicts ligand-binding pockets in protein structures from PDB files.
It voxelizes the protein and outputs the predicted pocket as a new PDB file.


## 🚀 Features

- Parses PDB protein structures and voxelizes them
- Uses a 3D CNN to predict pocket occupancy
- Outputs the pocket prediction as a pseudo-PDB file
- Supports CPU and automatically detects Apple M1/M2/M3 GPUs
- Includes a beautiful startup banner!


## ⚙️ Installation

>**Recommended**: Use a Python 3.10 environment (due to compatibility with PyTorch and MPS)

### 1. Clone the repo

```bash
git clone https://github.com/martagarnt/pocketmon.git
cd pocketmon
```

### 2. Create a virtual environment
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
This will let you call `pocketmon` from anywhere.


## 🔧 Usage

### Predict a binding pocket:
```bash
pocketmon --input 2ay2.pdb --model best_model.pt --output predicted_pocket.pdb
```

### Arguments:
- `--input`: Path to a protein PDB file
- `--model`: Path to the trained PyTorch model (default: `best_model.pt`)
- `--output`: Output filename (default: `predicted_pocket.pdb`)


## 📄 Output Format
The output is a PDB file where each predicted pocket voxel is written as a pseudo-atom. You can post-process this to map to real residues.

If you'd like to instead output **residue-level predictions** (e.g., `LEU 6 A`), let us know — it's supported with a small change.

## 🚫 MPS Support Note
Apple Silicon (M1/M2/M3) supports GPU acceleration via MPS, but **Conv3D is not yet supported**.
We automatically fallback to CPU if needed.


## ⚡️ Authors
- Marta García
- Karim Hamed
- Ivon Sánchez

Enjoy catching those pockets!


