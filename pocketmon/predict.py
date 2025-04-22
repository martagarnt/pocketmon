#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from Bio.PDB import PDBParser
import os
import platform
import torch.nn as nn
import sys

################
# Init program #
################

def print_banner(mode='normal'):
    if mode == 'help':
        print("""
 ____   ___   ____ _  __ _____ _____ __  __  ___  _   _ 
|  _ \ / _ \ / ___| |/ /| ____|_   _|  \/  |/ _ \| \ | |
| |_/ | | | | |   | ' / |  _|   | | | |\/| | | | |  \| |
|  __/| |_| | |___| . \ | |___  | | | |  | | |_| | |\  |
|_|    \___/ \____|_|\_\|_____| |_| |_|  |_|\___/|_| \_|

       ░▒▓█▇▆▅▃▂▁  P O C K É T M O N  ▁▁▂▃▅▆▇█▓▒░
   Pocketmon Help Page — Learn how to use this tool
        """)

    elif mode == 'install':
        print("""
 ____   ___   ____ _  __ _____ _____ __  __  ___  _   _ 
|  _ \ / _ \ / ___| |/ /| ____|_   _|  \/  |/ _ \| \ | |
| |_/ | | | | |   | ' / |  _|   | | | |\/| | | | |  \| |
|  __/| |_| | |___| . \ | |___  | | | |  | | |_| | |\  |
|_|    \___/ \____|_|\_\|_____| |_| |_|  |_|\___/|_| \_|

       ░▒▓█▇▆▅▃▂▁  P O C K É T M O N  ▁▁▂▃▅▆▇█▓▒░
Predict ligand-binding pockets in proteins using CNN
   Tip: Run `pocketmon --help` to get started.
        """)

    else: 
        print( """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀ ⠀⠀⠀ ⠀⢀⣠⡤⠶⠖⠒⠶⠤⣄⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀  ⠀  ⠀⠀⠀⢀⡰⠞⢉⢀⠀⡀⡀⡀   ⠀⠈⠓⢤⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀ ⠀⠀⠀ ⠀⠀⣠⠎⠀⣠⡠⢋⣬⠊⢄⠑⢀⠀⠀⠄⢀  ⠀⡹⣄⠀⠀ 
⠀⠀⠀⠀⠀⠀ ⠀ ⠀⠀ ⢰⠃⣠⣿⠟⠓⢙⣋⡓⠑⠐⢌⠎⠞⠲⠀ ⢤⣿⣿⡆⠀       ░▒▓█▇▆▅▃▂▁  P O C K É T M O N  ▁▁▂▃▅▆▇█▓▒░
⠀⠀⠀⠀ ⠀ ⠀    ⠛⢁⣴⣶⣿⣶⣼⡖⠀⢁⣀⣦⣴⣿⣿⣿⣿⡀                  Pockétmon has started running! 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠠⣷⣯⣿⣷⣿⢏⡴⡢⣌⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃                Let's catch some pockets!
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡟⠛⠿⠿⣿⡘⣄⢀⣼⢠⣿⣿⣿⣿⡿⠿⠿⠿⠛⢻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣻⡀⠀⠀⠘⠿⣦⣯⣴⡿⠋⠀⠀⠀⠀ ⠀⠀⠀⠀⣞⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⣷⡀⠀⠀⠀⠀⠐⠂⠁⠀⠀⠀⠀⠀ ⠀⠀⢀⡾⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢀⡴⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⢷⠦⢄⣀⣀⣀⣀⣠⠴⠚⠉⠀⠀⠀⠀⠀
""")

######################## 
# CNN Model Definition #
########################

class Pocket3DCNN(nn.Module):
    def __init__(self, in_channels=4):
        super(Pocket3DCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

#########################
# Voxelization Function #
#########################

def voxelize_structure(pdb_path, origin=None, grid_size=32, voxel_size=1.0, channels=['C', 'N', 'O', 'S']):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    if origin is None:
        coords = np.array([atom.coord for atom in structure.get_atoms()])
        origin = coords.mean(axis=0) - (grid_size * voxel_size / 2)

    grid = np.zeros((len(channels), grid_size, grid_size, grid_size), dtype=np.float32)
    atom_residue_map = {}

    for atom in structure.get_atoms():
        atom_type = atom.element.strip()
        if atom_type not in channels:
            continue
        idx = channels.index(atom_type)
        coord = np.array(atom.coord)
        voxel = ((coord - origin) / voxel_size).astype(int)
        if all(0 <= v < grid_size for v in voxel):
            grid[idx, voxel[0], voxel[1], voxel[2]] += 1
            atom_residue_map[tuple(voxel)] = atom.get_parent()

    return grid, origin, structure, atom_residue_map

######################
# Save Output to PDB # 
######################

def save_predicted_pocket_to_pdb(pred_grid, origin, voxel_size, pdb_filename, threshold=0.5, header_name=None):
    if torch.is_tensor(pred_grid):
        pred_grid = pred_grid.squeeze().detach().cpu().numpy()

    with open(pdb_filename, 'w') as f:
        if header_name:
            f.write(f"HEADER    Predicted pocket for {header_name}\n")
        atom_index = 1
        for x in range(pred_grid.shape[0]):
            for y in range(pred_grid.shape[1]):
                for z in range(pred_grid.shape[2]):
                    if pred_grid[x, y, z] > threshold:
                        coord = origin + np.array([x, y, z]) * voxel_size
                        f.write(
                            f"HETATM{atom_index:5d}  X   UNK     1    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           X\n"
                        )
                        atom_index += 1
        f.write("END\n")


########################
# Save residues to txt #
########################

def save_predicted_residues(pred_grid, origin, voxel_size, residue_lookup, output_file, threshold=0.5):
    if torch.is_tensor(pred_grid):
        pred_grid = pred_grid.squeeze().detach().cpu().numpy()

    seen_residues = set()
    residue_lines = []

    for x in range(pred_grid.shape[0]):
        for y in range(pred_grid.shape[1]):
            for z in range(pred_grid.shape[2]):
                if pred_grid[x, y, z] > threshold:
                    voxel = (x, y, z)
                    if voxel in residue_lookup:
                        res = residue_lookup[voxel]
                        res_id = (res.get_resname(), res.get_id()[1], res.get_parent().id)
                        if res_id not in seen_residues:
                            seen_residues.add(res_id)
                            residue_lines.append(f"{res_id[0]} {res_id[1]} {res_id[2]}")

    residue_lines.sort(key=lambda x: (x.split()[2], int(x.split()[1])))

    with open(output_file, 'w') as f:
        f.write("\n".join(residue_lines))

#########################
# CLI + Inference Logic #
#########################


def main():
    if len(sys.argv) == 1 or any(arg in ("-h", "--help") for arg in sys.argv):
        print_banner("help")
        print("""
☞ Predict protein binding pockets using a 3D CNN

Required arguments:
  --input        Path to input protein PDB file

Optional arguments:
  --model        Path to trained model weights (.pt)
  --output       Base name for output files (will generate <name>_predicted_pocket.pdb and <name>_predicted_residues.txt)
  --trust        Trust threshold for voxel classification (0.0–1.0, default: 0.5)
  -v, --verbose  Enable verbose mode for debugging output

Example usage:
  pocketmon --input protein.pdb --model best_model.pt --output result
  pocketmon -v --trust 0.6 --input test.pdb
        """)
        sys.exit(0)

    else:
        print_banner("run")
    
    parser = argparse.ArgumentParser(
        prog='pocketmon',
        description="Predict protein binding pockets using a 3D CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', required=True, metavar="pdb_protein.pdb", help="Path to input protein PDB file")
    parser.add_argument('--model', default='best_model_refined.pt', metavar="CNN_model.pt", help="Path to trained model weights (.pt)")
    parser.add_argument('--output', default='predicted_pocket.pdb', metavar="name_output_pocket.pdb", help="Base name for output files. Will generate <name>_predicted_pocket.pdb and <name>_predicted_residues.txt")
    parser.add_argument('--trust', type=float, default=0.5, metavar="trust_threshold", help="Trust threshold for voxel classification (default: 0.5)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")

    
    args = parser.parse_args()

    if args.verbose:
        print("[VERBOSE] Parsed command-line arguments:", args)

    # Smart device detection

    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        print(" ▲ MPS is available (Apple GPU), but Conv3D is NOT supported — using CPU instead.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print(" ✓ CUDA is available — using GPU")
        device = torch.device("cuda")
    else:
        print(" ▲ No GPU available — using CPU")
        device = torch.device("cpu")
    
    if args.verbose:
        print("[VERBOSE] Using device:", device)

    # Load model
    model = Pocket3DCNN(in_channels=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    if args.verbose:
        print("[VERBOSE] Model loaded from", args.model, "and moved to", device)
        print("[VERBOSE] Predicition thershold set to:", args.trust)

    # Preprocess input
    protein_grid, origin, structure, residue_lookup = voxelize_structure(args.input)
    X = torch.tensor(protein_grid, dtype=torch.float32).unsqueeze(0).to(device)
    if args.verbose:
        print("[VERBOSE] Voxelization done. Grid shape:", protein_grid.shape)

    # Run prediction
    with torch.no_grad():
        preds = model(X)
    pred_mask = (preds > args.trust).float()
    if args.verbose:
        print("[VERBOSE] Total predicted voxels above threshold:", np.count_nonzero(pred_mask.cpu().numpy()))


    # Extract protein name from input path
    protein_name = os.path.splitext(os.path.basename(args.input))[0]

    # Define dynamic output names
    pdb_output = f"{protein_name}_predicted_pocket.pdb"
    txt_output = f"{protein_name}_predicted_residues.txt"

    if args.verbose:
        print("[VERBOSE] Output files will be saved as:", pdb_output, txt_output)

    # Save PDB
    save_predicted_pocket_to_pdb(pred_mask, origin, voxel_size=1.0, pdb_filename=pdb_output, header_name=protein_name)
    print(f"✓ Pocket prediction saved to: {pdb_output}")

    # Save residue list
    save_predicted_residues(pred_mask, origin, 1.0, residue_lookup, txt_output, threshold=args.trust)
    print(f"✓ Residue list saved to: {txt_output}")

if __name__ == "__main__":
    main()