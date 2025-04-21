#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from Bio.PDB import PDBParser
import os
import platform

# ------------------------------
# 🧠 CNN Model Definition
# ------------------------------
import torch.nn as nn

print( """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀ ⠀⠀⠀ ⠀⢀⣠⡤⠶⠖⠒⠶⠤⣄⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀  ⠀  ⠀⠀⠀⢀⡰⠞⢉⢀⠀⡀⡀⡀   ⠀⠈⠓⢤⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀ ⠀⠀⠀ ⠀⠀⣠⠎⠀⣠⡠⢋⣬⠊⢄⠑⢀⠀⠀⠄⢀  ⠀⡹⣄⠀⠀ 
⠀⠀⠀⠀⠀⠀ ⠀ ⠀⠀ ⢰⠃⣠⣿⠟⠓⢙⣋⡓⠑⠐⢌⠎⠞⠲⠀ ⢤⣿⣿⡆⠀
⠀⠀⠀⠀ ⠀ ⠀    ⠛⢁⣴⣶⣿⣶⣼⡖⠀⢁⣀⣦⣴⣿⣿⣿⣿⡀           Pockétmon has started running! 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠠⣷⣯⣿⣷⣿⢏⡴⡢⣌⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃         Let's catch some pockets!
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡟⠛⠿⠿⣿⡘⣄⢀⣼⢠⣿⣿⣿⣿⡿⠿⠿⠿⠛⢻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣻⡀⠀⠀⠘⠿⣦⣯⣴⡿⠋⠀⠀⠀⠀ ⠀⠀⠀⠀⣞⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⣷⡀⠀⠀⠀⠀⠐⠂⠁⠀⠀⠀⠀⠀ ⠀⠀⢀⡾⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢀⡴⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⢷⠦⢄⣀⣀⣀⣀⣠⠴⠚⠉⠀⠀⠀⠀⠀

""")

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

# ------------------------------
# 🧊 Voxelization Function
# ------------------------------
def voxelize_structure(pdb_path, origin=None, grid_size=32, voxel_size=1.0, channels=['C', 'N', 'O', 'S'], return_origin=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    if origin is None:
        coords = np.array([atom.coord for atom in structure.get_atoms()])
        origin = coords.mean(axis=0) - (grid_size * voxel_size / 2)

    grid = np.zeros((len(channels), grid_size, grid_size, grid_size), dtype=np.float32)

    for atom in structure.get_atoms():
        atom_type = atom.element.strip()
        if atom_type not in channels:
            continue
        idx = channels.index(atom_type)
        coord = np.array(atom.coord)
        voxel = ((coord - origin) / voxel_size).astype(int)
        if all(0 <= v < grid_size for v in voxel):
            grid[idx, voxel[0], voxel[1], voxel[2]] += 1

    return (grid, origin) if return_origin else grid

# ------------------------------
# 💾 Save Output to PDB
# ------------------------------
def save_predicted_pocket_to_pdb(pred_grid, origin, voxel_size, pdb_filename, threshold=0.5):
    if torch.is_tensor(pred_grid):
        pred_grid = pred_grid.squeeze().detach().cpu().numpy()

    with open(pdb_filename, 'w') as f:
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

# ------------------------------
# 🖥️ CLI + Inference Logic
# ------------------------------
def main():

    parser = argparse.ArgumentParser(description="Predict binding pockets from protein PDB file using 3D CNN.")
    parser.add_argument('--input', required=True, help="Path to input protein PDB file")
    parser.add_argument('--model', default='best_model.pt', help="Path to trained PyTorch model weights")
    parser.add_argument('--output', default='predicted_pocket.pdb', help="Output PDB filename for predicted pocket")
    args = parser.parse_args()

    # Smart device detection
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        print("⚠️ MPS is available (Apple GPU), but Conv3D is NOT supported — using CPU instead.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("✅ CUDA is available — using GPU")
        device = torch.device("cuda")
    else:
        print("⚠️ No GPU available — using CPU")
        device = torch.device("cpu")

    # Load model
    model = Pocket3DCNN(in_channels=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess input
    protein_grid, origin = voxelize_structure(args.input, return_origin=True)
    X = torch.tensor(protein_grid, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, D, H, W)

    # Run prediction
    with torch.no_grad():
        preds = model(X)
    pred_mask = (preds > 0.5).float()

    # Save predicted mask to PDB
    save_predicted_pocket_to_pdb(pred_mask, origin, voxel_size=1.0, pdb_filename=args.output)
    print(f"✅ Prediction saved to: {args.output}")

if __name__ == "__main__":
    main()
