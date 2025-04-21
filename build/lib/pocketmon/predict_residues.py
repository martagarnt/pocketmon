import argparse
import torch
import numpy as np
from Bio.PDB import PDBParser
import os
import torch.nn as nn

# ------------------------------
# 🧠 CNN Model Definition
# ------------------------------
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
            atom_residue_map[tuple(voxel)] = atom.get_parent()  # residue

    return (grid, origin, atom_residue_map) if return_origin else (grid, None, atom_residue_map)

# ------------------------------
# 📋 Save Residue Predictions
# ------------------------------
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

    residue_lines.sort(key=lambda x: (x.split()[2], int(x.split()[1])))  # sort by chain then resnum

    with open(output_file, 'w') as f:
        f.write("\n".join(residue_lines))

# ------------------------------
# 🖥️ CLI + Inference Logic
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict binding pocket residues from protein PDB file using 3D CNN.")
    parser.add_argument('--input', required=True, help="Path to input protein PDB file")
    parser.add_argument('--model', default='best_model.pt', help="Path to trained PyTorch model weights")
    parser.add_argument('--output', default='predicted_residues.txt', help="Output text filename with predicted residues")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model = Pocket3DCNN(in_channels=4)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    protein_grid, origin, res_lookup = voxelize_structure(args.input, return_origin=True)
    X = torch.tensor(protein_grid, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(X)
    pred_mask = (preds > 0.5).float()

    save_predicted_residues(pred_mask, origin, voxel_size=1.0, residue_lookup=res_lookup, output_file=args.output)
    print(f"✅ Predicted residues saved to: {args.output}")

if __name__ == "__main__":
    main()
