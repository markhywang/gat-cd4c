import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import io
from PIL import Image

# Temporary imports
import numpy as np
import random
random.seed(0)


def create_gradient_circle(ax, x, y, radius, color, center_alpha=0.3, edge_alpha=0):
    # Create a grid for the gradient
    grid_size = 500
    x_vals = np.linspace(x - radius, x + radius, grid_size)
    y_vals = np.linspace(y - radius, y + radius, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute distances from the center
    dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    # Normalize distances
    norm = Normalize(vmin=0, vmax=radius)
    gradient = norm(dist_from_center)
    gradient = np.clip(1 - gradient, 0, 1)  # More intense color inside, fading outward

    # Fetch the specified color map
    color_map = plt.get_cmap(color)
    rgba_colors = color_map(gradient)

    # Adjust the alpha (opacity) channel
    alpha_gradient = np.clip(center_alpha * gradient + edge_alpha * (1 - gradient), edge_alpha, center_alpha)
    rgba_colors[..., 3] = alpha_gradient

    # Overlay the gradient image
    ax.imshow(
        rgba_colors,
        extent=(x - radius, x + radius, y - radius, y + radius),
        origin='lower',
        interpolation='bilinear',
    )


def visualize_molecule_with_positions(smiles):
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Error: Could not parse SMILES string: {smiles}")
        return None, None

    # Remove hydrogens from the molecule
    mol = Chem.RemoveHs(mol)

    # Generate 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)

    # Create a molecule drawing
    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    coords = []
    for atom in range(mol.GetNumAtoms()):
        coords.append(drawer.GetDrawCoords(atom))

    # Convert the drawing to a PIL Image
    png_data = drawer.GetDrawingText()
    image = Image.open(io.BytesIO(png_data))

    # Plot the image with matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Add circles over all atoms
    for pos in coords:
        # Scale the coordinates to match the image dimensions
        x, y = pos
        strength = min(0.5, np.random.pareto(5))
        create_gradient_circle(ax, x, y, 100, 'Reds' if random.random() < 0.5 else 'Greens', center_alpha=strength)
        #circle = mpatches.Circle((x, y), radius=10, alpha=0.5, color='blue')
        #ax.add_patch(circle)

    # Set axis limits and remove ticks
    ax.set_xlim(0, 1200)
    ax.set_ylim(1200, 0)  # Flip y-axis
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title with SMILES
    plt.title(f"Molecule: {smiles}")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    smiles = "COCCOCC#CC(=O)Nc1ccc2ncc(C#N)c(Nc3cccc(Br)c3)c2c1"
    visualize_molecule_with_positions(smiles)
