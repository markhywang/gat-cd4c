import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Chem

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import math
import io
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.dataset import DrugProteinDataset, DrugMolecule
from utils.helper_functions import set_seeds
from utils.functional_groups import *
from model import GraphAttentionNetwork


class AnalysisApp(tk.Tk):
    def __init__(self, data_path: str) -> None:
        super().__init__()

        self.title("CD4C Analysis")
        self.state("normal")

        data_df, protein_embeddings_df = load_data(data_path)
        # TODO - remove hard-coded model parameters
        self.model = GraphAttentionNetwork(
            "cpu",
            349,
            1,
            16,
            96,
            8,
            6,
            0.2,
            0.1,
            96
        ).to(torch.float32).to("cpu")
        self.model.load_state_dict(
            torch.load("../models/model.pth", weights_only=False, map_location=torch.device('cpu')))
        self.model.eval()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        # Increase the font size and padding for the tabs.
        ttk.Style().configure("TNotebook.Tab", padding=[10, 5], font=("Arial", 10))

        molecule_viewer = MoleculeViewer(self, data_df, protein_embeddings_df, self.model)
        self.notebook.add(molecule_viewer, text="Molecule Viewer")

        model_analysis = ModelAnalysis(self, data_df, protein_embeddings_df, self.model)
        self.notebook.add(model_analysis, text="Model Analysis")


class ModelAnalysis(tk.Frame):
    def __init__(self, root: tk.Tk, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame,
                 model: nn.Module) -> None:
        super().__init__(root)

        # Use a white background.
        self.configure(bg="white")

        self.model = model
        self.data_df = data_df
        self.protein_embeddings_df = protein_embeddings_df
        self._create_settings_frame()

        self.bind("<Destroy>", self._on_destroy)
        self.confusion_plot = None
        self.confusion_canvas = None
        self.scatter_plot = None
        self.scatter_canvas = None

    def _on_destroy(self, event) -> None:
        if self.confusion_plot is not None:
            plt.close(self.confusion_plot)
        if self.scatter_plot is not None:
            plt.close(self.scatter_plot)

    def _update_display(self) -> None:
        self.test_dataset = self._get_test_dataset(self.data_df, self.protein_embeddings_df)
        self.pchembl_preds, self.pchembl_labels = self._eval_model(self.percent_data_slider.get())

        if self.confusion_plot is not None:
            self.confusion_canvas.get_tk_widget().destroy()
            plt.close(self.confusion_plot)
        if self.scatter_plot is not None:
            self.scatter_canvas.get_tk_widget().destroy()
            plt.close(self.scatter_plot)

        self._update_confusion_matrix()
        self._update_scatter()

    def _create_settings_frame(self):
        settings_frame = tk.Frame(self, padx=5, pady=5)

        # Create a slider that controls the percentage of test data that is used to do the analysis.
        tk.Label(settings_frame, text="% Data Used for Analysis").grid(row=0, column=0, padx=20, pady=(20, 0))
        # Create a slider that controls the percentage of test data that is used to do the analysis.
        self.percent_data_slider = tk.Scale(settings_frame, from_=0.05, to=1.0, resolution=0.05, orient="horizontal")
        self.percent_data_slider.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        generate_button = tk.Button(settings_frame, text="Generate Analysis", command=self._update_display)
        generate_button.grid(row=0, column=1, rowspan=2, padx=20, pady=20)

        settings_frame.grid(row=0, column=0, padx=20, pady=20, sticky='w')

    def _update_confusion_matrix(self):
        confusion_dict = self._get_confusion_dict()
        self.confusion_plot = self._plot_confusion_matrix(confusion_dict)

        # Embed the plot into the Tkinter frame.
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_plot, master=self)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().config(width=500, height=400)
        self.confusion_canvas.get_tk_widget().grid(row=1, column=0, padx=20, pady=20)

    def _plot_confusion_matrix(self, confusion_dict: dict[str, float]) -> plt.Figure:
        # Extract values from the dictionary.
        tp = confusion_dict['true_positive']
        fp = confusion_dict['false_positive']
        tn = confusion_dict['true_negative']
        fn = confusion_dict['false_negative']

        # Construct the confusion matrix.
        matrix = np.array([[tp, fp], [fn, tn]])

        # Normalize by total values for color scaling.
        matrix_norm = matrix.astype('float') / matrix.sum()

        # Create the plot.
        fig, ax = plt.subplots(figsize=(4, 4))

        # Add a color bar on the right side that uses different shades of blue.
        cax = ax.matshow(matrix_norm, cmap='Blues')
        plt.colorbar(cax, shrink=0.8)

        # Set tick positions and labels.
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'], fontsize=8)
        ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=8)

        # Annotate each cell with its value.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{matrix[i, j]}", ha='center', va='center', color='black', fontsize=12)

        plt.title("Confusion Matrix", fontsize=16, pad=30)

        # Automatically adjust subplots to fit into the figure area.
        fig.tight_layout()

        return fig

    def _update_scatter(self):
        self.scatter_plot = self._plot_scatter(self.pchembl_preds, self.pchembl_labels)

        # Embed the plot into the Tkinter frame.
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_plot, master=self)
        self.scatter_canvas.draw()
        self.scatter_canvas.get_tk_widget().config(width=500, height=400)
        self.scatter_canvas.get_tk_widget().grid(row=1, column=1, padx=20, pady=20)

    def _plot_scatter(self, preds: torch.Tensor, labels: torch.Tensor) -> plt.Figure:
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        # Create the plot.
        fig, ax = plt.subplots(figsize=(4, 4))

        plt.scatter(labels, preds, alpha=0.5, label='Predictions')
        plt.title("pChEMBL Predictions", fontsize=16, pad=20)

        # Plot the y = x line
        min_val = min(labels.min(), preds.min())
        max_val = max(labels.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        plt.xlabel('Actual pChEMBL Score')
        plt.ylabel('Predicted pChEMBL Score')
        plt.legend()
        plt.grid(True)

        # Automatically adjust subplots to fit into the figure area.
        fig.tight_layout()

        return fig

    def _get_confusion_dict(self, pchembl_threshold: float = 7.0) -> dict[str, int]:
        positive_preds = [x >= pchembl_threshold for x in self.pchembl_preds]
        positive_labels = [x >= pchembl_threshold for x in self.pchembl_labels]
        confusion_dict = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}

        for pred, label in zip(positive_preds, positive_labels):
            pred_str = 'positive' if pred else 'negative'
            is_correct = 'true' if pred == label else 'false'
            confusion_dict[f"{is_correct}_{pred_str}"] += 1;

        return confusion_dict

    def _eval_model(self, percent_data, batch_size: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 < percent_data <= 1

        # Choose an arbitrary batch size to prevent input from utilizing too much RAM.
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        # Calculate the target amount of data.
        target_data_size = max(1, math.ceil(percent_data * len(self.test_dataset)))
        # Calculate the number of batches that need to be processed to get the target amount of data.
        num_batches = math.ceil(target_data_size / batch_size)

        # Create an empty tensor that will contain the pChEMBL predictions.
        pchembl_preds = torch.tensor([])
        pchembl_labels = torch.tensor([])
        for batch_num, batch in enumerate(test_loader):
            node_features, edge_features, adjacency_matrix, pchembl_scores = [
                x.to(torch.float32).to("cpu") for x in batch
            ]
            preds = self.model(node_features, edge_features, adjacency_matrix).squeeze(-1)
            pchembl_preds = torch.cat((pchembl_preds, preds))
            pchembl_labels = torch.cat((pchembl_labels, pchembl_scores))

            if batch_num == num_batches - 1:
                break

        # Crop the data to the right size.
        pchembl_preds = pchembl_preds[:target_data_size]
        pchembl_labels = pchembl_labels[:target_data_size]
        return pchembl_preds, pchembl_labels

    def _get_test_dataset(self, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame, seed: int = 42,
                          frac_validation: float = 0.15, frac_test: float = 0.15) -> DrugProteinDataset:
        data_df['stratify_col'] = data_df['Target_ID'] + "_" + data_df['label'].astype(str)
        training_df, remaining_df = train_test_split(data_df,
                                                     test_size=frac_validation + frac_test,
                                                     stratify=data_df['stratify_col'],
                                                     random_state=seed)
        validation_df, test_df = train_test_split(remaining_df,
                                                  test_size=frac_test / (frac_validation + frac_test),
                                                  stratify=remaining_df['stratify_col'],
                                                  random_state=seed)
        test_df = test_df.drop(columns='stratify_col')
        test_dataset = DrugProteinDataset(test_df, protein_embeddings_df)
        return test_dataset


class MoleculeViewer(tk.Frame):
    def __init__(self, root: tk.Tk, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame,
                 model: nn.Module) -> None:
        super().__init__(root)

        # Use a white background.
        self.configure(bg="white")

        # Make the 2nd and 3rd column the same size.
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        # Expand the first (and only) row across the entire window.
        self.grid_rowconfigure(0, weight=1)

        self.dataset = DrugProteinDataset(data_df, protein_embeddings_df)
        self.model = model

        # Create a dictionary that maps the pair of ChEMBL IDs to the index of that entry.
        self.pair_to_idx_mapping = {}
        # Create a dictionary that maps the protein's ChEMBL ID to the drugs tested for interaction with that protein.
        self.protein_to_drugs_mapping = {}

        for idx in data_df.index:
            protein_pchembl_id = data_df.at[idx, 'Target_ID']
            drug_pchembl_id = data_df.at[idx, 'ChEMBL_ID']
            self.pair_to_idx_mapping[(protein_pchembl_id, drug_pchembl_id)] = idx

            if protein_pchembl_id in self.protein_to_drugs_mapping:
                self.protein_to_drugs_mapping[protein_pchembl_id].append(drug_pchembl_id)
            else:
                self.protein_to_drugs_mapping[protein_pchembl_id] = [drug_pchembl_id]

        self._create_settings_frame()
        # Create a figure to show the drug molecule.
        self.fig, self.ax, self.canvas = self._init_canvas(column=1)
        self._create_info_frame()

    def _init_canvas(self, column: int, size: int = 1200, **kwargs) -> tuple[Figure, plt.Axes, FigureCanvasTkAgg]:
        fig = Figure()
        ax = fig.add_subplot()
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)  # Flip y-axis
        ax.set_xticks([])
        ax.set_yticks([])
        # Add a placeholder white image.
        ax.imshow(Image.new('RGB', (size, size), (255, 255, 255)))

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=0, column=column, sticky="nsew", padx=10, pady=10, **kwargs)

        return fig, ax, canvas

    def _create_settings_frame(self) -> None:
        # Use a larger font.
        font = ("Helvetica", 12)
        x_pad = 30

        # Create a left panel that contains various settings that the user can modify.
        self.settings_frame = tk.Frame(self)
        self.settings_frame.grid(row=0, column=0, padx=10, sticky="nsw")

        # Add a spacer at the top.
        tk.Label(self.settings_frame, text="", height=2).pack()

        tk.Label(self.settings_frame, text="Protein ChEMBL ID:", font=font).pack(padx=x_pad)
        self.protein_dropdown = ttk.Combobox(
            self.settings_frame, values=[""] + list(dict.fromkeys(self.dataset.protein_ids)),
            state="readonly", font=font)
        self.protein_dropdown.pack(pady=(5, 20), padx=x_pad)
        self.protein_dropdown.current(0)
        self.protein_dropdown.bind("<<ComboboxSelected>>", self._update_drug_dropdown)

        tk.Label(self.settings_frame, text="Drug ChEMBL ID:", font=font).pack(padx=x_pad)
        self.drug_dropdown = ttk.Combobox(self.settings_frame, state="readonly", font=font)
        self.drug_dropdown.pack(pady=(5, 20), padx=x_pad)
        self._update_drug_dropdown()

        self._create_mode_settings()

        self.submit_button = tk.Button(self.settings_frame, text="Draw Molecule", font=font,
                                       command=self._update_display)
        self.submit_button.pack(padx=x_pad)

    def _create_mode_settings(self):
        # by default, show node contributions.
        self.node_contributions_mode = tk.IntVar(value=1)
        self.functional_groups_mode = tk.IntVar(value=0)

        node_contributions_checkbox = tk.Checkbutton(self.settings_frame, text="Show Node Contributons",
                                                     variable=self.node_contributions_mode,
                                                     command=lambda: self._toggle_mode('node_contributions'))
        node_contributions_checkbox.pack()

        self.node_contributions_intensity_slider = tk.Scale(self.settings_frame, from_=0.1, to=0.7,
                                                            resolution=0.05, orient="horizontal")
        self.node_contributions_intensity_slider.set(0.4)
        self.node_contributions_intensity_slider.pack()

        functional_groups_checkbox = tk.Checkbutton(self.settings_frame, text="Show Functional Groups",
                                                    variable=self.functional_groups_mode,
                                                    command=lambda: self._toggle_mode('functional_groups'))
        functional_groups_checkbox.pack()

        self._create_functional_group_settings()

    def _toggle_mode(self, selected_mode: str) -> None:
        if selected_mode == 'node_contributions':
            self.functional_groups_mode.set(0)
        elif selected_mode == 'functional_groups':
            self.node_contributions_mode.set(0)

    def _create_functional_group_settings(self) -> None:
        self.functional_groups = {'ketone': Ketone(), 'ether': Ether(), 'alcohol': Alcohol(), 'amine': Amine()}
        self.functional_group_colors = {'ketone': "#0000FF", 'ether': "#800080",
                                        'alcohol': "#FFA500", 'amine': "#8B4513"}
        self.functional_group_color_schemes = {'ketone': "Blues", 'ether': "Purples",
                                               'alcohol': "Oranges", 'amine': "copper"}
        self.functional_group_toggle = {}

        for key in self.functional_groups:
            functional_group_obj = self.functional_groups[key]
            color = self.functional_group_colors[key]

            self.functional_group_toggle[key] = tk.IntVar(value=1)
            check_button = tk.Checkbutton(self.settings_frame, text=functional_group_obj.name,
                                          variable=self.functional_group_toggle[key], fg=color)
            check_button.pack()

    def _create_info_frame(self) -> None:
        self.info_frame = tk.Frame(self, width=400, padx=10, pady=10)
        self.info_frame.grid_propagate(False)
        self.info_frame.grid(row=0, column=2, sticky="ns", padx=50, pady=100)

        self._update_info_frame("", "", "")

    def _update_info_frame(self, actual_pchembl_str: str, pred_pchembl_str: str, smiles_str: str) -> None:
        for widget in self.info_frame.winfo_children():
            widget.destroy()

        data = [
            ("Protein ID", self.protein_dropdown.get(), False),
            ("Drug ID", self.drug_dropdown.get(), False),
            ("Drug SMILES String", smiles_str, True),
            ("", "", False),  # Add an extra label for padding
            ("Actual pChEMBL", actual_pchembl_str, False),
            ("Predicted pChEMBL", pred_pchembl_str, False),
        ]

        row = 0
        for label, value, newline in data:
            if newline:
                label = tk.Message(self.info_frame, text=label, font=("Arial", 10, "bold"), anchor="w", width=280)
                label.grid(row=row, column=0, sticky="w", padx=5, pady=(5, 0), columnspan=2)
                value = tk.Message(self.info_frame, text=value, font=("Arial", 10), anchor="w", width=280)
                value.grid(row=row + 1, column=0, sticky="w", padx=5, pady=(0, 5), columnspan=2)
                row += 2
            else:
                label = tk.Message(self.info_frame, text=label, font=("Arial", 10, "bold"), anchor="w", width=140)
                label.grid(row=row, column=0, sticky="w", padx=5, pady=5, columnspan=1)
                value = tk.Message(self.info_frame, text=value, font=("Arial", 10), anchor="w", width=140)
                value.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                row += 1

    def _update_drug_dropdown(self, event: tk.Event = None) -> None:
        selected_protein = self.protein_dropdown.get()
        if selected_protein == "":
            self.drug_dropdown["values"] = [""]
            self.drug_dropdown.current(0)
        else:
            new_drug_options = self.protein_to_drugs_mapping[selected_protein]
            self.drug_dropdown["values"] = [""] + new_drug_options
            if self.drug_dropdown.get() not in new_drug_options:
                self.drug_dropdown.current(0)

    def _update_display(self) -> None:
        protein_chembl_id = self.protein_dropdown.get()
        drug_chembl_id = self.drug_dropdown.get()

        # Can only create a molecule graph if both the protein and the drug are specified.
        if protein_chembl_id == "" or drug_chembl_id == "":
            return

        idx = self.pair_to_idx_mapping[(protein_chembl_id, drug_chembl_id)]
        node_contributions, pred_pchembl = self._get_node_contributions(idx)
        drug_graph = self.dataset.drug_graphs[idx]
        mol = drug_graph.mol
        actual_pchembl = self.dataset.pchembl_scores[idx]
        drug_smiles_str = self.dataset.smiles_strs[idx]

        self._draw_molecule(mol, node_contributions, drug_graph)
        self._update_info_frame(str(round(actual_pchembl, 2)), str(round(pred_pchembl, 2)), drug_smiles_str)

    def _draw_molecule(self, mol: rdkit.Chem.Mol, node_contributions: list[float], drug_graph: DrugMolecule) -> None:
        # Generate 2D coordinates for the molecule.
        AllChem.Compute2DCoords(mol)

        # Create a molecule drawing.
        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert the drawing to a PIL Image.
        png_data = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(png_data))
        self.ax.imshow(image)

        if self.node_contributions_mode.get() == 1:
            self._display_node_contributions(mol, drawer, node_contributions)
        elif self.functional_groups_mode.get() == 1:
            self._display_functional_groups(mol, drawer, drug_graph)

        self.canvas.draw()

    def _display_node_contributions(self, mol: rdkit.Chem.Mol, drawer: rdkit.Chem.Draw.rdMolDraw2D,
                                    node_contributions: list[float]) -> None:
        for i in range(mol.GetNumAtoms()):
            # Get the coordinates for the ith atom.
            x, y = drawer.GetDrawCoords(i)
            # Calculate the scaled contribution of this atom
            scaled_contribution = max(-1.0, min(1.0, node_contributions[i] *
                                                self.node_contributions_intensity_slider.get()))

            color_scheme = 'Reds' if scaled_contribution < 0 else 'Greens'
            # Create a circle around this atom to show its relative contribution to the interaction strength.
            self._create_gradient_circle(x, y, 200, color_scheme, center_alpha=abs(scaled_contribution))

    def _display_functional_groups(self, mol: rdkit.Chem.Mol, drawer: rdkit.Chem.Draw.rdMolDraw2D,
                                   drug_graph: DrugMolecule) -> None:
        for key in self.functional_groups:
            if self.functional_group_toggle[key].get() == 0:
                continue

            color_scheme = self.functional_group_color_schemes[key]
            matches = drug_graph.find_functional_group(self.functional_groups[key])
            for match in matches:
                nodes = set(match.values())
                for node in nodes:
                    x, y = drawer.GetDrawCoords(node)
                    self._create_gradient_circle(x, y, 200, color_scheme)

    def _create_gradient_circle(self, x: int, y: int, radius: int, color_scheme: str,
                                center_alpha: float = 0.3, edge_alpha: float = 0) -> None:
        # Create a grid for the gradient.
        grid_size = 500
        x_vals = np.linspace(x - radius, x + radius, grid_size)
        y_vals = np.linspace(y - radius, y + radius, grid_size)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Compute distances from the center of the circle.
        dist_from_center = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

        # Linearly scale the distances to be from 0 to the radius.
        norm = Normalize(vmin=0, vmax=radius)
        gradient = norm(dist_from_center)
        # The center should have the most intense color, fading outwars from there.
        gradient = np.clip(1 - gradient, 0, 1)

        # Fetch the specified color map.
        color_map = plt.get_cmap(color_scheme)
        rgba_colors = color_map(gradient)

        # Adjust the alpha (opacity) channel to fade out towards the edge.
        alpha_gradient = np.clip(center_alpha * gradient + edge_alpha * (1 - gradient), edge_alpha, center_alpha)
        rgba_colors[..., 3] = alpha_gradient

        # Overlay the gradient circle.
        self.ax.imshow(
            rgba_colors,
            extent=(x - radius, x + radius, y - radius, y + radius),
            origin='lower',
            interpolation='bilinear'
        )

    def _get_node_contributions(self, idx: int) -> tuple[list[float], float]:
        node_features, edge_features, adjacency_matrix, pchembl_score = self.dataset[idx]
        # Count the number of atoms in the drug molecule (excluding atoms that were added for padding).
        num_real_nodes = (adjacency_matrix.sum(dim=1) != 0).sum().item()

        # Add an extra dimension to allow for masking one node in each sample.
        node_features = node_features.unsqueeze(0).repeat(num_real_nodes + 1, 1, 1)
        edge_features = edge_features.repeat(num_real_nodes + 1, 1, 1, 1)
        adjacency_matrix = adjacency_matrix.repeat(num_real_nodes + 1, 1, 1)

        for i in range(num_real_nodes):
            # Mask the node features for the ith node.
            node_features[i, i, :] = 0.0
            # Mask the edge features for edges connecting to the ith node.
            edge_features[i, i, :, :] = 0.0
            edge_features[i, :, i, :] = 0.0
            # Mask the edges connecting to the ith node.
            adjacency_matrix[i, i, :] = 0.0
            adjacency_matrix[i, :, i] = 0.0

        preds = self.model(node_features, edge_features, adjacency_matrix).squeeze(-1).tolist()
        # The real pChEMBL prediction uses the whole drug graph without masking any nodes.
        real_pred = preds.pop()
        node_contributions = []
        for x in preds:
            # The contribution of a node is given by the change in the pChEMBL prediction when that
            # node is masked (e.g. if the prediction drops when the node is masked, then the node has
            # a positive contribution).
            node_contributions.append(real_pred - x)

        return node_contributions, real_pred


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # TODO - change to using filtered_cancer_all.csv dataset
    data_df = pd.read_csv(f'{data_path}/filtered_cancer_small.csv')
    protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)
    return data_df, protein_embeddings_df


if __name__ == '__main__':
    set_seeds()
    app = AnalysisApp('../data')
    app.mainloop()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': [
            'torch',
            'torchvision',
            'numpy',
            'pandas',
            'scikit-learn',
            'rdkit',
            'typing',
            'matplotlib',
            'transformers',
            'tqdm',
            'argparse',
            'tensorboard',
            'jupyterlab',
            'notebook',
            'regex',
            'xgboost',
            'hypothesis',
            'pytest',
            'python-ta~=2.9.1'
        ],
        'allowed-io': [],
        'max-line-length': 120
    })
