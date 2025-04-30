import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors

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
from sklearn.metrics import roc_curve, auc
import sys
import os
import argparse

# Import local code from src directory.
sys.path.append(os.path.abspath("src"))
from utils.dataset import DrugProteinDataset, DrugMolecule
from utils.helper_functions import set_seeds, get_r_squared
from utils.functional_groups import *
from train import collate_drug_prot
from model import DualGraphAttentionNetwork
from config import final_model


class AnalysisApp(tk.Tk):
    """Tkinter window to display analysis of dataset and model.

    Instance Attributes:
        - model: nn.Module for the Graph Attention Network that has already been trained
        - notebook: ttk.Notebook that tracks the tabs in the window
    """
    model: nn.Module
    notebook: ttk.Notebook

    def __init__(self, data_path: str) -> None:
        super().__init__()
        super().title("CandidateDrug4Cancer Analysis")

        args_dict={
            "use_small_dataset": False,
            "batch_size": 48,
            "stoppage_epochs": 64,
            "max_epochs": 512,
            "seed": 0,
            "data_path": "../data",
            "protein_graph_dir": "../data/protein_graphs",
            "frac_train": 0.8,
            "frac_validation": 0.1,
            "frac_test": 0.1,
            "huber_beta": 1.0,
            "weight_decay": 1e-3,
            "lr": 1e-4,
            "scheduler_patience": 16,
            "scheduler_factor": 0.5,
            "hidden_size": 128,
            "emb_size": 128,
            "num_layers": 4,
            "num_attn_heads": 8,
            "dropout": 0.1,
            "mlp_dropout": 0.2,
            "pooling_dim": 128,
            "mlp_hidden": 192,
            "max_nodes": 80, # Max number of amino acids
        }
        
        args = argparse.Namespace(**args_dict)
        data_df, protein_embeddings_df = self.load_data(data_path)

        self.model = DualGraphAttentionNetwork(
            drug_in_features=29,
            prot_in_features=1283,
            hidden_size=args.hidden_size,
            emb_size=getattr(args, "emb_size", args.hidden_size),
            drug_edge_features=17,
            prot_edge_features=1,
            num_layers=args.num_layers,
            num_heads=args.num_attn_heads,
            dropout=args.dropout,
            mlp_dropout=args.mlp_dropout,
            pooling_dim=args.pooling_dim,
            mlp_hidden=getattr(args, "mlp_hidden", 128),
            device="cpu"
        ).to(torch.float32).to("cpu")
        self.model.load_state_dict(
            torch.load("models/model.pth", weights_only=False, map_location=torch.device('cpu')))
        self.model.eval()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        # Increase the font size and padding for the tabs.
        ttk.Style().configure("TNotebook.Tab", padding=[10, 5], font=("Arial", 10))

        molecule_viewer = MoleculeViewer(self, data_df, protein_embeddings_df, self.model, max_nodes=args.max_nodes)
        self.notebook.add(molecule_viewer, text="Molecule Viewer")

        model_analysis = ModelBenchmark(self, data_df, protein_embeddings_df, self.model)
        self.notebook.add(model_analysis, text="Model Benchmark")

    def load_data(self, data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the drug-protein interaction data from the specified directory."""
        data_df = pd.read_csv(f'{data_path}/filtered_cancer_all.csv')
        protein_embeddings_df = pd.read_csv(f'{data_path}/protein_embeddings.csv', index_col=0)
        return data_df, protein_embeddings_df


class ModelBenchmark(tk.Frame):
    """Tkinter frame to visually represent the results of the model benchmark.

    Instance Attributes:
        - fig_size: Tuple of width and height for plotted figures (controls resolution)
        - canvas_size: Tuple of width and height for the canvas used to display the figures
        - model: nn.Module for the Graph Attention Network that has already been trained
        - data_df: pd.DataFrame containing the drug-protein data
        - protein_embeddings_df: pd.DataFrame with the protein embeddings
        - plots: Dictionary that maps plot names to their corresponding Matplotlib plot
        - canvases: Dictionary that maps canvas names to their correspond Tkinter canvas
        - percent_data_slider: A Tkinter Scale widget to control the percentage of test data used for benchmarking
    """
    fig_size: tuple[float, float]
    canvas_size: tuple[int, int]
    model: nn.Module
    data_df: pd.DataFrame
    protein_embeddings_df: pd.DataFrame
    plots: dict[str, plt.Figure | None]
    canvases: dict[str, FigureCanvasTkAgg | None]
    percent_data_slider: tk.Scale

    def __init__(self, root: tk.Tk, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame, model: nn.Module,
                 fig_size: tuple[float, float] = (4.5, 5), canvas_size: tuple[int, int] = (360, 400)) -> None:
        super().__init__(root)

        # Use a white background.
        self.configure(bg="white")

        self.fig_size = fig_size
        self.canvas_size = canvas_size

        self.model = model
        self.data_df = data_df
        self.protein_embeddings_df = protein_embeddings_df
        self._create_settings_frame()

        self.plots = {'confusion': None, 'scatter': None, 'auc_roc': None}
        self.canvases = {'confusion': None, 'scatter': None, 'auc_roc': None}
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event) -> None:
        """Close all Matplotlib plots when the frame is destroyed."""
        self._close_plots()

    def _close_plots(self):
        """Close all Matplotlib plots to free system resources."""
        if self.plots['confusion'] is not None:
            plt.close(self.plots['confusion'])
        if self.plots['scatter'] is not None:
            plt.close(self.plots['scatter'])
        if self.plots['auc_roc'] is not None:
            plt.close(self.plots['auc_roc'])

    def _update_display(self) -> None:
        """Update the display with three plots (confusion matrix, scatter plot, and ROC curve)."""
        test_dataset = self._get_test_dataset(self.data_df, self.protein_embeddings_df)
        pchembl_preds, pchembl_labels = self._eval_model(self.percent_data_slider.get(), test_dataset)

        self._close_plots()

        self._update_confusion_matrix(pchembl_preds, pchembl_labels)
        self._update_scatter(pchembl_preds, pchembl_labels)
        self._update_auc_roc(pchembl_preds, pchembl_labels)

    def _create_settings_frame(self):
        """Create a settings frame with options that the user can interact with."""
        settings_frame = tk.Frame(self, padx=5, pady=5)

        # Create a slider that controls the percentage of test data that is used to do the analysis.
        tk.Label(settings_frame, text="% Data Used for Analysis").grid(row=0, column=0, padx=20, pady=(20, 0))
        # Create a slider that controls the percentage of test data that is used to do the analysis.
        self.percent_data_slider = tk.Scale(settings_frame, from_=0.05, to=1.0, resolution=0.05, orient="horizontal")
        self.percent_data_slider.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        generate_button = tk.Button(settings_frame, text="Generate Analysis", command=self._update_display)
        generate_button.grid(row=0, column=1, rowspan=2, padx=20, pady=20)

        settings_frame.grid(row=0, column=0, padx=20, pady=(20, 0), sticky='w')

    def _update_confusion_matrix(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor) -> None:
        """Create and display the confusion matrix."""
        confusion_dict = self._get_confusion_dict(pchembl_preds, pchembl_labels)
        self.plots['confusion'], precision, recall, accuracy = self._plot_confusion_matrix(confusion_dict)

        # Embed the plot into the Tkinter frame.
        self.canvases['confusion'] = FigureCanvasTkAgg(self.plots['confusion'], master=self)
        self.canvases['confusion'].draw()
        self.canvases['confusion'].get_tk_widget().config(width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvases['confusion'].get_tk_widget().grid(row=1, column=0, padx=25, pady=5)

        # Add a label showing the relevant numerical metrics.
        tk.Label(self, text=f"Precision = {precision:.1%}, Recall = {recall:.1%}, Accuracy = {accuracy:.1%}",
                 font=("Arial", 12, "bold"), bg="white", wraplength=300).grid(row=2, column=0)

    def _plot_confusion_matrix(self, confusion_dict: dict[str, float]) -> tuple[plt.Figure, float, float, float]:
        """Create a figure of the confusion matrix and return relevant numerical metrics."""
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
        fig, ax = plt.subplots(figsize=(self.fig_size[0], self.fig_size[1]))

        # Add a color bar on the right side that uses different shades of blue.
        cax = ax.matshow(matrix_norm, cmap='Blues')
        # Shrink the color bar to better match the size of the confusion matrix.
        plt.colorbar(cax, shrink=0.5)

        # Set tick positions and labels.
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=8)
        ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=8)

        # Annotate each cell with its value.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{matrix[i, j]}", ha='center', va='center', color='black', fontsize=12)

        plt.title("Confusion Matrix", fontsize=16, pad=40)

        # Automatically adjust subplots to fit into the figure area.
        fig.tight_layout()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return fig, precision, recall, accuracy

    def _get_confusion_dict(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor,
                            pchembl_threshold: float = 7.0) -> dict[str, int]:
        """Get a dictionary with the confusion matrix."""
        positive_preds = [x >= pchembl_threshold for x in pchembl_preds]
        positive_labels = [x >= pchembl_threshold for x in pchembl_labels]
        confusion_dict = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}

        for pred, label in zip(positive_preds, positive_labels):
            pred_str = 'positive' if pred else 'negative'
            is_correct = 'true' if pred == label else 'false'
            confusion_dict[f"{is_correct}_{pred_str}"] += 1

        return confusion_dict

    def _update_scatter(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor):
        """Create and display the scatter plot."""
        self.plots['scatter'], r_squared, percent_close_preds = self._plot_scatter(pchembl_preds, pchembl_labels)

        # Embed the plot into the Tkinter frame.
        self.canvases['scatter'] = FigureCanvasTkAgg(self.plots['scatter'], master=self)
        self.canvases['scatter'].draw()
        self.canvases['scatter'].get_tk_widget().config(width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvases['scatter'].get_tk_widget().grid(row=1, column=1, padx=25, pady=5)

        # Add a label showing the relevant numerical metrics.
        tk.Label(self, text=f"R² = {r_squared:.3f}\n% Close Predictions = {percent_close_preds:.1%}",
                 font=("Arial", 12, "bold"), bg="white", wraplength=300).grid(row=2, column=1)

    def _plot_scatter(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor) \
            -> tuple[plt.Figure, float, float]:
        """Create a figure of the scatter plot and return relevant numerical metrics."""
        preds = pchembl_preds.cpu().detach().numpy()
        labels = pchembl_labels.cpu().detach().numpy()
        r_squared = get_r_squared(preds, labels)
        percent_close_preds = np.mean(np.abs(preds - labels) <= 1)

        # Create the plot.
        fig, ax = plt.subplots(figsize=(self.fig_size[0], self.fig_size[1]))

        # Reduce point size from 10 to 1.
        plt.scatter(labels, preds, alpha=0.5, label='Predictions', s=1)
        plt.title("pChEMBL Predictions", fontsize=16, pad=20)

        min_val = min(labels.min(), preds.min())
        max_val = max(labels.max(), preds.max())

        # Plot a red highlight over all points that are within 1 of the y = x line.
        close_pred_fill = np.linspace(min_val, max_val, 2)
        plt.fill_between(close_pred_fill, close_pred_fill - 1, close_pred_fill + 1, color='red', alpha=0.1,
                         label='Close Predictions (±1)')

        # Plot the y = x line.
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        plt.xlabel('Actual pChEMBL Score')
        plt.ylabel('Predicted pChEMBL Score')
        plt.legend()
        plt.grid(True)

        # Automatically adjust subplots to fit into the figure area.
        fig.tight_layout()

        return fig, r_squared, percent_close_preds

    def _update_auc_roc(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor):
        """Create and display the ROC curve."""
        self.plots['auc_roc'], auc_roc_score = self._plot_auc_roc(pchembl_preds, pchembl_labels)

        # Embed the plot into the Tkinter frame.
        self.canvases['auc_roc'] = FigureCanvasTkAgg(self.plots['auc_roc'], master=self)
        self.canvases['auc_roc'].draw()
        self.canvases['auc_roc'].get_tk_widget().config(width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvases['auc_roc'].get_tk_widget().grid(row=1, column=2, padx=25, pady=5)

        # Add a label showing the relevant numerical metrics.
        tk.Label(self, text=f"AUC-ROC = {auc_roc_score:.3f}", font=("Arial", 12, "bold"),
                 bg="white", wraplength=300).grid(row=2, column=2)

    def _plot_auc_roc(self, pchembl_preds: torch.Tensor, pchembl_labels: torch.Tensor, pchembl_threshold: float = 7.0) \
            -> tuple[plt.Figure, float]:
        """Create a figure of the ROC curve and return relevant numerical metrics."""
        preds = pchembl_preds.cpu().tolist()
        labels = [x >= pchembl_threshold for x in pchembl_labels]

        # Compute ROC curve and area under curve.
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_roc_score = auc(fpr, tpr)

        # Create the plot
        fig, ax = plt.subplots(figsize=(self.fig_size[0], self.fig_size[1]))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='y = x')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title("AUC-ROC Curve", fontsize=16, pad=20)
        ax.legend(loc='lower right')
        ax.grid(True)

        fig.tight_layout()

        return fig, auc_roc_score

    def _eval_model(self, percent_data, test_dataset: DrugProteinDataset, batch_size: int = 50) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the model on the test data to get pChEMBL predictions."""
        # build DataLoader exactly like training
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_drug_prot(
                b,
                "data/protein_graphs",  # your protein_graph_dir
                80,                     # hard_limit (must match max_nodes)
                17                      # num_edge_feats
            )
        )

        target_data_size = max(1, math.ceil(percent_data * len(test_dataset)))
        num_batches = math.ceil(target_data_size / batch_size)

        pchembl_preds = torch.tensor([], device="cpu")
        pchembl_labels = torch.tensor([], device="cpu")
        with torch.no_grad():
            for batch_num, batch in enumerate(test_loader):
                # collate_drug_prot returns:
                #   drug_ns, drug_es, drug_as,
                #   prot_ns, prot_es, prot_as,
                #   labels
                d_n, d_e, d_a, p_n, p_e, p_a, labels = batch

                # move everything to float
                d_n, d_e, d_a = d_n.float(), d_e.float(), d_a.float()
                p_n, p_e, p_a = p_n.float(), p_e.float(), p_a.float()
                labels = labels.float()

                # forward pass
                preds = self.model(d_n, d_e, d_a, p_n, p_e, p_a).squeeze(-1)

                # accumulate on CPU
                pchembl_preds = torch.cat((pchembl_preds, preds.cpu()))
                pchembl_labels = torch.cat((pchembl_labels, labels.cpu()))

                if batch_num == num_batches - 1:
                    break

        # trim to exact size
        return pchembl_preds[:target_data_size], pchembl_labels[:target_data_size]

    def _get_test_dataset(self, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame, seed: int = 0,
                          frac_validation: float = 0.1, frac_test: float = 0.1) -> DrugProteinDataset:
        """Get the full test dataset using the splits used for training."""
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
        test_dataset = DrugProteinDataset(test_df, protein_embeddings_df, "data/protein_graphs")
        return test_dataset


class MoleculeViewer(tk.Frame):
    """Tkinter frame to visually display the drug molecule.

    Instance Attributes:
        - dataset: Dataset object with pChEMBL interaction data
        - model: nn.Module for the Graph Attention Network that has already been trained
        - pair_to_idx_mapping: Dictionary that maps a pair of drug/protein pChEMBL IDs to their corresponding
          index in the dataset
        - protein_to_drugs_mapping: Dictionary that maps a protein pChEMBL ID to the drugs it has been tested with
        - fig: Matplolib figure used to display the drug molecule
        - ax: Axes for fig
        - tk_widgets: Dictionary that maps the name of a widget to its correspond Tkinter widget
        - functional_groups: Dictionary that maps the name of a functional group to an object
    """
    dataset: DrugProteinDataset
    model: nn.Module
    pair_to_idx_mapping: dict[tuple[str, str], int]
    protein_to_drugs_mapping: dict[str, list[str]]
    fig: plt.Figure
    ax: plt.Axes
    tk_widgets: dict[str, Any]
    functional_groups: dict[str, FunctionalGroup]

    def __init__(self, root: tk.Tk, data_df: pd.DataFrame, protein_embeddings_df: pd.DataFrame,
                 model: nn.Module, max_nodes: int) -> None:
        super().__init__(root)
        self.max_nodes = max_nodes
        # Use a white background.
        self.configure(bg="white")

        # Make the 2nd and 3rd column the same size.
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        # Expand the first (and only) row across the entire window.
        self.grid_rowconfigure(0, weight=1)

        self.dataset = DrugProteinDataset(data_df, protein_embeddings_df, "data/protein_graphs")
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

        self.tk_widgets = {}
        self._create_settings_frame()
        # Create a figure to show the drug molecule.
        self.fig, self.ax, self.tk_widgets['canvas'] = self._init_canvas()
        self._create_info_frame()

    def _init_canvas(self, size: int = 1200, **kwargs) -> tuple[Figure, plt.Axes, FigureCanvasTkAgg]:
        """Initialize an empty canvas which will later be used to display the drug molecule."""
        fig = Figure()
        ax = fig.add_subplot()
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)  # Flip y-axis
        ax.set_xticks([])
        ax.set_yticks([])
        # Add a placeholder white image.
        ax.imshow(Image.new('RGB', (size, size), (255, 255, 255)))

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=10, pady=10, **kwargs)

        return fig, ax, canvas

    def _create_settings_frame(self) -> None:
        """Create a settings frame with options that the user can interact with."""
        # Use a larger font.
        font = ("Arial", 12)
        x_pad = 30

        # Create a left panel that contains various settings that the user can modify.
        self.tk_widgets['settings_frame'] = tk.Frame(self)
        self.tk_widgets['settings_frame'].grid(row=0, column=0, padx=10, sticky="nsw")

        # Add a spacer at the top.
        tk.Label(self.tk_widgets['settings_frame'], text="", height=2).pack()

        tk.Label(self.tk_widgets['settings_frame'], text="Protein ChEMBL ID:", font=font).pack(padx=x_pad)
        self.tk_widgets['protein_dropdown'] = ttk.Combobox(
            self.tk_widgets['settings_frame'], values=[""] + list(dict.fromkeys(self.dataset.prot_ids)),
            state="readonly", font=font)
        self.tk_widgets['protein_dropdown'].pack(pady=(5, 20), padx=x_pad)
        self.tk_widgets['protein_dropdown'].current(0)
        self.tk_widgets['protein_dropdown'].bind("<<ComboboxSelected>>", self._update_drug_dropdown)

        tk.Label(self.tk_widgets['settings_frame'], text="Drug ChEMBL ID:", font=font).pack(padx=x_pad)
        self.tk_widgets['drug_dropdown'] = ttk.Combobox(self.tk_widgets['settings_frame'], state="readonly", font=font)
        self.tk_widgets['drug_dropdown'].pack(pady=(5, 20), padx=x_pad)
        self._update_drug_dropdown()

        self._create_mode_settings()

        self.tk_widgets['submit_button'] = tk.Button(self.tk_widgets['settings_frame'], text="Draw Molecule",
                                                     font=font, command=self._update_display)
        self.tk_widgets['submit_button'].pack(padx=x_pad, pady=20)

    def _create_mode_settings(self):
        """Create a frame that controls which mode to use (node contributions mode or functional groups mode)."""
        # Create variables to track which mode is currently selected.
        self.tk_widgets['node_contributions_mode'] = tk.IntVar()
        self.tk_widgets['functional_groups_mode'] = tk.IntVar()

        node_contributions_checkbox = tk.Checkbutton(
            self.tk_widgets['settings_frame'], text="Show Node Contributons",
            variable=self.tk_widgets['node_contributions_mode'],
            command=lambda: self._toggle_mode('node_contributions'))
        node_contributions_checkbox.pack(pady=(20, 0))

        tk.Label(self.tk_widgets['settings_frame'], text="Display Intensity").pack()
        self.tk_widgets['node_contributions_intensity_slider'] = tk.Scale(
            self.tk_widgets['settings_frame'], from_=0.1, to=0.7, resolution=0.05, orient="horizontal")
        self.tk_widgets['node_contributions_intensity_slider'].set(0.4)
        self.tk_widgets['node_contributions_intensity_slider'].pack()

        functional_groups_checkbox = tk.Checkbutton(
            self.tk_widgets['settings_frame'],
            text="Show Functional Groups",
            variable=self.tk_widgets['functional_groups_mode'],
            command=lambda: self._toggle_mode('functional_groups'),
        )

        functional_groups_checkbox.pack(pady=(20, 0))

        self._create_functional_group_settings()

    def _toggle_mode(self, selected_mode: str) -> None:
        """Update checkboxes to ensure that both modes aren't selected simultaneously."""
        if selected_mode == 'node_contributions':
            self.tk_widgets['functional_groups_mode'].set(0)
        elif selected_mode == 'functional_groups':
            self.tk_widgets['node_contributions_mode'].set(0)

    def _create_functional_group_settings(self) -> None:
        """Create the settings that allow the user to select which functional groups to highlight."""
        self.functional_groups = {'ketone': Ketone(), 'ether': Ether(), 'alcohol': Alcohol(), 'amine': Amine()}

        for key in self.functional_groups:
            functional_group_obj = self.functional_groups[key]
            color = functional_group_obj.color

            self.tk_widgets[key + '_toggle'] = tk.IntVar(value=1)
            check_button = tk.Checkbutton(self.tk_widgets['settings_frame'], text=functional_group_obj.name,
                                          variable=self.tk_widgets[key + '_toggle'], fg=color)
            check_button.pack()

    def _create_info_frame(self) -> None:
        """Create a frame the contains information about the drug molecule being displayed."""
        self.tk_widgets['info_frame'] = tk.Frame(self, width=400, padx=10, pady=10)
        self.tk_widgets['info_frame'].grid_propagate(False)
        self.tk_widgets['info_frame'].grid(row=0, column=2, sticky="ns", padx=50, pady=100)

        self._update_info_frame("", "", "")

    def _update_info_frame(self, actual_pchembl_str: str, pred_pchembl_str: str, smiles_str: str) -> None:
        """Populate the information frame given the user's inputs."""
        for widget in self.tk_widgets['info_frame'].winfo_children():
            widget.destroy()

        data = [
            ("Protein ID", self.tk_widgets['protein_dropdown'].get(), False),
            ("Drug ID", self.tk_widgets['drug_dropdown'].get(), False),
            ("Drug SMILES String", smiles_str, True),
            ("", "", False),  # Add an extra label for padding
            ("Actual pChEMBL", actual_pchembl_str, False),
            ("Predicted pChEMBL", pred_pchembl_str, False),
        ]

        row = 0
        for label, value, newline in data:
            if newline:
                label = tk.Message(self.tk_widgets['info_frame'], text=label, font=("Arial", 10, "bold"), anchor="w", width=280)
                label.grid(row=row, column=0, sticky="w", padx=5, pady=(5, 0), columnspan=2)
                value = tk.Message(self.tk_widgets['info_frame'], text=value, font=("Arial", 10), anchor="w", width=280)
                value.grid(row=row + 1, column=0, sticky="w", padx=5, pady=(0, 5), columnspan=2)
                row += 2
            else:
                label = tk.Message(self.tk_widgets['info_frame'], text=label, font=("Arial", 10, "bold"), anchor="w", width=140)
                label.grid(row=row, column=0, sticky="w", padx=5, pady=5, columnspan=1)
                value = tk.Message(self.tk_widgets['info_frame'], text=value, font=("Arial", 10), anchor="w", width=140)
                value.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                row += 1

    def _update_drug_dropdown(self, event: tk.Event = None) -> None:
        """Filter the drug dropdown to only include molecules that have been tested with the selected protein."""
        selected_protein = self.tk_widgets['protein_dropdown'].get()
        if selected_protein == "":
            self.tk_widgets['drug_dropdown']["values"] = [""]
            self.tk_widgets['drug_dropdown'].current(0)
        else:
            new_drug_options = self.protein_to_drugs_mapping[selected_protein]
            self.tk_widgets['drug_dropdown']["values"] = [""] + new_drug_options
            if self.tk_widgets['drug_dropdown'].get() not in new_drug_options:
                self.tk_widgets['drug_dropdown'].current(0)

    def _update_display(self) -> None:
        """Update the display to show the specified drug-protein interaction."""
        protein_chembl_id = self.tk_widgets['protein_dropdown'].get()
        drug_chembl_id = self.tk_widgets['drug_dropdown'].get()

        # Can only create a molecule graph if both the protein and the drug are specified.
        if protein_chembl_id == "" or drug_chembl_id == "":
            return

        idx = self.pair_to_idx_mapping[(protein_chembl_id, drug_chembl_id)]
        # 1) get the masked‐node contributions and the masked‐out prediction
        node_contributions, pred_pchembl, actual_pchembl = self._get_node_contributions(idx)

        # 2) rebuild the RDKit Mol from the SMILES in data_df
        drug_smiles_str = self.data_df.loc[idx, 'smiles']
        dm = DrugMolecule(drug_smiles_str, max_nodes=self.max_nodes)
        mol = dm.mol
        self._draw_molecule(mol, node_contributions, drug_graph)
        self._update_info_frame(str(round(actual_pchembl, 2)), str(round(pred_pchembl, 2)), drug_smiles_str)

    def _draw_molecule(self, mol: rdkit.Chem.Mol, node_contributions: list[float], drug_graph: DrugMolecule) -> None:
        """Draw the drug molecule on the canvas."""
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

        if self.tk_widgets['node_contributions_mode'].get() == 1:
            self._display_node_contributions(mol, drawer, node_contributions)
        elif self.tk_widgets['functional_groups_mode'].get() == 1:
            self._display_functional_groups(drawer, drug_graph)

        self.tk_widgets['canvas'].draw()

    def _display_node_contributions(self, mol: rdkit.Chem.Mol, drawer: rdkit.Chem.Draw.rdMolDraw2D,
                                    node_contributions: list[float]) -> None:
        """Highlight atoms on the drug graph based on their contribution to the interaction."""
        for i in range(mol.GetNumAtoms()):
            # Get the coordinates for the ith atom.
            x, y = drawer.GetDrawCoords(i)
            # Calculate the scaled contribution of this atom
            scaled_contribution = max(-1.0, min(1.0, node_contributions[i] *
                                                self.tk_widgets['node_contributions_intensity_slider'].get()))

            color_scheme = 'Reds' if scaled_contribution < 0 else 'Greens'
            # Create a circle around this atom to show its relative contribution to the interaction strength.
            self._create_gradient_circle(x, y, 200, color_scheme, center_alpha=abs(scaled_contribution))

    def _display_functional_groups(self, drawer: rdkit.Chem.Draw.rdMolDraw2D, drug_graph: DrugMolecule) -> None:
        """Highlight the functional groups present in the drug molecule."""
        for key in self.functional_groups:
            if self.tk_widgets[key + '_toggle'].get() == 0:
                continue

            functional_group_obj = self.functional_groups[key]
            color = functional_group_obj.color
            matches = drug_graph.find_functional_group(functional_group_obj)
            for match in matches:
                nodes = match.values()
                for node in nodes:
                    x, y = drawer.GetDrawCoords(node)
                    self._create_uniform_circle(x, y, 150, color, 0.2)

    def _create_gradient_circle(self, x: int, y: int, radius: int, color_scheme: str,
                                center_alpha: float = 0.3, edge_alpha: float = 0) -> None:
        """Create a circle on the canvas with a gradient that is more intense towards the center."""
        # Create a grid for the gradient.
        grid_size = 500
        x_vals = np.linspace(x - radius, x + radius, grid_size)
        y_vals = np.linspace(y - radius, y + radius, grid_size)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Compute distances from the center of the circle.
        dist_from_center = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

        # Linearly scale the distances to be from 0 to the radius.
        norm = mcolors.Normalize(vmin=0, vmax=radius)
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

    def _create_uniform_circle(self, x: int, y: int, radius: int, color: str, alpha: float) -> None:
        """Create a circle on the canvas with a uniform opacity."""
        # Convert color string to RGB tuple.
        rgb = mcolors.to_rgb(color)

        # Create a grid for the gradient.
        grid_size = 500
        x_vals = np.linspace(x - radius, x + radius, grid_size)
        y_vals = np.linspace(y - radius, y + radius, grid_size)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Compute distances from the center of the circle.
        dist_from_center = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

        # Linearly scale the distances to be from 0 to the radius.
        norm = mcolors.Normalize(vmin=0, vmax=radius)
        gradient = norm(dist_from_center)
        # The center should have the most intense color, fading outwards.
        gradient = np.clip(1 - gradient, 0, 1)

        # Create an RGBA array filled with the specified color.
        rgba_colors = np.zeros((grid_size, grid_size, 4))
        rgba_colors[..., :3] = np.array(rgb)  # Apply the RGB color
        rgba_colors[..., 3] = gradient * alpha  # Apply the alpha fading

        # Overlay the uniform gradient circle.
        self.ax.imshow(
            rgba_colors,
            extent=(x - radius, x + radius, y - radius, y + radius),
            origin='lower',
            interpolation='bilinear'
        )

    def _get_node_contributions(self, idx: int) -> tuple[list[float], float]:
        """
        Evaluate the model to determine which atoms contribute most to the interaction.
        Returns (node_contributions, real_prediction).
        """
        import torch
        from torch.nn.functional import pad as torch_pad

        # 1) Unpack full dataset entry
        d_n, d_e, d_a, p_n, p_e, p_i, lbl = self.dataset[idx]
        # d_n: [N_drug, F_n], d_e: [N_drug, N_drug, F_e], d_a: [N_drug, N_drug]
        # p_n: [N_prot, F_n'], p_e: [E_prot, F_e'], p_i: [2, E_prot], lbl: scalar

        # 2) Count real drug atoms (non-padded rows in adjacency)
        num_real_drug = (d_a.sum(dim=1) != 0).sum().item()

        # 3) Build masked‐drug batch of size B = num_real_drug + 1
        B = num_real_drug + 1
        drug_ns = d_n.unsqueeze(0).repeat(B, 1, 1)          # [B, N_drug, F_n]
        drug_es = d_e.unsqueeze(0).repeat(B, 1, 1, 1)       # [B, N_drug, N_drug, F_e]
        drug_as = d_a.unsqueeze(0).repeat(B, 1, 1)          # [B, N_drug, N_drug]

        for i in range(num_real_drug):
            # mask out node i and its incident edges
            drug_ns[i, i, :] = 0.0
            drug_es[i, i, :, :] = 0.0
            drug_es[i, :, i, :] = 0.0
            drug_as[i, i, :] = 0.0
            drug_as[i, :, i] = 0.0

        # 4) Build **dense** protein inputs (pad/truncate to match model's max_nodes)
        H_drug = d_n.size(0)   # your model uses fixed H = max_nodes for both graphs
        # helper to pad a single tensor to shape (H, dim)
        def pad_to(x, shape):
            pad = []
            for cur, tgt in zip(reversed(x.shape), reversed(shape)):
                pad += [0, tgt - cur]
            return torch.pad(x, pad)

        # pad node features to [H, F_n'] and tile B times → [B, H, F_n']
        prot_ns = pad_to(p_n, (H_drug, p_n.size(1))).unsqueeze(0).repeat(B, 1, 1)

        # build dense adjacency [H, H] and edge_attrs [H, H, F_e']
        prot_as = torch.zeros((H_drug, H_drug), dtype=torch.float32)
        prot_es = torch.zeros((H_drug, H_drug, p_e.size(-1)), dtype=torch.float32)
        for j in range(p_i.size(1)):
            i0, i1 = int(p_i[0, j]), int(p_i[1, j])
            if i0 < H_drug and i1 < H_drug:
                prot_as[i0, i1] = 1.0
                prot_es[i0, i1] = p_e[j]
        # tile to [B, H, H] and [B, H, H, F_e']
        prot_as = prot_as.unsqueeze(0).repeat(B, 1, 1)
        prot_es = prot_es.unsqueeze(0).repeat(B, 1, 1, 1)

        # 5) Run model on both drug **and** protein batches
        with torch.no_grad():
            preds = self.model(
                drug_ns, drug_es, drug_as,
                prot_ns, prot_es, prot_as
            ).squeeze(-1).tolist()  # length B

        # last entry is the “unmasked” prediction
        real_pred = preds.pop()

        # 6) Contribution of node i = real_pred – pred_with_node_i_masked
        node_contributions = [real_pred - masked_pred for masked_pred in preds]

        return node_contributions, real_pred

if __name__ == '__main__':
    set_seeds(seed=0)
    app = AnalysisApp('data')
    app.mainloop()

    import python_ta
    # Did not work for us (maybe PythonTA has a bug)
    # AttributeError: 'ClassDef' object has no attribute 'value'. Did you mean: 'values'?
    python_ta.check_all(config={
        'extra-imports': [
            'tkinter',
            'tkinter.ttk',
            'matplotlib.pyplot',
            'matplotlib.figure',
            'matplotlib.backends.backend_tkagg',
            'matplotlib.colors',
            'rdkit.Chem',
            'rdkit.Chem.AllChem',
            'rdkit.Chem.Draw.rdMolDraw2D',
            'torch',
            'torch.nn',
            'torch.utils.data',
            'pandas',
            'numpy',
            'math',
            'io',
            'PIL.Image',
            'sklearn.model_selection',
            'sklearn.metrics',
            'sys',
            'os',
            'argparse',
            'config',
            'model',
            'utils.dataset',
            'utils.helper_functions',
            'utils.functional_groups'
        ],
        'disable': [],
        'allowed-io': ['load_data'],
        'max-line-length': 120,
    })