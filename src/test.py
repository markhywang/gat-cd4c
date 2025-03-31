import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Sample confusion dictionary
confusion_dict = {
    'true_positive': 10,
    'false_positive': 2,
    'true_negative': 15,
    'false_negative': 5
}


def plot_confusion_matrix(confusion_dict):
    """Plots a confusion matrix with labels and color shading based on proportion."""

    # Extract values from the dictionary
    tp = confusion_dict['true_positive']
    fp = confusion_dict['false_positive']
    tn = confusion_dict['true_negative']
    fn = confusion_dict['false_negative']

    # Construct the confusion matrix
    matrix = np.array([[tp, fp], [fn, tn]])

    # Normalize by total values for color scaling
    matrix_norm = matrix.astype('float') / matrix.sum()

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust size here
    cax = ax.matshow(matrix_norm, cmap='Blues')  # Use 'Blues' colormap for shading

    # Add a color bar
    plt.colorbar(cax)

    # Set tick positions
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Now set labels correctly (2 tick labels for 2 positions)
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])

    # Annotate each cell with its value
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matrix[i, j]}", ha='center', va='center', color='black', fontsize=12)

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Increase title size and add spacing (pad)
    plt.title("Confusion Matrix", fontsize=16, pad=20)

    # Automatically adjust subplots to fit into the figure area
    fig.tight_layout()

    return fig


def embed_plot_in_tkinter():
    """Creates the Tkinter window and embeds the Matplotlib plot inside it."""
    root = tk.Tk()
    root.title("Confusion Matrix in Tkinter")

    # Allow window resizing
    root.geometry("800x600")  # Set the window size
    root.resizable(True, True)  # Allow resizing both horizontally and vertically

    # Create a frame to hold the plot
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Plot the confusion matrix
    fig = plot_confusion_matrix(confusion_dict)

    # Embed the plot into the Tkinter window using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()


# Call the function to display the plot in Tkinter
embed_plot_in_tkinter()
