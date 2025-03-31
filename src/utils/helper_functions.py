import torch
from torch import nn
import torchvision

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
        dir_path (str): target directory

    Returns:
        A print out of:
          - number of subdirectories in dir_path
          - number of images (files) in each subdirectory
          - name of each subdirectory

    Example:
        >> import os
        >> os.makedirs("temp/subdir", exist_ok=True)
        >> with open("temp/subdir/image.png", "w") as f:
        ..     f.write("dummy image")
        >> walk_through_dir("temp")
        There are 1 directories and 0 images in 'temp'
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)

    Args:
        model (torch.nn.Module): The PyTorch model used for prediction.
        X (torch.Tensor): Input features of shape (n_samples, 2).
        y (torch.Tensor): Target labels.

    Returns:
        None
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots linear training data and test data and compares predictions.

    Args:
        train_data: Training data (x values).
        train_labels: Labels corresponding to train_data.
        test_data: Test data (x values).
        test_labels: Labels corresponding to test_data.
        predictions: (Optional) Predicted labels for test_data.

    Returns:
        None

    Example:
        >> import numpy as np
        >> train_data = np.array([1, 2, 3])
        >> train_labels = np.array([2, 4, 6])
        >> test_data = np.array([4, 5])
        >> test_labels = np.array([8, 10])
        >> plot_predictions(train_data, train_labels, test_data, test_labels)
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


def mse_func(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes the Mean Squared Error (MSE) between predictions and true values.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

    Returns:
        float: The mean squared error.

    Examples:
        >>> import torch
        >>> mse_func(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
        0.5
    """
    squared_diff = (y_pred - y_true) ** 2
    return float(torch.mean(squared_diff).item())


def mae_func(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes the Mean Absolute Error (MAE) between predictions and true values.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

    Returns:
        float: The mean absolute error.

    Examples:
        >>> import torch
        >>> mae_func(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
        0.5
    """
    abs_diff = torch.abs(y_pred - y_true)
    return float(torch.mean(abs_diff).item())


def accuracy_func(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float) -> int:
    """Calculates regression accuracy given labels and predictions.

    Args:
        y_pred (torch.Tensor): Predictions to be compared.
        y_true (torch.Tensor): Truth labels for predictions.
        threshold (float): Acceptasble error bound for a prediction to be considered close

    Returns:
        int: Number of close predictions, e.g. 783

    Examples:
        >>> import torch
        >>> y_true = torch.tensor([1.0, 2.0, 3.0])
        >>> y_pred = torch.tensor([1.1, 2.0, 2.9])
        >>> accuracy_func(y_pred, y_true, 0.05)
        tensor(1)
    """
    return (abs(y_true - y_pred) < threshold).sum()


def print_train_time(start, end, device=None):
    """
    Prints the difference between start and end time.

    Args:
        start (float): Start time of computation.
        end (float): End time of computation.
        device: Device that computation is running on (optional).

    Returns:
        float: Time difference in seconds.

    Examples:
        >>> result = print_train_time(1.0, 3.5, 'cpu')
        <BLANKLINE>
        Train time on cpu: 2.500 seconds
        >>> result
        2.5
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def to_cpu_scalar(x):
    """
    Converts a tensor to a CPU scalar, or returns the value if it's not a tensor.

    Args:
        x: A torch.Tensor or a scalar.

    Returns:
        The scalar value of x.

    Examples:
        >>> import torch
        >>> to_cpu_scalar(torch.tensor(5))
        5
        >>> to_cpu_scalar(3.14)
        3.14
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return x


def plot_loss_curves(results):
    """
    Plots training curves from a results dictionary.

    Args:
        results (pd.DataFrame): DataFrame containing 'train_loss', 'train_acc',
                                'validation_loss', and 'validation_acc' columns.

    Returns:
        None

    Example:
        >> import pandas as pd
        >> import torch
        >> data = {'train_loss': [torch.tensor(0.5), torch.tensor(0.4)],
        ..         'validation_loss': [torch.tensor(0.6), torch.tensor(0.5)],
        ..         'train_acc': [torch.tensor(0.8), torch.tensor(0.85)],
        ..         'validation_acc': [torch.tensor(0.75), torch.tensor(0.8)]}
        >> df = pd.DataFrame(data)
        >> plot_loss_curves(df)
    """
    loss = [to_cpu_scalar(x) for x in results["train_loss"].tolist()]
    validation_loss = [to_cpu_scalar(x) for x in results["validation_loss"].tolist()]

    accuracy = [to_cpu_scalar(x) for x in results["train_acc"].tolist()]
    validation_accuracy = [to_cpu_scalar(x) for x in results["validation_acc"].tolist()]

    epochs = range(results.shape[0])

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, validation_loss, label="validation_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, validation_accuracy, label="validation_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): Trained PyTorch image classification model.
        image_path (str): Filepath to target image.
        class_names (List[str], optional): List of class names for target image. Defaults to None.
        transform: Optional transform to be applied to the image.
        device (torch.device, optional): Device for computation. Defaults to "cuda" if available, else "cpu".

    Returns:
        None

    Example usage:
        >> pred_and_plot_image(model=model,
        ..                     image_path="some_image.jpeg",
        ..                     class_names=["class_1", "class_2", "class_3"],
        ..                     transform=torchvision.transforms.ToTensor(),
        ..                     device="cpu")
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for torch operations to ensure reproducibility.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.

    Returns:
        None

    Example:
        >> set_seeds(123)
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def count_model_params(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Counts the total number of parameters in a model.

    Args:
        model (nn.Module): The model whose parameters are to be counted.
        only_trainable (bool, optional): If True, counts only parameters that require gradients. Defaults to False.

    Returns:
        int: Total number of parameters.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 1)
        >>> count_model_params(model)
        11
        >>> model.weight.requires_grad = False
        >>> count_model_params(model, only_trainable=True)
        1
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def plot_preds_vs_labels(preds: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Plots a scatter plot comparing predicted values to true labels and displays a reference line (y = x).

    Args:
        preds (torch.Tensor): Predicted values.
        labels (torch.Tensor): True values.

    Returns:
        None

    Example:
        >> import torch
        >> preds = torch.tensor([2.5, 3.0, 4.1])
        >> labels = torch.tensor([3.0, 3.0, 4.0])
        >> plot_preds_vs_labels(preds, labels)
    """
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(labels, preds, alpha=0.5, label='Predictions')

    # Plot the y = x line
    min_val = min(labels.min(), preds.min())
    max_val = max(labels.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Labels')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_r_squared(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination) between predictions and true labels.

    Args:
        preds (np.ndarray): Predicted values.
        labels (np.ndarray): True values.

    Returns:
        float: The R-squared value.

    Examples:
        >>> import numpy as np
        >>> get_r_squared(np.array([2, 3, 4]), np.array([2, 3, 4]))
        np.float64(1.0)
        >>> get_r_squared(np.array([2, 3, 5]), np.array([2, 3, 4]))
        np.float64(0.5)
    """
    ss_res = np.sum((preds - labels) ** 2)  # Residual sum of squares
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
