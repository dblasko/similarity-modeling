import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import numpy as np
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, title, ax):
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = matrix[1, 1]
    false_positive = matrix[0, 1]
    false_negative = matrix[1, 0]
    true_negative = matrix[0, 0]
    ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.text(0, 0, f"True Neg: {true_negative}", ha="center", va="center", color="black")
    ax.text(
        1, 0, f"False Pos: {false_positive}", ha="center", va="center", color="black"
    )
    ax.text(
        0, 1, f"False Neg: {false_negative}", ha="center", va="center", color="black"
    )
    ax.text(1, 1, f"True Pos: {true_positive}", ha="center", va="center", color="black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)


def plot_global_confusion_matrix(y_true, y_pred, character1, character2):
    # Assuming y_true and y_pred are your true and predicted labels
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Plotting the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues")
    plt.title("General confusion matrix (all classes)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(
        ticks=[0.5, 1.5, 2.5], labels=["No character", character1, character2]
    )  # Adjust ticks for class labels
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["No character", character1, character2])


def plot_precision_recall_curve(y_true, y_pred, title, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ax.step(recall, precision, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)


def plot_roc_curve(y_true, y_pred, title, ax):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
