import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import cycle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef
from scipy import interp


def plot_confusion_matrix(cm, target_names, title="Confusion matrix", cmap=None, normalize=True):



    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    plt.show()

    return accuracy


def plot_roc(pred_probas, val_gts, class_names, title):

    # Plot linewidth.
    lw = 2
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(val_gts[:, i], pred_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(val_gts.ravel(), pred_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print()

    # Plot all ROC curves
    plt.figure(1, figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
            label="micro-average ROC curve (AUC = {0:0.2f})"
                    "".format(roc_auc["micro"]),
            color="deeppink", linestyle=":", linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label="macro-average ROC curve (AUC = {0:0.2f})"
                    "".format(roc_auc["macro"]),
            color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "grey"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label="ROC curve of class {0} (AUC = {1:0.2f})"
                "".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    print()

    # Zoom in view of the upper left corner.
    plt.figure(2, figsize=(8, 6))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
            label="micro-average ROC curve (AUC = {0:0.2f})"
                    "".format(roc_auc["micro"]),
            color="deeppink", linestyle=":", linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label="macro-average ROC curve (AUC = {0:0.2f})"
                    "".format(roc_auc["macro"]),
            color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "grey"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label="ROC curve of class {0} (AUC = {1:0.2f})"
                "".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def train_curves(x_history, model_name):
    acc = x_history["accuracy"]
    val_acc = x_history["val_accuracy"]

    plt.figure(figsize=(8, 6))
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name)
    plt.show()