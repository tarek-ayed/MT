import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def print_metrics(
    dataset, predictions_scores, outlier_detection_name, y_true, all_preds
):
    print(
        f"Accuracy with {outlier_detection_name}: {accuracy_score(all_preds, y_true):.4f}"
    )
    print(f"F1 Score with {outlier_detection_name}: {f1_score(all_preds, y_true):.4f}")
    print(
        f"Precision with {outlier_detection_name}: {precision_score(all_preds, y_true):.4f}"
    )
    print(
        f"Recall with {outlier_detection_name}: {recall_score(all_preds, y_true):.4f}"
    )
    auc_score, precision_at_recall_objective = None, None
    if len(predictions_scores[outlier_detection_name]) > 0:
        all_scores = np.concatenate(predictions_scores[outlier_detection_name], axis=0)
        auc_score = handle_roc(dataset, outlier_detection_name, y_true, all_scores)
        precision_at_recall_objective = handle_prc(
            outlier_detection_name, y_true, all_scores
        )
    print("------------------------------------------")
    return auc_score, precision_at_recall_objective


def handle_roc(dataset, outlier_detection_name, y_true, all_scores):
    try:
        auc_score = roc_auc_score(y_true, all_scores)
        print(f"ROC AUC with {outlier_detection_name}: {auc_score:.4f}")
        fpr, tpr, _ = roc_curve(y_true, all_scores)
        plt.clf()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.title(f"ROC Curve using {outlier_detection_name} on {dataset}")
        plt.savefig(f"ROC_{outlier_detection_name}")
        return auc_score
    except ValueError:
        print(f"Unable to compute ROC Curve for {outlier_detection_name}")


def handle_prc(outlier_detection_name, y_true, all_scores, objective=0.8):
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, all_scores)
        precision_at_recall_objective = precisions[
            next(i for i, value in enumerate(recalls) if value < objective)
        ]
        recall_at_precision_objective = recalls[
            next(i for i, value in enumerate(precisions) if value > objective)
        ]
        print(f"Precision for recall={objective}: {precision_at_recall_objective:.4f}")
        print(f"Recall for precision={objective}: {recall_at_precision_objective:.4f}")
        return precision_at_recall_objective
    except ValueError:
        print(f"Unable to PRC Curve for {outlier_detection_name}")
