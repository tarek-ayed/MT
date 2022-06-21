import streamlit as st
from torch.utils.data import DataLoader
import torch
import numpy as np

from utils.config import DATASETS, OUTLIER_DETECTION_METHODS

st.set_page_config("Outlier Detection", ":eyeglasses:", layout="wide")
st.title("FSL Outlier Detection Demo")

DATASET = "CUB"
MODEL_PATH = "models/resnet18_pt_CUB_1st"
METHODS_TO_EXCLUDE = ["DBSCAN", "EllipticEnvelope", "EllipticEnvelopePCA"]


def badge(value, color: str = "green", label: str = "") -> str:
    """Create a markdown badge using shields.io"""
    return f"![{value}](https://img.shields.io/badge/{label}-{value}-{color})"


dataset = DATASETS[DATASET]()

with st.sidebar:
    kept_methods = [
        method
        for method in list(OUTLIER_DETECTION_METHODS)
        if method not in METHODS_TO_EXCLUDE
    ]
    outlier_detection_method_name = st.selectbox(
        "Outlier Detection Method",
        kept_methods,
        index=kept_methods.index("IsolationForest"),
    )
    proportion_outliers = (
        st.selectbox("Percentage of Outliers", [10 * k for k in range(1, 11)]) / 100
    )
    k_shot = st.selectbox("k-shot", [5, 10, 30], index=1)
    num_classes = st.selectbox("Number of classes", [1, 2, 3, 5, 10])

    class_choice_col1, class_choice_col2 = st.columns(2)
    with class_choice_col1:
        class_name = st.text_input("Inlier class:")
    with class_choice_col2:
        pass
    class_indices = [
        k for k in dataset.labels if dataset.class_names[k].split(".")[-1] == class_name
    ]
    if class_indices == []:
        class_indices = None

    sample_button = st.button("Sample")


if "image_indices" not in st.session_state:
    (
        st.session_state.image_indices,
        st.session_state.outlier_labels,
    ) = dataset.sample_class_with_outliers(
        proportion_outliers=proportion_outliers,
        num_classes=num_classes,
        limit_num_samples=k_shot,
        class_indices=class_indices,
    )
loader = DataLoader(
    dataset,
    sampler=st.session_state.image_indices,  # image indices
    batch_size=64,
)


if sample_button:
    (
        st.session_state.image_indices,
        st.session_state.outlier_labels,
    ) = dataset.sample_class_with_outliers(
        proportion_outliers=proportion_outliers,
        num_classes=num_classes,
        limit_num_samples=k_shot,
    )

button_show_pred = st.button("Compute Outlier Prediction")
if "predicted_scores" not in st.session_state or button_show_pred:
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    features_backbone_list = []
    for imgs, _ in loader:
        features_backbone_list.append(model.backbone(imgs))
    features_backbone = torch.cat(features_backbone_list)
    detect_outliers = OUTLIER_DETECTION_METHODS[outlier_detection_method_name]
    _, st.session_state.predicted_scores = detect_outliers(features_backbone)


cols = st.columns(4)


counter = 0

for i, (item, outlier_label) in enumerate(
    zip(st.session_state.image_indices, st.session_state.outlier_labels)
):
    counter += 1
    with cols[counter % 4]:
        img, label = dataset[item]
        class_name = dataset.class_names[label].split(".")[-1]
        img_numpy = img.permute((1, 2, 0)).numpy()
        img_std = img_numpy - img_numpy.min()
        img_std = img_std / img_std.max()
        st.image(img_std, use_column_width="auto")
        predicted_scores = np.array(st.session_state.predicted_scores)
        predicted_scores = predicted_scores - predicted_scores.min()
        predicted_scores = predicted_scores / predicted_scores.max()
        if button_show_pred:
            value_for_color = int(255 * (2 * predicted_scores[i] - 1))
            if value_for_color > 0:
                value = hex(value_for_color)
                string_color = f"{value[2:]}0000".upper()
            else:
                value = hex(-value_for_color)
                string_color = f"00{value[2:]}00".upper()
            prediction_badge = badge(
                "Prediction", label=str(predicted_scores[i])[:4], color=string_color
            )
        else:
            prediction_badge = ""
        if outlier_label:
            # st.markdown("<font color=‘red’>Outlier</font>", unsafe_allow_html=True)
            st.markdown(badge(class_name, color="red") + "    " + prediction_badge)
        else:
            st.markdown(badge(class_name) + "    " + prediction_badge)
        st.text("")
