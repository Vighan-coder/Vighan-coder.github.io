# ============================================================
#  MediScan — Medical Image Classifier (Chest X-Ray / Pneumonia)
#  Author : Vighan Raj Verma (@Vighan-coder)
#  GitHub : https://github.com/Vighan-coder/MediScan
# ============================================================
#
#  SETUP:
#    pip install tensorflow numpy pandas matplotlib seaborn opencv-python
#                scikit-learn Pillow
#
#  DATASET (recommended):
#    Kaggle: "chest-xray-pneumonia"  (5,863 images, Normal / Pneumonia)
#    Place images as:
#      data/train/NORMAL/
#      data/train/PNEUMONIA/
#      data/test/NORMAL/
#      data/test/PNEUMONIA/
#
#  RUN:
#    python mediscan.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DATA_DIR    = "data"          # set your dataset path here
USE_SYNTH   = not os.path.exists(DATA_DIR)


# ════════════════════════════════════════════════════════════
#  1. SYNTHETIC DATA (if no real dataset)
# ════════════════════════════════════════════════════════════
def make_synthetic_dataset(n_per_class=200):
    """Create random grayscale images as placeholder data."""
    print("[MediScan] Creating synthetic X-ray placeholders …")
    X, y = [], []
    for label in [0, 1]:        # 0=Normal, 1=Pneumonia
        for _ in range(n_per_class):
            img = np.random.randint(30, 220,
                  (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # Simulate lung texture difference
            if label == 1:
                cv2.circle(img,
                    (np.random.randint(40, 180), np.random.randint(40, 180)),
                    np.random.randint(20, 60), (200, 200, 200), -1)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)
    X = np.array(X); y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ════════════════════════════════════════════════════════════
#  2. DATA PIPELINE (real dataset)
# ════════════════════════════════════════════════════════════
def build_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
    )
    test_flow = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )
    return train_flow, test_flow


# ════════════════════════════════════════════════════════════
#  3. MODEL — ResNet-50 Transfer Learning
# ════════════════════════════════════════════════════════════
def build_model(fine_tune_layers=30):
    base = ResNet50(weights="imagenet",
                    include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Freeze all but last N layers
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64,  activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1,  activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )
    print(f"[MediScan] Model params: {model.count_params():,}")
    return model


# ════════════════════════════════════════════════════════════
#  4. TRAINING
# ════════════════════════════════════════════════════════════
def train_model(model, train_data, val_data):
    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                    patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint("mediscan_best.h5", monitor="val_auc",
                                  save_best_only=True, mode="max"),
    ]
    history = model.fit(
        train_data, epochs=EPOCHS,
        validation_data=val_data,
        callbacks=cbs,
        verbose=1,
    )
    return history


def train_on_arrays(model, X, y):
    """Train directly on numpy arrays (synthetic mode)."""
    split = int(len(X) * 0.80)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", patience=4,
                                restore_best_weights=True, mode="max"),
    ]
    history = model.fit(
        X_tr, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=cbs, verbose=1,
    )
    return history, X_val, y_val


# ════════════════════════════════════════════════════════════
#  5. GRAD-CAM
# ════════════════════════════════════════════════════════════
def grad_cam(model, img_array, last_conv_layer="conv5_block3_out"):
    """Generate Grad-CAM heatmap for a single image (H,W,3)."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, 0]

    grads   = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
    cam     = np.maximum(cam, 0)
    cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def visualise_gradcam(model, images, labels, n=4):
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))
    for i in range(n):
        img   = images[i]
        label = int(labels[i])
        try:
            heatmap = grad_cam(model, img)
            overlay = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) / 255.0
            blended = 0.55 * img + 0.45 * overlay
            blended = np.clip(blended, 0, 1)
        except Exception:
            blended = img  # fallback if layer name differs

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{'Pneumonia' if label else 'Normal'}")
        axes[0, i].axis("off")

        axes[1, i].imshow(blended)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.suptitle("MediScan — Grad-CAM Visualisation", fontsize=14)
    plt.tight_layout()
    plt.savefig("mediscan_gradcam.png", dpi=150)
    plt.show()
    print("[Saved] mediscan_gradcam.png")


# ════════════════════════════════════════════════════════════
#  6. TRAINING HISTORY PLOT
# ════════════════════════════════════════════════════════════
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("loss","Loss"),("auc","AUC"),("recall","Recall")]
    for ax, (m, title) in zip(axes, metrics):
        ax.plot(history.history.get(m,[]),     label="Train", color="#7cff67")
        ax.plot(history.history.get(f"val_{m}",[]), label="Val", color="#B19EEF")
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
    plt.suptitle("MediScan — Training History")
    plt.tight_layout()
    plt.savefig("mediscan_history.png", dpi=150)
    plt.show()
    print("[Saved] mediscan_history.png")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = build_model()

    if USE_SYNTH:
        print("[MediScan] No dataset found — using synthetic images.")
        X, y = make_synthetic_dataset(n_per_class=150)
        history, X_val, y_val = train_on_arrays(model, X, y)
        y_pred_proba = model.predict(X_val).flatten()
        y_pred       = (y_pred_proba > 0.5).astype(int)
        print("\n── Evaluation ──────────────────────────────────────")
        print(classification_report(y_val, y_pred,
                                    target_names=["Normal","Pneumonia"]))
        print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
        visualise_gradcam(model, X_val[:4], y_val[:4])
    else:
        train_flow, test_flow = build_generators()
        history = train_model(model, train_flow, test_flow)
        y_pred_proba = model.predict(test_flow).flatten()
        y_true       = test_flow.classes
        y_pred       = (y_pred_proba > 0.5).astype(int)
        print("\n── Evaluation ──────────────────────────────────────")
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal","Pneumonia"]))
        print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
        # Visualise Grad-CAM on a few test images
        sample_imgs = np.array([test_flow[0][0][i] for i in range(4)])
        sample_lbls = y_true[:4]
        visualise_gradcam(model, sample_imgs, sample_lbls)

    plot_history(history)
    model.save("mediscan_final.h5")
    print("[Saved] mediscan_final.h5")
    print("\n[MediScan] Done!")