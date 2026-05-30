"""
train_model.py
==============
Waste classifier - model training and evaluation.

  - MobileNetV2 as base (transfer learning, ImageNet weights)
  - Phase 1: train only the new classification head (base frozen)
  - Phase 2: fine-tune the top layers of the base model
  - Augmentation on train set, normalization only on val/test
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  - Accuracy and loss curves (train vs val)
  - Confusion matrix (saved as PNG)
  - Per-class precision, recall, F1-score
  - All plots and reports saved to ./output/

Requirements:
    pip install tensorflow scikit-learn matplotlib seaborn numpy Pillow

Usage:
    python train_model.py
    python train_model.py --data ./dataset_split --epochs1 15 --epochs2 20
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed, saves to file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "data_dir":       "./dataset_split",
    "output_dir":     "./output",
    "model_path":     "./output/model_final.keras",
    "best_ckpt_path": "./output/model_best.keras",

    "img_size":       224,
    "batch_size":     32,

    # Phase 1: train head only (base frozen)
    "epochs_phase1":  15,
    # Phase 2: fine-tune top layers of base
    "epochs_phase2":  50,
    # How many layers from the end of MobileNetV2 to unfreeze in phase 2
    "unfreeze_layers": 40,

    # Learning rates
    "lr_phase1": 1e-3,
    "lr_phase2": 1e-5,   # must be very small for fine-tuning

    # ImageNet normalization values
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_preprocess_fn(mean, std):
    """
    Returns a preprocessing function for ImageDataGenerator.
    Normalizes pixel values using ImageNet mean and std.
    This matches the statistics used when MobileNetV2 was pre-trained.
    """
    mean_arr = np.array(mean, dtype="float32")
    std_arr  = np.array(std,  dtype="float32")

    def preprocess(x):
        x = x.astype("float32") / 255.0
        return (x - mean_arr) / std_arr

    return preprocess

def make_train_preprocess_fn(mean, std):
    """Like make_preprocess_fn, but applies production-like augmentations
       BEFORE ImageNet normalization. Used only for the train generator."""
    import io
    from PIL import Image, ImageFilter

    mean_arr = np.array(mean, dtype="float32")
    std_arr  = np.array(std,  dtype="float32")

    def preprocess(x):
        # x: float32 HxWx3 in [0, 255] (after ImageDataGenerator ops)
        img = np.clip(x, 0, 255).astype("uint8")
        pil = Image.fromarray(img)

        # 1) Gaussian blur, sigma 0–3, p=0.4
        if np.random.random() < 0.4:
            sigma = np.random.uniform(0.0, 3.0)
            if sigma > 0.1:
                pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        # 2) JPEG quality 30–100, p=0.5
        if np.random.random() < 0.5:
            q = int(np.random.uniform(30, 100))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            pil = Image.open(buf); pil.load()

        # 3) Hue/Saturation jitter, p=0.4
        if np.random.random() < 0.4:
            hsv = pil.convert("HSV")
            h, s, v = hsv.split()
            h_arr = (np.array(h, dtype="int16") + np.random.randint(-12, 13)) % 256
            s_arr = np.clip(np.array(s, dtype="int16") + np.random.randint(-25, 26), 0, 255)
            pil = Image.merge("HSV", (
                Image.fromarray(h_arr.astype("uint8"), "L"),
                Image.fromarray(s_arr.astype("uint8"), "L"),
                v,
            )).convert("RGB")

        x = np.array(pil, dtype="float32")

        # 4) Gaussian noise, p=0.3
        if np.random.random() < 0.3:
            std_n = np.random.uniform(2.0, 10.0)
            x = np.clip(x + np.random.normal(0, std_n, x.shape), 0, 255)

        # 5) Random erasing, p=0.25
        if np.random.random() < 0.25:
            h, w = x.shape[:2]
            area = np.random.uniform(0.02, 0.15) * h * w
            ar = np.random.uniform(0.3, 3.3)
            eh = int(np.sqrt(area * ar)); ew = int(np.sqrt(area / ar))
            if 0 < eh < h and 0 < ew < w:
                t = np.random.randint(0, h - eh); l = np.random.randint(0, w - ew)
                x[t:t+eh, l:l+ew, :] = np.random.uniform(0, 255, (eh, ew, 3))

        # ImageNet normalization
        x = x / 255.0
        return (x - mean_arr) / std_arr

    return preprocess

def build_generators(data_dir, batch_size, img_size, mean, std):
    """
    Build ImageDataGenerators for train, val, and test.
    Train uses augmentation; val and test use normalization only.
    """
    # Import here so the script fails early if TF is not installed
    from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

    preprocess_fn = make_preprocess_fn(mean, std)
    train_preprocess_fn = make_train_preprocess_fn(mean, std)

    train_datagen = ImageDataGenerator(
        preprocessing_function=train_preprocess_fn,
        rotation_range=35,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.10,
        zoom_range=[0.75, 1.25],
        horizontal_flip=True,
        brightness_range=[0.70, 1.30],
        channel_shift_range=15.0,
        fill_mode="reflect",
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    size = (img_size, img_size)

    train_gen = train_datagen.flow_from_directory(
        str(data_dir / "train"),
        target_size=size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = eval_datagen.flow_from_directory(
        str(data_dir / "val"),
        target_size=size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        str(data_dir / "test"),
        target_size=size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,   # must be False so label order matches predictions
    )

    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes, img_size, lr):
    """
    MobileNetV2 base + custom classification head.

    Architecture:
        MobileNetV2 (frozen, ImageNet weights, no top layer)
        -> GlobalAveragePooling2D
        -> Dense(256, relu)  + BatchNorm + Dropout(0.4)
        -> Dense(128, relu)  + BatchNorm + Dropout(0.3)
        -> Dense(num_classes, softmax)

    BatchNorm helps stabilize training when the base is partially frozen.
    Two Dense layers give the head enough capacity for 10 classes.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model # type: ignore
    from tensorflow.keras.applications import MobileNetV2 # type: ignore

    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # frozen for phase 1

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model, base


def unfreeze_top_layers(model, base, n_layers, lr):
    """
    Unfreeze the last n_layers of the base model for fine-tuning.
    Use a very small learning rate to avoid destroying pre-trained weights.
    """
    import tensorflow as tf

    base.trainable = True

    # Freeze everything except the last n_layers
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print("  Unfrozen base layers: " + str(trainable_count))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(best_ckpt_path, phase_label, initial_best=None):
    from tensorflow.keras.callbacks import ( # type: ignore
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
    )
    from pathlib import Path

    Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    log_path = str(Path(best_ckpt_path).parent / ("log_" + phase_label + ".csv"))

    return [
        ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            initial_value_threshold=initial_best,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=7,          # stop if no improvement for 7 epochs
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,          # multiply LR by 0.3 on plateau
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(log_path, append=False),
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history1, history2, output_dir):
    """Plot accuracy and loss curves for both training phases."""

    def merge(key):
        v1 = history1.history.get(key, [])
        v2 = history2.history.get(key, [])
        return v1 + v2

    epochs_total = len(merge("accuracy"))
    phase2_start = len(history1.history.get("accuracy", []))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=14)

    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "loss",     "Loss"),
    ]:
        train_vals = merge(metric)
        val_vals   = merge("val_" + metric)
        xs = range(1, len(train_vals) + 1)

        ax.plot(xs, train_vals, label="train",      color="#2E86AB")
        ax.plot(xs, val_vals,   label="validation", color="#E84855", linestyle="--")
        ax.axvline(x=phase2_start + 0.5, color="#888", linestyle=":", linewidth=1.2,
                   label="fine-tune start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(output_dir / "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved: " + path)


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix as a heatmap."""

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (test set)", fontsize=13)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = str(output_dir / "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved: " + path)


def plot_per_class_metrics(report_dict, class_names, output_dir):
    """Bar chart of precision, recall and F1-score per class."""

    precision = [report_dict[c]["precision"] for c in class_names]
    recall    = [report_dict[c]["recall"]    for c in class_names]
    f1        = [report_dict[c]["f1-score"]  for c in class_names]

    x = np.arange(len(class_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 5))
    ax.bar(x - width, precision, width, label="Precision", color="#2E86AB", alpha=0.85)
    ax.bar(x,         recall,    width, label="Recall",    color="#E84855", alpha=0.85)
    ax.bar(x + width, f1,        width, label="F1-score",  color="#3BB273", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Metrics (test set)", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = str(output_dir / "per_class_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved: " + path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_gen, class_names, output_dir):
    """
    Run evaluation on the test set:
    - Overall accuracy and loss
    - Confusion matrix
    - Classification report (precision, recall, F1 per class)
    Saves all results to output_dir.
    """
    print("\n>> Evaluating on test set...")

    # Get predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Overall metrics
    loss, acc = model.evaluate(test_gen, verbose=0)
    print("  Test accuracy: " + str(round(acc * 100, 2)) + "%")
    print("  Test loss:     " + str(round(loss, 4)))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, output_dir)

    # Classification report
    report_str  = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        digits=4, output_dict=True)

    print("\n  Per-class results:")
    print(report_str)

    # Save report to text file
    report_path = str(output_dir / "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test accuracy: " + str(round(acc * 100, 2)) + "%\n")
        f.write("Test loss:     " + str(round(loss, 4)) + "\n\n")
        f.write(report_str)
    print("  Saved: " + report_path)

    # Per-class metrics bar chart
    plot_per_class_metrics(report_dict, class_names, output_dir)

    # Save raw metrics as JSON (useful for later comparison between experiments)
    metrics_path = str(output_dir / "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": round(acc, 6),
            "test_loss":     round(loss, 6),
            "per_class":     {c: report_dict[c] for c in class_names},
            "macro_avg":     report_dict.get("macro avg", {}),
            "weighted_avg":  report_dict.get("weighted avg", {}),
        }, f, indent=2)
    print("  Saved: " + metrics_path)

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Waste classifier - train and evaluate")
    parser.add_argument("--data",     default=CONFIG["data_dir"],    help="Path to dataset_split/")
    parser.add_argument("--output",   default=CONFIG["output_dir"],  help="Output directory for model and plots")
    parser.add_argument("--epochs1",  type=int, default=CONFIG["epochs_phase1"],  help="Epochs for phase 1 (head training)")
    parser.add_argument("--epochs2",  type=int, default=CONFIG["epochs_phase2"],  help="Epochs for phase 2 (fine-tuning)")
    parser.add_argument("--batch",    type=int, default=CONFIG["batch_size"])
    parser.add_argument("--unfreeze", type=int, default=CONFIG["unfreeze_layers"], help="Layers to unfreeze in phase 2")
    args = parser.parse_args()

    set_seeds(CONFIG["random_seed"])

    data_dir   = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = str(output_dir / "model_best.keras")
    final_model_path = str(output_dir / "model_final.keras")

    print("")
    print("  Data:    " + str(data_dir.resolve()))
    print("  Output:  " + str(output_dir.resolve()))
    print("  Batch:   " + str(args.batch))
    print("  Phase 1 epochs: " + str(args.epochs1))
    print("  Phase 2 epochs: " + str(args.epochs2))
    print("")

    # -----------------------------------------------------------------------
    # 1. Build data generators
    # -----------------------------------------------------------------------
    print(">> Building data generators...")
    train_gen, val_gen, test_gen = build_generators(
        data_dir,
        args.batch,
        CONFIG["img_size"],
        CONFIG["imagenet_mean"],
        CONFIG["imagenet_std"],
    )

    num_classes  = train_gen.num_classes
    class_names  = list(train_gen.class_indices.keys())

    print("  Classes found: " + str(num_classes))
    print("  " + str(class_names))
    print("  Train batches: " + str(len(train_gen)))
    print("  Val   batches: " + str(len(val_gen)))
    print("  Test  batches: " + str(len(test_gen)))

    # Save class index mapping for use in the API later
    class_map_path = str(output_dir / "class_indices.json")
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print("  Class map saved -> " + class_map_path)

    # Compute class weights to handle any imbalance in the training set
    counts = np.bincount(train_gen.classes, minlength=num_classes)
    raw = counts.sum() / (num_classes * counts)
    cw_values = np.sqrt(raw)
    cw_values = cw_values / cw_values.mean()    # normalize so mean weight = 1.0
    class_weight_dict = {i: float(w) for i, w in enumerate(cw_values)}
    print("  Class weights (sqrt-balanced): " + str({k: round(v, 2) for k, v in class_weight_dict.items()}))

    # -----------------------------------------------------------------------
    # 2. Build model
    # -----------------------------------------------------------------------
    print("\n>> Building model (MobileNetV2 + custom head)...")
    model, base_model = build_model(
        num_classes,
        CONFIG["img_size"],
        CONFIG["lr_phase1"],
    )
    model.summary()

    # -----------------------------------------------------------------------
    # 3. Phase 1 - train head only (base frozen)
    # -----------------------------------------------------------------------
    print("\n>> Phase 1 - training head only (base frozen)...")
    print("   LR = " + str(CONFIG["lr_phase1"]))

    history1 = model.fit(
        train_gen,
        epochs=args.epochs1,
        validation_data=val_gen,
        callbacks=make_callbacks(best_ckpt, "phase1"),
        class_weight=class_weight_dict,
        verbose=1,
    )

    phase1_best_val_loss = min(history1.history["val_loss"])
    print("  Phase 1 best val_loss: " + str(round(phase1_best_val_loss, 4)))

    # -----------------------------------------------------------------------
    # 4. Phase 2 - fine-tune top layers
    # -----------------------------------------------------------------------
    print("\n>> Phase 2 - fine-tuning top " + str(args.unfreeze) + " layers of base...")
    print("   LR = " + str(CONFIG["lr_phase2"]))

    unfreeze_top_layers(model, base_model, args.unfreeze, CONFIG["lr_phase2"])

    history2 = model.fit(
        train_gen,
        epochs=args.epochs2,
        validation_data=val_gen,
        callbacks=make_callbacks(best_ckpt, "phase2", initial_best=phase1_best_val_loss),
        class_weight=class_weight_dict,
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # 5. Save final model
    # -----------------------------------------------------------------------
    print("\n>> Saving final model -> " + final_model_path)
    model.save(final_model_path)

    # -----------------------------------------------------------------------
    # 6. Plot training curves
    # -----------------------------------------------------------------------
    print("\n>> Saving training curves...")
    plot_history(history1, history2, output_dir)

    # -----------------------------------------------------------------------
    # 7. Evaluate on test set (Step 3)
    # -----------------------------------------------------------------------
    
    # Load model_best.keras from disk - this is the model that goes to production
    import tensorflow as tf
    print("\n>> Loading model_best.keras for evaluation...")
    best_model = tf.keras.models.load_model(best_ckpt)

    print(">> Evaluating: " + best_ckpt)
    test_acc = evaluate_model(best_model, test_gen, class_names, output_dir)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print("\n" + "=" * 52)
    print("  DONE")
    print("=" * 52)
    print("  Best model test accuracy: " + str(round(test_acc * 100, 2)) + "%")
    print("  Best checkpoint model saved (production one):     " + best_ckpt)
    print("  Final model saved:         " + final_model_path)
    print("  Outputs in:          " + str(output_dir.resolve()))
    print("")
    print("  Files produced:")
    print("    model_best.keras           - best checkpoint (val_loss) - this is the one that goes on production")
    print("    model_final.keras          - final model")
    print("    class_indices.json         - class name -> index mapping")
    print("    training_curves.png        - accuracy and loss plots")
    print("    confusion_matrix.png       - confusion matrix heatmap")
    print("    per_class_metrics.png      - precision / recall / F1 per class")
    print("    classification_report.txt  - full text report")
    print("    metrics.json               - raw metrics for later comparison")
    print("")
    print("  Next step -> build REST API")
    print("")


if __name__ == "__main__":
    main()
