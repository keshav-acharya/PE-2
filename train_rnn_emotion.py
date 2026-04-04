"""
╔══════════════════════════════════════════════════════════════╗
║   Speech Emotion Recognition — CREMA-D Dataset              ║
║   Model : Bidirectional LSTM (RNN)                           ║
║   Framework : TensorFlow / Keras                             ║
╚══════════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────────────────
 FIRST-TIME SETUP (run these once in your terminal)
──────────────────────────────────────────────────────────────
  pip install tensorflow librosa scikit-learn numpy
  pip install matplotlib seaborn pandas joblib

──────────────────────────────────────────────────────────────
 HOW TO RUN
──────────────────────────────────────────────────────────────
  1. Put this file in the same folder as your "AudioWAV" folder
     (the one you downloaded from Kaggle)
  2. Open a terminal in that folder
  3. Run:  python train_rnn_emotion.py

──────────────────────────────────────────────────────────────
 FILES THIS SCRIPT CREATES (all in the same folder)
──────────────────────────────────────────────────────────────
  emotion_rnn_model.h5   → your saved trained RNN model
  scaler.pkl             → feature scaler (needed to use model later)
  label_encoder.pkl      → label encoder
  training_curves.png    → loss & accuracy plot across epochs
  confusion_matrix.png   → per-emotion performance heatmap
  training_report.html   → full readable report — open in browser!

──────────────────────────────────────────────────────────────
 WHAT IS AN RNN / LSTM? (plain English)
──────────────────────────────────────────────────────────────
  Random Forest: takes a SINGLE snapshot (180 numbers) of audio
  LSTM: reads audio as a SEQUENCE of 128 frames over time
        — like reading a sentence word by word, not all at once

  Bidirectional LSTM reads the audio BOTH forwards and backwards,
  which helps catch patterns regardless of where in the clip they appear.

  Expected accuracy on CREMA-D: 65–80% (better than Random Forest)
"""

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import sys
import time
import glob
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    print(f"TensorFlow version: {tf.__version__}")
    # Reduce TF log noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("\n[ERROR] TensorFlow is not installed.")
    print("  Run:  pip install tensorflow")
    sys.exit(1)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — change these to experiment later
# ══════════════════════════════════════════════════════════════

AUDIO_FOLDER  = "AudioWAV"     # folder with .wav files (next to this script)

# ── Feature extraction ──────────────────────────────────────
N_MFCC        = 40             # MFCC coefficients per frame
MAX_TIMESTEPS = 128            # pad/truncate every file to this many frames
                               # ~3 seconds at 22 kHz with hop_length=512

# ── Model ───────────────────────────────────────────────────
LSTM_UNITS_1  = 128            # units in the first BiLSTM layer
LSTM_UNITS_2  = 64             # units in the second BiLSTM layer
DENSE_UNITS   = 64             # units in the dense hidden layer
DROPOUT_RATE  = 0.3            # dropout fraction (0 = no dropout, 0.5 = heavy)
LEARNING_RATE = 0.001          # Adam optimizer learning rate

# ── Training ─────────────────────────────────────────────────
EPOCHS        = 50             # maximum number of training epochs
BATCH_SIZE    = 32             # samples per gradient update
TEST_SIZE     = 0.20           # 20% of data for testing
VAL_SIZE      = 0.15           # 15% of training data for validation
RANDOM_STATE  = 42

# ── Callbacks ────────────────────────────────────────────────
PATIENCE_ES   = 10             # stop if val_loss doesn't improve for 10 epochs
PATIENCE_LR   = 5              # halve LR if val_loss doesn't improve for 5 epochs
MIN_LR        = 1e-6           # lower bound for learning rate schedule


# ══════════════════════════════════════════════════════════════
#  EMOTION LABEL MAP
#  CREMA-D encodes emotion in the filename — e.g.
#  1001_DFA_ANG_XX.wav → ANG = Angry (3rd part, split by "_")
# ══════════════════════════════════════════════════════════════

EMOTION_MAP = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — SEQUENTIAL MFCC
#
#  Unlike Random Forest (which used the MEAN of MFCCs),
#  an LSTM reads a SEQUENCE of frames:
#
#    Audio file → 128 time frames × 40 MFCC values
#                 shape: (128, 40)
#
#  This preserves HOW the voice changes over time,
#  which is crucial for emotion (e.g. rising pitch = angry/excited)
# ══════════════════════════════════════════════════════════════

def extract_mfcc_sequence(file_path, max_timesteps=MAX_TIMESTEPS, n_mfcc=N_MFCC):
    """
    Load one .wav file and return a 2D array of shape (max_timesteps, n_mfcc).
    Short files are zero-padded; long files are truncated.
    Returns None if the file cannot be read.
    """
    try:
        # Load audio (sr=None = keep original sample rate)
        audio, sr = librosa.load(file_path, sr=None, duration=4.0)

        # Extract MFCCs — result shape: (n_mfcc, n_frames)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=512)

        # Transpose to (n_frames, n_mfcc) — time axis first
        mfccs = mfccs.T  # shape: (n_frames, n_mfcc)

        # Pad or truncate to exactly max_timesteps
        if mfccs.shape[0] < max_timesteps:
            # Pad with zeros at the end
            pad_width = max_timesteps - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode="constant")
        else:
            # Truncate
            mfccs = mfccs[:max_timesteps, :]

        return mfccs  # shape: (max_timesteps, n_mfcc) = (128, 40)

    except Exception as e:
        print(f"  [Warning] Could not process {os.path.basename(file_path)}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  BUILD THE LSTM MODEL
#
#  Architecture explained:
#
#  Input          → (128, 40)   — 128 time frames, 40 MFCC features each
#  BiLSTM 128     → reads sequence forward AND backward, 128 units each direction
#  Dropout 0.3    → randomly turns off 30% of neurons during training
#                   (prevents overfitting — memorising instead of learning)
#  BiLSTM 64      → second LSTM layer for deeper patterns
#  Dropout 0.3
#  Dense 64       → fully connected layer to combine LSTM output
#  Dropout 0.3
#  Dense 6        → one output per emotion class
#  Softmax        → converts scores to probabilities (sum = 1.0)
# ══════════════════════════════════════════════════════════════

def build_model(input_shape, num_classes):
    """
    Build and compile a Bidirectional LSTM model.

    input_shape : (max_timesteps, n_mfcc) e.g. (128, 40)
    num_classes : number of emotion categories (6 for CREMA-D)
    """
    model = keras.Sequential([

        # Input layer — tells Keras the shape of one sample
        layers.Input(shape=input_shape),

        # First Bidirectional LSTM
        # return_sequences=True → passes the full sequence to the next LSTM
        layers.Bidirectional(
            layers.LSTM(LSTM_UNITS_1, return_sequences=True)
        ),
        layers.Dropout(DROPOUT_RATE),

        # Second Bidirectional LSTM
        # return_sequences=False → passes only the final hidden state
        layers.Bidirectional(
            layers.LSTM(LSTM_UNITS_2, return_sequences=False)
        ),
        layers.Dropout(DROPOUT_RATE),

        # Dense hidden layer with ReLU activation
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dropout(DROPOUT_RATE),

        # Output layer — 6 neurons, one per emotion
        # Softmax turns raw scores into probabilities
        layers.Dense(num_classes, activation="softmax"),
    ])

    # Compile: choose optimizer, loss function, and metrics to track
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",   # correct loss for multi-class classification
        metrics=["accuracy"],
    )

    return model


# ══════════════════════════════════════════════════════════════
#  CUSTOM TRAINING CALLBACK — prints a clean progress line per epoch
# ══════════════════════════════════════════════════════════════

class TrainingProgressCallback(keras.callbacks.Callback):
    """Prints one clean line per epoch with loss and accuracy."""

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start  = 0
        self.history_log  = []   # we store log here for the HTML report

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        ep       = epoch + 1
        t_loss   = logs.get("loss",          0)
        t_acc    = logs.get("accuracy",      0)
        v_loss   = logs.get("val_loss",      0)
        v_acc    = logs.get("val_accuracy",  0)
        lr_val   = float(self.model.optimizer.learning_rate)

        # Store for report
        self.history_log.append({
            "epoch": ep, "loss": t_loss, "accuracy": t_acc,
            "val_loss": v_loss, "val_accuracy": v_acc,
            "lr": lr_val, "time_s": duration,
        })

        bar_done = int(ep / self.total_epochs * 20)
        bar      = "█" * bar_done + "░" * (20 - bar_done)

        print(
            f"  Epoch {ep:>3}/{self.total_epochs}  [{bar}]  "
            f"loss {t_loss:.4f}  acc {t_acc:.4f}  "
            f"val_loss {v_loss:.4f}  val_acc {v_acc:.4f}  "
            f"lr {lr_val:.2e}  {duration:.1f}s"
        )


# ══════════════════════════════════════════════════════════════
#  SAVE TRAINING CURVES IMAGE
# ══════════════════════════════════════════════════════════════

def save_training_curves(history, path="training_curves.png"):
    """Plots loss and accuracy over epochs — both train and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training curves — Speech Emotion RNN", fontsize=13, fontweight="bold")

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs_range, history.history["loss"],     label="Train loss",     color="#4C8BF5", lw=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Validation loss", color="#EA4335", lw=2, ls="--")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, history.history["accuracy"],     label="Train accuracy",     color="#34A853", lw=2)
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Validation accuracy", color="#FBBC04", lw=2, ls="--")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
#  SAVE CONFUSION MATRIX IMAGE
# ══════════════════════════════════════════════════════════════

def save_confusion_matrix(y_true_names, y_pred_names, class_names, path="confusion_matrix.png"):
    cm         = confusion_matrix(y_true_names, y_pred_names, labels=class_names)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion matrix — Speech Emotion RNN", fontsize=13, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d",   cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Raw counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Percentage per true class")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
#  GENERATE HTML REPORT
# ══════════════════════════════════════════════════════════════

def generate_html_report(report_data, history_log, path="training_report.html"):
    cr = report_data["classification_report"]

    # Per-class table rows
    per_class_rows = ""
    for label, metrics in cr.items():
        if isinstance(metrics, dict):
            f1    = metrics["f1-score"]
            badge = "good" if f1 >= 0.70 else ("ok" if f1 >= 0.50 else "warn")
            per_class_rows += (
                f"<tr><td><b>{label}</b></td>"
                f"<td>{metrics['precision']:.3f}</td>"
                f"<td>{metrics['recall']:.3f}</td>"
                f"<td><span class='badge {badge}'>{f1:.3f}</span></td>"
                f"<td>{int(metrics['support'])}</td></tr>"
            )

    # Epoch table rows (last 20 epochs)
    epoch_rows = ""
    display_epochs = history_log[-20:] if len(history_log) > 20 else history_log
    for row in display_epochs:
        epoch_rows += (
            f"<tr><td>{row['epoch']}</td>"
            f"<td>{row['loss']:.4f}</td>"
            f"<td>{row['accuracy']:.4f}</td>"
            f"<td>{row['val_loss']:.4f}</td>"
            f"<td>{row['val_accuracy']:.4f}</td>"
            f"<td>{row['lr']:.2e}</td>"
            f"<td>{row['time_s']:.1f}s</td></tr>"
        )
    if len(history_log) > 20:
        epoch_rows = f"<tr><td colspan='7' style='text-align:center;color:#888;font-size:12px'>... showing last 20 of {len(history_log)} epochs ...</td></tr>" + epoch_rows

    # Emotion distribution
    em = report_data["emotion_distribution"]
    em_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in em.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RNN Speech Emotion Recognition — Training Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:#f4f6fb;color:#333;padding:28px}}
  .container{{max-width:980px;margin:0 auto}}
  h1{{font-size:26px;font-weight:700;margin-bottom:4px;color:#1a1a2e}}
  .subtitle{{color:#666;font-size:14px;margin-bottom:32px}}
  .card{{background:#fff;border-radius:14px;padding:26px;
          box-shadow:0 1px 6px rgba(0,0,0,.07);margin-bottom:26px}}
  h2{{font-size:18px;font-weight:600;margin-bottom:16px;color:#222;
       border-bottom:2px solid #f0f0f0;padding-bottom:10px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}}
  .stat{{background:#f8f9ff;border:1px solid #e5e8f5;border-radius:10px;
          padding:18px;text-align:center}}
  .stat .value{{font-size:30px;font-weight:700;color:#4361ee}}
  .stat .label{{font-size:12px;color:#777;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:14px}}
  th{{background:#f7f8fc;text-align:left;padding:10px 14px;
       font-weight:600;border-bottom:2px solid #e5e8f5}}
  td{{padding:9px 14px;border-bottom:1px solid #f3f4f8}}
  tr:last-child td{{border-bottom:none}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500}}
  .good{{background:#e6f9ee;color:#1b7a44}}
  .ok{{background:#fff8e1;color:#b45a00}}
  .warn{{background:#fdecea;color:#9b2020}}
  img{{width:100%;border-radius:10px;border:1px solid #eee;margin-top:8px}}
  .arch-box{{background:#f8f9ff;border:1px solid #e5e8f5;border-radius:10px;
              padding:18px;font-size:13px;line-height:1.8;margin-top:8px}}
  .arch-layer{{display:flex;justify-content:space-between;align-items:center;
                padding:7px 14px;border-radius:7px;margin-bottom:6px;font-size:13px}}
  .al-lstm{{background:#ede9fe;color:#4c1d95}}
  .al-dense{{background:#fef3c7;color:#78350f}}
  .al-out{{background:#d1fae5;color:#064e3b}}
  .al-drop{{background:#f3f4f6;color:#4b5563}}
  .al-in{{background:#dbeafe;color:#1e3a8a}}
  .tip-box{{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
             padding:18px;margin-top:8px;font-size:14px;line-height:1.8}}
  .tip-box ul{{padding-left:22px;margin-top:6px}}
  .tip-box li{{margin-bottom:6px}}
  @media(max-width:640px){{.stat-grid{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="container">

  <h1>Speech Emotion Recognition — RNN Training Report</h1>
  <p class="subtitle">
    Model: Bidirectional LSTM &nbsp;|&nbsp; Dataset: CREMA-D &nbsp;|&nbsp;
    Generated: {report_data['timestamp']}
  </p>

  <!-- Key metrics -->
  <div class="card">
    <h2>Key results</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{report_data['test_accuracy']:.1%}</div>
        <div class="label">Test accuracy</div></div>
      <div class="stat"><div class="value">{report_data['train_accuracy']:.1%}</div>
        <div class="label">Train accuracy</div></div>
      <div class="stat"><div class="value">{report_data['best_val_acc']:.1%}</div>
        <div class="label">Best val accuracy</div></div>
      <div class="stat"><div class="value">{report_data['epochs_trained']}</div>
        <div class="label">Epochs trained</div></div>
      <div class="stat"><div class="value">{report_data['n_samples']:,}</div>
        <div class="label">Total audio files</div></div>
      <div class="stat"><div class="value">{report_data['total_params']:,}</div>
        <div class="label">Model parameters</div></div>
    </div>
  </div>

  <!-- Model architecture -->
  <div class="card">
    <h2>Model architecture</h2>
    <p style="font-size:13px;color:#666;margin-bottom:12px">
      Input shape: ({report_data['max_timesteps']} timesteps × {report_data['n_mfcc']} MFCC features)
    </p>
    <div class="al-in arch-layer"><span>Input layer</span>
      <span>({report_data['max_timesteps']}, {report_data['n_mfcc']})</span></div>
    <div class="al-lstm arch-layer"><span>Bidirectional LSTM — layer 1</span>
      <span>{report_data['lstm1']} units × 2 directions = {report_data['lstm1']*2} total</span></div>
    <div class="al-drop arch-layer"><span>Dropout</span>
      <span>{report_data['dropout']:.0%} of neurons dropped during training</span></div>
    <div class="al-lstm arch-layer"><span>Bidirectional LSTM — layer 2</span>
      <span>{report_data['lstm2']} units × 2 directions = {report_data['lstm2']*2} total</span></div>
    <div class="al-drop arch-layer"><span>Dropout</span>
      <span>{report_data['dropout']:.0%} of neurons dropped during training</span></div>
    <div class="al-dense arch-layer"><span>Dense hidden layer (ReLU)</span>
      <span>{report_data['dense']} units</span></div>
    <div class="al-drop arch-layer"><span>Dropout</span>
      <span>{report_data['dropout']:.0%} of neurons dropped during training</span></div>
    <div class="al-out arch-layer"><span>Dense output layer (Softmax)</span>
      <span>{report_data['num_classes']} units — one per emotion</span></div>
    <div class="arch-box" style="margin-top:12px">
      <b>Total trainable parameters:</b> {report_data['total_params']:,}<br>
      <b>Optimizer:</b> Adam (lr={report_data['initial_lr']})<br>
      <b>Loss function:</b> Categorical Crossentropy<br>
      <b>Callbacks:</b> EarlyStopping (patience={report_data['patience_es']}) ·
                       ReduceLROnPlateau (patience={report_data['patience_lr']})
    </div>
  </div>

  <!-- Training timeline -->
  <div class="card">
    <h2>Training timeline</h2>
    <table>
      <thead><tr><th>Step</th><th>Time taken</th></tr></thead>
      <tbody>
        <tr><td>Feature extraction (all files)</td><td>{report_data['feature_time']:.1f}s</td></tr>
        <tr><td>Preprocessing + splitting</td><td>{report_data['preprocess_time']:.1f}s</td></tr>
        <tr><td>Model training ({report_data['epochs_trained']} epochs)</td><td>{report_data['train_time']:.1f}s</td></tr>
        <tr><td>Evaluation + report</td><td>{report_data['eval_time']:.1f}s</td></tr>
        <tr><td><b>Total time</b></td><td><b>{report_data['total_time']:.1f}s</b></td></tr>
      </tbody>
    </table>
  </div>

  <!-- Epoch log -->
  <div class="card">
    <h2>Epoch-by-epoch log</h2>
    <div style="overflow-x:auto">
    <table>
      <thead><tr><th>Epoch</th><th>Train loss</th><th>Train acc</th>
                 <th>Val loss</th><th>Val acc</th><th>LR</th><th>Time</th></tr></thead>
      <tbody>{epoch_rows}</tbody>
    </table>
    </div>
  </div>

  <!-- Per-class performance -->
  <div class="card">
    <h2>Performance per emotion</h2>
    <table>
      <thead><tr><th>Emotion</th><th>Precision</th><th>Recall</th>
                 <th>F1-Score</th><th>Test samples</th></tr></thead>
      <tbody>{per_class_rows}</tbody>
    </table>
    <p style="font-size:12px;color:#999;margin-top:10px">
      Precision = of all predictions for this emotion, how many were correct.<br>
      Recall = of all real samples of this emotion, how many did the model find.<br>
      F1-Score = harmonic mean of the two (≥0.70 = good, 0.50-0.70 = ok, &lt;0.50 = needs work).
    </p>
  </div>

  <!-- Dataset distribution -->
  <div class="card">
    <h2>Dataset emotion distribution</h2>
    <table>
      <thead><tr><th>Emotion</th><th>File count</th></tr></thead>
      <tbody>{em_rows}</tbody>
    </table>
  </div>

  <!-- Charts -->
  <div class="card">
    <h2>Training curves</h2>
    <p style="font-size:13px;color:#666">
      Left: loss should go down over epochs. Right: accuracy should go up.
      If training is much better than validation, the model is overfitting — try more dropout.
    </p>
    <img src="training_curves.png" alt="Training curves">
  </div>

  <div class="card">
    <h2>Confusion matrix</h2>
    <p style="font-size:13px;color:#666;margin-bottom:8px">
      Diagonal = correct predictions. Off-diagonal = the model confused one emotion for another.
    </p>
    <img src="confusion_matrix.png" alt="Confusion matrix">
  </div>

  <!-- Next steps -->
  <div class="card">
    <h2>What to try next</h2>
    <div class="tip-box">
      <ul>
        <li>If val accuracy is below 65%, try adding more MFCC features (set N_MFCC = 80) or increasing epochs to 100.</li>
        <li>If train accuracy is much higher than val accuracy (overfitting), increase DROPOUT_RATE to 0.4 or 0.5.</li>
        <li>Add delta and delta-delta MFCCs — they capture how the voice is changing, not just its current state.</li>
        <li>Try 1D Convolutional layers before the LSTM — they act as feature detectors and often improve accuracy by 5-8%.</li>
        <li>Data augmentation: add slight pitch shifting or background noise to the training data to make the model more robust.</li>
        <li>Try a Transformer / Attention model (wav2vec2 fine-tuning) for state-of-the-art results on this dataset.</li>
      </ul>
    </div>
  </div>

</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    total_start = time.time()
    print("\n" + "=" * 65)
    print("  Speech Emotion Recognition — Bidirectional LSTM (RNN)")
    print("  Dataset: CREMA-D  |  Framework: TensorFlow / Keras")
    print("=" * 65)

    # ── Check the audio folder ──────────────────────────────────
    if not os.path.isdir(AUDIO_FOLDER):
        print(f"\n[ERROR] Folder '{AUDIO_FOLDER}' not found.")
        print(f"  Current directory: {os.getcwd()}")
        print("  Make sure the 'AudioWAV' folder is next to this script.")
        return

    audio_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.wav")))
    print(f"\n[1/7] Found {len(audio_files):,} audio files in '{AUDIO_FOLDER}'")

    if len(audio_files) == 0:
        print("  No .wav files found! Check the folder path.")
        return

    # ── FEATURE EXTRACTION ─────────────────────────────────────
    print(f"\n[2/7] Extracting sequential MFCC features...")
    print(f"  Each file → shape ({MAX_TIMESTEPS}, {N_MFCC}) = {MAX_TIMESTEPS * N_MFCC} values")
    print(f"  (vs Random Forest which used just 180 mean values)")
    print(f"  This takes longer but gives the model temporal information.\n")

    feat_start   = time.time()
    features_all = []
    labels_all   = []
    skipped      = 0
    emotion_dist = {}

    for i, fp in enumerate(audio_files):
        fname  = os.path.basename(fp)
        parts  = fname.replace(".wav", "").split("_")
        if len(parts) < 3:
            skipped += 1
            continue
        code = parts[2].upper()
        if code not in EMOTION_MAP:
            skipped += 1
            continue

        seq = extract_mfcc_sequence(fp)
        if seq is None:
            skipped += 1
            continue

        emotion_name = EMOTION_MAP[code]
        features_all.append(seq)
        labels_all.append(emotion_name)
        emotion_dist[emotion_name] = emotion_dist.get(emotion_name, 0) + 1

        if (i + 1) % 500 == 0 or (i + 1) == len(audio_files):
            elapsed = time.time() - feat_start
            eta     = elapsed / (i + 1) * (len(audio_files) - i - 1)
            pct     = (i + 1) / len(audio_files) * 100
            print(f"  {i+1:>5}/{len(audio_files)}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    feature_time = time.time() - feat_start
    print(f"\n  Done! {len(features_all):,} files processed, {skipped} skipped  "
          f"({feature_time:.1f}s)")
    print(f"\n  Emotion distribution:")
    for em, cnt in sorted(emotion_dist.items()):
        print(f"    {em:<10}  {cnt:>5} files")

    # ── PREPROCESSING ───────────────────────────────────────────
    print(f"\n[3/7] Preprocessing...")
    pre_start = time.time()

    # X shape: (n_samples, MAX_TIMESTEPS, N_MFCC)
    X = np.array(features_all, dtype=np.float32)
    print(f"  Feature tensor shape: {X.shape}")

    # Encode string labels to integers
    le       = LabelEncoder()
    y_int    = le.fit_transform(labels_all)
    classes  = list(le.classes_)
    n_cls    = len(classes)
    print(f"  Classes ({n_cls}): {classes}")

    # One-hot encode for categorical crossentropy
    y_onehot = keras.utils.to_categorical(y_int, num_classes=n_cls)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test, y_int_train, y_int_test = train_test_split(
        X, y_onehot, y_int,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_int,
    )

    # Validation split from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"  Training   : {len(X_train):,} samples")
    print(f"  Validation : {len(X_val):,} samples")
    print(f"  Test       : {len(X_test):,} samples")

    # Feature scaling — fit on training data, apply to all
    # We reshape to 2D for the scaler, then back to 3D
    n_train, n_steps, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    def scale_3d(X_3d):
        n, t, f = X_3d.shape
        return scaler.transform(X_3d.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    X_train = scale_3d(X_train)
    X_val   = scale_3d(X_val)
    X_test  = scale_3d(X_test)
    print(f"  Features scaled with StandardScaler")

    preprocess_time = time.time() - pre_start

    # ── BUILD MODEL ─────────────────────────────────────────────
    print(f"\n[4/7] Building Bidirectional LSTM model...")
    model = build_model(
        input_shape=(MAX_TIMESTEPS, N_MFCC),
        num_classes=n_cls,
    )
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    # ── CALLBACKS ───────────────────────────────────────────────
    progress_cb = TrainingProgressCallback(total_epochs=EPOCHS)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_ES,
        restore_best_weights=True,  # rewind to best epoch automatically
        verbose=0,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,              # halve the learning rate
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
        verbose=0,
    )

    # ── TRAIN ───────────────────────────────────────────────────
    print(f"\n[5/7] Training the model...")
    print(f"  Max epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}")
    print(f"  EarlyStopping: stops if val_loss doesn't improve for {PATIENCE_ES} epochs")
    print(f"  ReduceLROnPlateau: halves LR if val_loss stalls for {PATIENCE_LR} epochs\n")

    train_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[progress_cb, early_stop, reduce_lr],
        verbose=0,          # suppress default Keras output (we have progress_cb)
        shuffle=True,
    )
    train_time    = time.time() - train_start
    epochs_done   = len(history.history["loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"  Epochs trained: {epochs_done} (max was {EPOCHS})")
    print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc:.1%})")

    # ── EVALUATE ────────────────────────────────────────────────
    print(f"\n[6/7] Evaluating on the test set...")
    eval_start = time.time()

    # Train accuracy (on the final model state)
    _, train_acc_final = model.evaluate(X_train, y_train, verbose=0)

    # Test accuracy
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Detailed predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_int   = np.argmax(y_pred_probs, axis=1)
    y_true_int   = y_int_test

    y_true_names = le.inverse_transform(y_true_int)
    y_pred_names = le.inverse_transform(y_pred_int)

    report_dict = classification_report(
        y_true_names, y_pred_names, output_dict=True
    )
    print(f"\n  Test accuracy : {test_acc:.4f} ({test_acc:.1%})")
    print(f"  Train accuracy: {train_acc_final:.4f} ({train_acc_final:.1%})\n")
    print(classification_report(y_true_names, y_pred_names))

    eval_time = time.time() - eval_start

    # ── SAVE OUTPUTS ────────────────────────────────────────────
    print(f"\n[7/7] Saving model and generating report files...")

    # Keras model
    model.save("emotion_rnn_model.h5")
    print("  Saved: emotion_rnn_model.h5")

    # Scaler and label encoder (needed to use the model on new data)
    joblib.dump(scaler, "scaler.pkl")
    print("  Saved: scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("  Saved: label_encoder.pkl")

    # Training curves
    save_training_curves(history, path="training_curves.png")

    # Confusion matrix
    save_confusion_matrix(
        y_true_names, y_pred_names, class_names=classes,
        path="confusion_matrix.png"
    )

    # HTML report
    total_time = time.time() - total_start
    generate_html_report(
        report_data={
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_accuracy":    test_acc,
            "train_accuracy":   train_acc_final,
            "best_val_acc":     best_val_acc,
            "epochs_trained":   epochs_done,
            "n_samples":        len(X),
            "total_params":     total_params,
            "max_timesteps":    MAX_TIMESTEPS,
            "n_mfcc":           N_MFCC,
            "lstm1":            LSTM_UNITS_1,
            "lstm2":            LSTM_UNITS_2,
            "dense":            DENSE_UNITS,
            "dropout":          DROPOUT_RATE,
            "num_classes":      n_cls,
            "initial_lr":       LEARNING_RATE,
            "patience_es":      PATIENCE_ES,
            "patience_lr":      PATIENCE_LR,
            "feature_time":     feature_time,
            "preprocess_time":  preprocess_time,
            "train_time":       train_time,
            "eval_time":        eval_time,
            "total_time":       total_time,
            "emotion_distribution": {k: v for k, v in sorted(emotion_dist.items())},
            "classification_report": report_dict,
        },
        history_log=progress_cb.history_log,
        path="training_report.html",
    )

    # ── FINAL SUMMARY ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ALL DONE!")
    print("=" * 65)
    print(f"\n  Test accuracy   : {test_acc:.1%}")
    print(f"  Best val acc    : {best_val_acc:.1%}")
    print(f"  Epochs trained  : {epochs_done}")
    print(f"  Total time      : {total_time:.1f}s\n")
    print("  Output files:")
    print("    emotion_rnn_model.h5    → your trained RNN model")
    print("    scaler.pkl              → feature scaler (keep this!)")
    print("    label_encoder.pkl       → maps numbers back to emotion names")
    print("    training_curves.png     → loss & accuracy over epochs")
    print("    confusion_matrix.png    → per-emotion performance")
    print("    training_report.html    → OPEN THIS IN YOUR BROWSER!")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
