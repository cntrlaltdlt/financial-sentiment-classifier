"""
Component 5: Dashboard
Tkinter GUI that ties the components together: choose a CSV, train, then analyse text.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .data_collection import (check_trainable, clean_data, compute_class_weights,
                              load_csv_data, make_label_maps, split_data)
from .model_training import create_loader, fit, get_device, set_seed, setup_training
from .model_validation import print_validation_metrics
from .models import TextDataset
from .prediction import predict_sentiment

SEED = 42
EPOCHS = 3
VAL_SIZE = 0.2


class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analyser")
        self.geometry("900x640")
        self.configure(bg="#f5f5f5")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)

        # Model state
        self.model = None
        self.tokeniser = None
        self.id2label = None
        self.device = get_device()
        self.max_len = 128

        main_frame = ttk.Frame(self, padding=15)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(main_frame, text="Sentiment Analyser", style="Header.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(main_frame, text="Train on your CSV and analyse the sentiment of new text.").grid(
            row=0, column=0, sticky="e")

        controls = ttk.Frame(main_frame)
        controls.grid(row=1, column=0, sticky="nsew")
        controls.grid_columnconfigure(0, weight=1)

        # 1. Training data
        ttk.Label(controls, text="1. Training data (CSV):").grid(row=0, column=0, sticky="w", pady=(0, 2))
        csv_frame = ttk.Frame(controls)
        csv_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        csv_frame.grid_columnconfigure(0, weight=1)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.csv_path_var, state="readonly").grid(
            row=0, column=0, sticky="ew", padx=(0, 5))
        self.browse_button = ttk.Button(csv_frame, text="Browse…", command=self.browse_csv)
        self.browse_button.grid(row=0, column=1, sticky="e")

        col_frame = ttk.Frame(controls)
        col_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        col_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(col_frame, text="Text column:").grid(row=0, column=0, sticky="w")
        ttk.Label(col_frame, text="Label column:").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.text_col_var = tk.StringVar(value="text")
        self.label_col_var = tk.StringVar(value="label")
        ttk.Entry(col_frame, textvariable=self.text_col_var).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Entry(col_frame, textvariable=self.label_col_var).grid(row=1, column=1, sticky="ew", padx=(5, 0))

        self.train_button = ttk.Button(controls, text="Train Model", command=self.start_training_thread)
        self.train_button.grid(row=3, column=0, sticky="ew", pady=5)
        self.status_var = tk.StringVar(value="Status: Waiting for CSV...")
        ttk.Label(controls, textvariable=self.status_var, foreground="#555555", wraplength=850).grid(
            row=4, column=0, sticky="w", pady=(5, 15))

        # 2. Analyse text
        ttk.Label(controls, text="2. Analyse text:").grid(row=5, column=0, sticky="w")
        self.input_text = tk.Text(controls, height=6, wrap="word", font=("Segoe UI", 10))
        self.input_text.grid(row=6, column=0, sticky="nsew", pady=(3, 5))
        controls.grid_rowconfigure(6, weight=1)
        self.predict_button = ttk.Button(controls, text="Predict Sentiment", command=self.predict_sentiment)
        self.predict_button.grid(row=7, column=0, sticky="ew", pady=5)
        self.prediction_var = tk.StringVar(value="Prediction:")
        ttk.Label(controls, textvariable=self.prediction_var, foreground="#222222").grid(
            row=8, column=0, sticky="w", pady=(5, 10))

        # Probability chart
        ttk.Label(controls, text="Sentiment probabilities:").grid(row=9, column=0, sticky="w", pady=(5, 0))
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=controls)
        self.canvas.get_tk_widget().grid(row=10, column=0, sticky="ew", pady=(5, 0))
        self.draw_empty_chart()

    # -------- GUI callbacks --------

    def browse_csv(self):
        """Component 1: Data Collection - choose a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.csv_path_var.set(file_path)
            self.status_var.set(f"Status: CSV selected ({os.path.basename(file_path)})")

    def set_busy(self, busy):
        """Lock the buttons while training so the model cannot be used half-trained."""
        state = "disabled" if busy else "normal"
        for button in (self.train_button, self.browse_button, self.predict_button):
            button.configure(state=state)

    def set_status(self, text):
        """Safe to call from the training thread: Tkinter must be updated on the main thread."""
        self.after(0, lambda: self.status_var.set(f"Status: {text}"))

    def start_training_thread(self):
        if not self.csv_path_var.get():
            messagebox.showwarning("No CSV", "Please select a CSV file first.")
            return
        self.set_busy(True)
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        """Runs on a background thread so the window stays responsive."""
        try:
            set_seed(SEED)
            self.set_status("Loading and cleaning the data...")
            # Component 1: Data Collection
            text_col = self.text_col_var.get().strip()
            label_col = self.label_col_var.get().strip()
            raw = load_csv_data(self.csv_path_var.get(), text_col, label_col)
            df, report = clean_data(raw, text_col, label_col)
            check_trainable(df)
            train_df, val_df = split_data(df, val_size=VAL_SIZE, seed=SEED)
            label2id, id2label = make_label_maps(df["label"])
            class_weights = compute_class_weights(train_df["label"], label2id)
            print(f"\n{report['rows_out']} distinct sentences kept from {report['rows_in']} rows "
                  f"({report['duplicates_removed']} duplicates removed)")
            print(f"Training on {len(train_df)}, validating on {len(val_df)}")

            # Component 2: Model Training (with Component 3 after every epoch)
            self.set_status("Downloading/loading DistilBERT...")
            model, tokeniser = setup_training(len(label2id), class_weights, self.device)
            train_loader = create_loader(
                TextDataset(train_df["text"], train_df["label"], tokeniser, self.max_len, label2id),
                tokeniser, shuffle=True, seed=SEED)
            val_loader = create_loader(
                TextDataset(val_df["text"], val_df["label"], tokeniser, self.max_len, label2id), tokeniser)

            def on_epoch(epoch, total, train_loss, metrics):
                print_validation_metrics(epoch, total, train_loss, metrics)
                self.set_status(f"Training... epoch {epoch + 1}/{total} done "
                                f"(validation accuracy {metrics['overall_accuracy']:.1%})")

            self.set_status(f"Training on {len(train_df)} sentences (this may take a few minutes)...")
            _, best = fit(model, train_loader, val_loader, self.device, id2label,
                          epochs=EPOCHS, on_epoch=on_epoch)

            self.model, self.tokeniser, self.id2label = model, tokeniser, id2label
            removed = report["duplicates_removed"] + report["conflicting_sentences_removed"]
            summary = (f"Trained. Best epoch {best['epoch']}: validation accuracy "
                       f"{best['overall_accuracy']:.1%}, macro-F1 {best['macro_f1']:.2f} "
                       f"on {len(val_df)} unseen sentences ({removed} duplicate rows removed).")
            self.set_status(summary)
            self.after(0, lambda: messagebox.showinfo("Training complete", summary))
        except Exception as e:
            message = str(e)
            self.set_status("Training failed.")
            self.after(0, lambda: messagebox.showerror("Error during training", message))
        finally:
            self.after(0, lambda: self.set_busy(False))

    def predict_sentiment(self):
        """Component 4: Prediction"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No text", "Please enter some text to analyse.")
            return
        if self.model is None:
            messagebox.showwarning("Model not trained", "Please train the model first.")
            return
        try:
            label, probs = predict_sentiment(self.model, self.tokeniser, text, self.device,
                                             self.max_len, self.id2label)
            self.prediction_var.set(f"Prediction: {label} ({probs.max():.0%})")
            self.update_chart(probs)
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    # -------- Chart --------

    def draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probability")
        self.ax.set_xticks([])
        self.ax.text(0.5, 0.5, "Train a model, then predict to see probabilities",
                     ha="center", va="center", transform=self.ax.transAxes, color="#777777")
        self.fig.tight_layout()
        self.canvas.draw()

    def update_chart(self, probs):
        """Component 5: Dashboard visualisation - one bar per label in the training data."""
        labels = [self.id2label[i] for i in range(len(probs))]
        self.ax.clear()
        self.ax.bar(labels, probs, color="#4c72b0")
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Sentiment")
        self.ax.set_ylabel("Probability")
        for i, v in enumerate(probs):
            self.ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
