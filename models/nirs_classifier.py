# Copyright 2026 Kernel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NIRS-based Motor Intent Classifier.

Standalone module for NIRS classification that can be used without Holoscan.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


class RobotAction(str, Enum):
    """Robot control actions."""
    FORWARD = "forward"
    BACKWARD = "backward"
    STOP = "stop"


class ClassificationResult(NamedTuple):
    """Result from classification."""
    action: RobotAction
    class_name: str
    confidence: float
    all_probs: dict[str, float]


# Class to action mapping
CLASS_TO_ACTION = {
    "Right Fist": RobotAction.FORWARD,
    "Tongue Tapping": RobotAction.BACKWARD,
    "Left Fist": RobotAction.STOP,
    "Both Fists": RobotAction.STOP,
    "Relax": RobotAction.STOP,
}


class NIRSNet(nn.Module):
    """CNN for NIRS time series classification."""

    def __init__(self, n_features: int = 480, n_classes: int = 5, dropout: float = 0.4) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class NIRSClassifier:
    """
    NIRS-based motor intent classifier.
    
    Example usage:
        classifier = NIRSClassifier.load()
        result = classifier.predict(nirs_window)
        print(f"Action: {result.action}, Confidence: {result.confidence:.1%}")
    """

    def __init__(
        self,
        model: NIRSNet,
        classes: list[str],
        device: torch.device,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.model = model
        self.classes = classes
        self.device = device
        self.confidence_threshold = confidence_threshold

    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.3,
        device: str = "auto",
    ) -> "NIRSClassifier":
        """
        Load a trained classifier.
        
        Args:
            model_path: Path to model checkpoint. Defaults to bundled model.
            confidence_threshold: Minimum confidence for action (else STOP).
            device: Device to use ("auto", "cuda", "mps", "cpu").
        
        Returns:
            Loaded NIRSClassifier instance.
        """
        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent / "nirs_net.pt"
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)

        classes = checkpoint["label_encoder_classes"]
        n_features = checkpoint["n_features"]
        n_classes = checkpoint["n_classes"]

        model = NIRSNet(n_features=n_features, n_classes=n_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)
        model.eval()

        return cls(model, classes, dev, confidence_threshold)

    def preprocess(self, nirs_window: np.ndarray) -> torch.Tensor:
        """
        Preprocess NIRS window for inference.

        Args:
            nirs_window: Shape (time, modules=40, sds_buckets=3, wavelengths=2, moments=3)
                         OR shape (time, features=480) if already flattened.

        Returns:
            Preprocessed tensor ready for model input.
        """
        # If already flattened, just normalize
        if nirs_window.ndim == 2 and nirs_window.shape[1] == 480:
            nirs_flat = nirs_window.astype(np.float32)
        else:
            # Full preprocessing from raw NIRS window
            # Use short + medium range (indices 0 and 1), skip long range (index 2) which may be NaN
            short_range = nirs_window[:, :, 0, :, :]  # (time, 40, 2, 3)
            medium_range = nirs_window[:, :, 1, :, :]  # (time, 40, 2, 3)

            # Stack short and medium range
            nirs_valid = np.stack([short_range, medium_range], axis=2)  # (time, 40, 2, 2, 3)

            # Handle NaN values
            nirs_valid = np.nan_to_num(nirs_valid, nan=0.0)

            # Use stimulus period (skip first ~3 seconds = 14 samples at 4.76Hz)
            if nirs_valid.shape[0] > 58:
                nirs_stim = nirs_valid[-58:, :, :, :, :]
                baseline = nirs_valid[:-58, :, :, :, :].mean(axis=0, keepdims=True)
            else:
                nirs_stim = nirs_valid
                baseline = nirs_valid[:14, :, :, :, :].mean(axis=0, keepdims=True) if nirs_valid.shape[0] > 14 else 0

            # Baseline correction
            nirs_corrected = nirs_stim - baseline

            # Flatten: 40 modules * 2 SDS * 2 wavelengths * 3 moments = 480 features
            nirs_flat = nirs_corrected.reshape(nirs_corrected.shape[0], -1).astype(np.float32)

        # Z-score normalization
        mean = nirs_flat.mean()
        std = nirs_flat.std()
        if std > 0:
            nirs_flat = (nirs_flat - mean) / std

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(nirs_flat).unsqueeze(0)  # (1, time, features)
        return tensor.to(self.device)

    def predict(self, nirs_window: np.ndarray) -> ClassificationResult:
        """
        Predict motor intent from NIRS window.

        Args:
            nirs_window: NIRS data array.

        Returns:
            ClassificationResult with action, class name, confidence, and probabilities.
        """
        # Preprocess
        tensor = self.preprocess(nirs_window)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        # Get class name and action
        class_name = self.classes[pred_idx]

        # Apply confidence threshold
        if confidence >= self.confidence_threshold:
            action = CLASS_TO_ACTION.get(class_name, RobotAction.STOP)
        else:
            action = RobotAction.STOP

        # Build probability dict
        all_probs = {cls: probs[0, i].item() for i, cls in enumerate(self.classes)}

        return ClassificationResult(
            action=action,
            class_name=class_name,
            confidence=confidence,
            all_probs=all_probs,
        )
