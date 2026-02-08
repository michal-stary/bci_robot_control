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
NIRS-based Motor Intent Classifier (3-Class Version).

Classes:
  - Fist (any fist) -> FORWARD
  - Tongue -> BACKWARD  
  - Relax -> STOP

Cross-subject LOSO accuracy: ~56-58%
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


# 3-Class to action mapping
CLASS_TO_ACTION = {
    "Fist": RobotAction.FORWARD,
    "Tongue": RobotAction.BACKWARD,
    "Relax": RobotAction.STOP,
}


class NIRSNet3Class(nn.Module):
    """MLP for 3-class NIRS classification."""

    def __init__(self, n_features: int = 27840, n_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NIRSClassifier:
    """
    NIRS-based motor intent classifier (3-class).
    
    Example usage:
        classifier = NIRSClassifier.load()
        result = classifier.predict(nirs_window)
        print(f"Action: {result.action}, Confidence: {result.confidence:.1%}")
    """

    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        device: torch.device,
        confidence_threshold: float = 0.4,
    ) -> None:
        self.model = model
        self.classes = classes
        self.device = device
        self.confidence_threshold = confidence_threshold

    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.4,
        device: str = "auto",
    ) -> "NIRSClassifier":
        """
        Load a trained classifier.
        
        Args:
            model_path: Path to model checkpoint. Defaults to bundled 3-class model.
            confidence_threshold: Minimum confidence for action (else STOP).
            device: Device to use ("auto", "cuda", "mps", "cpu").
        
        Returns:
            Loaded NIRSClassifier instance.
        """
        # Default to 3-class model
        if model_path is None:
            model_path = Path(__file__).parent / "nirs_3class.pt"
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

        classes = checkpoint["classes"]
        n_features = checkpoint.get("n_features", 27840)
        n_classes = checkpoint.get("n_classes", 3)

        model = NIRSNet3Class(n_features=n_features, n_classes=n_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)
        model.eval()

        return cls(model, classes, dev, confidence_threshold)

    def preprocess(self, nirs_window: np.ndarray) -> torch.Tensor:
        """
        Preprocess NIRS window for inference.

        Args:
            nirs_window: Shape (time, modules=40, sds_buckets=3, wavelengths=2, moments=3)
                         OR shape (time, features) if already flattened.

        Returns:
            Preprocessed tensor ready for model input.
        """
        # If already flattened
        if nirs_window.ndim == 2:
            nirs_flat = nirs_window.flatten().astype(np.float32)
        else:
            # Full preprocessing from raw NIRS window
            # Use short + medium range (indices 0 and 1), skip long range (index 2) which is NaN
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
            elif nirs_valid.shape[0] > 14:
                nirs_stim = nirs_valid[14:]
                baseline = nirs_valid[:14].mean(axis=0, keepdims=True)
            else:
                nirs_stim = nirs_valid
                baseline = 0

            # Baseline correction
            nirs_corrected = nirs_stim - baseline

            # Flatten
            nirs_flat = nirs_corrected.flatten().astype(np.float32)

        # Z-score normalization
        mean = nirs_flat.mean()
        std = nirs_flat.std()
        if std > 0:
            nirs_flat = (nirs_flat - mean) / std

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(nirs_flat).unsqueeze(0)  # (1, features)
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
