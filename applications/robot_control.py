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
Brain-Computer Interface Robot Control Application.

Uses NIRS brain signals to classify motor intent and control robot movement:
- Right Fist → Move FORWARD
- Tongue Tapping → Move BACKWARD
- Other intents → STOP
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from holoscan.core import Application, ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext

from operators.inference import InferenceOperator, InferenceOutput, RobotAction
from operators.stream import EegStreamOperator, NirsStreamOperator
from operators.window import WindowOperator
from streams.kernel_sdk import KernelSDKNirsStream, KernelSdkEegStream


class RobotControlOperator(Operator):
    """
    Operator that receives inference results and controls the robot.

    This operator receives motor intent classifications and translates them
    to robot control commands.
    """

    def __init__(self, *, fragment: Any | None = None) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._last_action: RobotAction = RobotAction.STOP

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("action")

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del op_output, context

        inference: InferenceOutput = op_input.receive("action")

        if inference.action != self._last_action:
            self._last_action = inference.action
            print(f"[RobotControl] ACTION CHANGED: {inference.action.value.upper()}")

            # Here you would send the actual robot control command
            # Example integration with unitree_sdk2py:
            #
            # if inference.action == RobotAction.FORWARD:
            #     controller.set_action(Action.FORWARD)
            # elif inference.action == RobotAction.BACKWARD:
            #     controller.set_action(Action.BACKWARD)
            # else:
            #     controller.set_action(Action.STOP)


class RobotControlApplication(Application):
    """
    Holoscan application for BCI-based robot control.

    Streams NIRS and EEG data from Kernel Flow headset, windows the data,
    runs inference to classify motor intent, and controls the robot.
    """

    def __init__(
        self,
        *,
        nirs_window_size: int = 72,
        eeg_window_size: int = 7499,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self._nirs_window_size = nirs_window_size
        self._eeg_window_size = eeg_window_size
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold

    def compose(self) -> Sequence[Operator]:
        fragment = self

        # Data streaming operators
        nirs_operator = NirsStreamOperator(stream=KernelSDKNirsStream(), fragment=fragment)
        eeg_operator = EegStreamOperator(stream=KernelSdkEegStream(), fragment=fragment)

        # Windowing operator
        window_operator = WindowOperator(
            nirs_window_size=self._nirs_window_size,
            eeg_window_size=self._eeg_window_size,
            fragment=fragment,
        )

        # Inference operator
        inference_operator = InferenceOperator(
            model_path=self._model_path,
            confidence_threshold=self._confidence_threshold,
            fragment=fragment,
        )

        # Robot control operator
        robot_operator = RobotControlOperator(fragment=fragment)

        # Connect the pipeline
        self.add_flow(nirs_operator, window_operator, {("samples", "nirs_samples")})
        self.add_flow(eeg_operator, window_operator, {("eeg_data", "eeg_samples")})
        self.add_flow(window_operator, inference_operator, {("window", "window")})
        self.add_flow(inference_operator, robot_operator, {("action", "action")})


def main() -> None:
    """Entry point for the BCI robot control application."""
    parser = argparse.ArgumentParser(
        description="BCI Robot Control - Control robots with brain signals"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model (default: models/nirs_net.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for action (default: 0.3)",
    )
    parser.add_argument(
        "--nirs-window",
        type=int,
        default=72,
        help="NIRS window size in samples (default: 72 = ~15s @ 4.76Hz)",
    )
    parser.add_argument(
        "--eeg-window",
        type=int,
        default=7499,
        help="EEG window size in samples (default: 7499 = ~15s @ 500Hz)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BCI Robot Control")
    print("=" * 60)
    print(f"  Model: {args.model or 'models/nirs_net.pt'}")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  NIRS window: {args.nirs_window} samples")
    print(f"  EEG window: {args.eeg_window} samples")
    print()
    print("Action mapping:")
    print("  Right Fist     -> FORWARD")
    print("  Tongue Tapping -> BACKWARD")
    print("  Other          -> STOP")
    print("=" * 60)

    app = RobotControlApplication(
        nirs_window_size=args.nirs_window,
        eeg_window_size=args.eeg_window,
        model_path=args.model,
        confidence_threshold=args.confidence,
    )
    app.run()


if __name__ == "__main__":
    main()
