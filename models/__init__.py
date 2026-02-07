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
NIRS Motor Intent Classification Models.

Example usage:
    from models import NIRSClassifier, RobotAction
    
    classifier = NIRSClassifier.load()
    result = classifier.predict(nirs_window)
    
    if result.action == RobotAction.FORWARD:
        robot.move_forward()
    elif result.action == RobotAction.BACKWARD:
        robot.move_backward()
    else:
        robot.stop()
"""

from .nirs_classifier import (
    NIRSClassifier,
    NIRSNet,
    RobotAction,
    ClassificationResult,
    CLASS_TO_ACTION,
)

__all__ = [
    "NIRSClassifier",
    "NIRSNet",
    "RobotAction",
    "ClassificationResult",
    "CLASS_TO_ACTION",
]
