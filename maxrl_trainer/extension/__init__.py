"""MaxRL extension: imperfect reward signals on Countdown.

This package implements the *extension* portion of our CS224R default project
(the only component for which AI-tool assistance is permitted under the course
honor-code policy). The core SFT / IPO / RLOO implementations live elsewhere
and are untouched by this package.

Modules
-------
reward_noise
    Symmetric reward-flip noise model used to simulate an imperfect verifier.
curriculum
    Dynamic prompt re-weighting that down-weights prompts whose observed
    success rate is indistinguishable from the noise floor.
rlaif
    A second (LLM-judge) reward channel and combination rules that reduce the
    effective flip rate of the supervision signal (RLAIF / d-RLAIF).

All modules depend only on ``numpy`` (plus the standard library); heavyweight
dependencies such as ``torch`` and ``vllm`` are imported lazily so the building
blocks remain unit-testable on CPU without a GPU.
"""

from .reward_noise import SymmetricRewardFlipNoise, binarize_rewards
from .curriculum import PromptReweightingCurriculum
from .rlaif import (
    SimulatedJudge,
    VLLMJudge,
    combine_rewards,
    RLAIFConfig,
)

__all__ = [
    "SymmetricRewardFlipNoise",
    "binarize_rewards",
    "PromptReweightingCurriculum",
    "SimulatedJudge",
    "VLLMJudge",
    "combine_rewards",
    "RLAIFConfig",
]
