"""Compatibility entry point for the audited reviewer-revision DP trainer.

Historical setup with silent optimizer/accountant fallbacks was retired.
"""
from revision_training import PrivateTrainResult, train_private_model

__all__ = ["PrivateTrainResult", "train_private_model"]
