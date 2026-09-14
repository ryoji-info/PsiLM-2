"""ΨLM-2: two frozen partners on one frozen backbone.

Depends on the ΨLM package for the bridges, the staged forward and the two
single-channel models it composes:  pip install -e ../PsiLM
"""

from .dual import CONST, PHYS, ChannelState, DualState, PsiDualMLX, dual_meta

__all__ = ["PsiDualMLX", "DualState", "ChannelState", "dual_meta", "PHYS", "CONST"]
