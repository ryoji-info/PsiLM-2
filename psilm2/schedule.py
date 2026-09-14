"""The training schedule for the dual stack.

The composition needs a training signal neither single-channel run could have
produced, and that signal is the whole reason this file exists.

Each campaign trained its gate to shut on *off-task* prompts, and for both of
them "off-task" meant GSM8K and MMLU (data/noharm_qwen35_all.json). Neither
channel was ever shown the OTHER channel's on-task prompts as negatives, because
in a single-channel run there was no other channel. So nothing in either trained
gate knows that:

  - on a physics question the constitution gate should be SHUT
  - on a red-team prompt the physics gate should be SHUT
  - on GSM8K or MMLU both should be shut (this part they do know)

Left alone, two gates that were each trained to open on "their" prompts and shut
on arithmetic have no reason to stay out of each other's way, and the guard-rail
evidence says the gate is what does the safety work: at 9B the constitution
channel's KL to the base was 302x larger on red-team prompts than on arithmetic,
and that selectivity, not the write mask, is why full width cost nothing. A
composition that lets both gates drift open on everything would throw exactly
that away.

Hence the cross-gate penalty: on each task's own batches, the other channel's
mean gate is penalised. It is the same mechanism the no-harm arm already uses,
pointed at a negative the single-channel runs did not have.

Phases, in the order they should run:

  0  parity      no training. psilm2.verify_qwen35 confirms the warm-started dual
                 reproduces both single-channel models bit for bit, and each
                 channel's single-channel metrics are recorded as the baseline
                 the composition must not regress.

  1  coexist     gate-only. Both bridges warm-started from their trained
                 checkpoints and frozen except the two gate MLPs. This is the
                 phase that does the real work, and it is cheap for a structural
                 reason: with only the gates trainable, no gradient flows below
                 the injection depths, so the backward pass spans layers 24-32
                 rather than 13-32 and needs no gradient checkpointing.

  2  finetune    optional, and only if phase 1 leaves a regression on either
                 channel's own held-out metric. Unfreezes both bridges at a
                 reduced learning rate, same interleave, early-stopped on the
                 metric that regressed.

Phase 1 before phase 2 is not caution for its own sake. The readouts, the token
heads and the injections already work -- what is new in the composition is each
gate's decision function, because its input stream can now carry the other
channel's write. Training only the gates first says exactly that hypothesis out
loud, and if it succeeds, nothing that took 11,500 physics steps to learn was put
at risk to get there.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from .dual import CONST, PHYS

TASKS = ("physics", "constitution", "noharm")

#: the gate MLP inside either inject module: the only thing phase 1 trains
GATE_KEYS = ("g1", "g2")


@dataclass
class Phase:
    name: str
    steps: int
    lr: float
    gate_only: bool           # freeze everything but inject.g1/g2 in both bridges
    lam_cross: float          # weight on the OTHER channel's gate, on a task batch
    lam_gate: float           # weight on the sum of gates, on a no-harm batch
    cycle: Tuple[str, ...] = TASKS


#: The default plan. Step counts come from what the campaigns measured rather
#: than from habit: the constitution channel converged inside 500 steps at 9B and
#: every width plateaued by chunk 2, so 600 steps of a 3-task cycle gives each arm
#: 200 steps -- comparable to the 500 that trained the channel from scratch, for a
#: phase that only has to adjust two gate MLPs. lam_gate 1.0 is the value both
#: campaigns used; lam_cross is the one genuinely new hyperparameter, started at
#: the same 1.0 on the grounds that "shut on the other task's prompts" is the same
#: demand as "shut on arithmetic", and swept only if phase 1 misses.
PHASES: Tuple[Phase, ...] = (
    Phase("coexist", steps=600, lr=3e-4, gate_only=True, lam_cross=1.0, lam_gate=1.0),
    Phase("finetune", steps=600, lr=1e-4, gate_only=False, lam_cross=1.0, lam_gate=1.0),
)


class DualSchedule:
    """Deterministic task interleave plus the composite loss for each task.

    The cycle is one physics batch, one constitution batch, one no-harm batch,
    repeated. Each channel therefore sees pressure to shut on two of every three
    steps -- once from the no-harm batch and once from the other task's batch --
    against one step of pressure to open. The single-channel runs ran their
    no-harm arm every second step, so this is a comparable duty cycle arrived at
    from the other direction.
    """

    def __init__(self, phase: Phase, sources: Dict[str, Callable[[], dict]]):
        missing = [t for t in phase.cycle if t not in sources]
        if missing:
            raise ValueError(f"no batch source for {missing}; needed {list(phase.cycle)}")
        self.phase = phase
        self.sources = sources
        self.i = 0
        self.counts = {t: 0 for t in phase.cycle}

    def next(self) -> Tuple[str, dict]:
        task = self.phase.cycle[self.i % len(self.phase.cycle)]
        self.i += 1
        self.counts[task] += 1
        batch = self.sources[task]()
        if task == "noharm":
            batch = dict(batch); batch["noharm"] = True
        return task, batch

    # -- the composite loss -------------------------------------------------
    def loss(self, psi, task: str, batch: dict):
        """(loss, stats) for one batch of `task`.

        The task loss comes from the model; the penalties are added here because
        they are a property of the SCHEDULE, not of either channel. psi.last holds
        the gates from the forward pass that the task loss just ran, still inside
        the same graph, so penalising them is differentiable.
        """
        ph = self.phase
        if task == "physics":
            loss, aux = psi.loss_physics(batch)
            other = CONST
        elif task == "constitution":
            loss, aux = psi.loss_constitution(batch)
            other = PHYS
        elif task == "noharm":
            # the model's own no-harm arm already penalises the sum of open gates
            # with psi.lam_gate; nothing to add here beyond what it does
            psi.lam_gate = ph.lam_gate
            loss, aux = psi.loss_noharm(batch)
            other = None
        else:
            raise ValueError(f"unknown task {task!r}")

        stats = {"task": task}
        for ch in (PHYS, CONST):
            g = psi.last.gate(ch)
            stats[f"gate_{ch}"] = None if g is None else g.mean()
        if other is not None and ph.lam_cross > 0:
            g = psi.last.gate(other)
            if g is not None:
                valid = batch["p_attn"].astype(mx.float32)
                cross = (g * valid).sum() / (valid.sum() + 1e-6)
                loss = loss + ph.lam_cross * cross
                stats["cross"] = cross
        return loss, stats


def gate_only_grads(grads: dict) -> dict:
    """Zero every gradient except inject.g1/g2 in both channels.

    Same discipline as the single-channel trainers' --noharm-gate-only, applied to
    a whole phase rather than to the no-harm steps: closing or opening the gate is
    the one route the optimizer is allowed, so nothing that was trained over
    11,500 physics steps or 1,000 constitution steps can move.
    """
    def zero(t):
        if isinstance(t, mx.array):
            return mx.zeros_like(t)
        if isinstance(t, dict):
            return {k: zero(v) for k, v in t.items()}
        if isinstance(t, (list, tuple)):
            return type(t)(zero(v) for v in t)
        return t

    out = zero(grads)
    for ch, g in grads.items():
        inj = g.get("inject") if isinstance(g, dict) else None
        if not isinstance(inj, dict):
            continue
        for k in GATE_KEYS:
            if k in inj:
                out[ch]["inject"][k] = inj[k]
    return out


def trainable_leaf_names(grads: dict) -> List[str]:
    """Flat names of the gradient leaves that are not identically zero."""
    names = []

    def walk(t, pre):
        if isinstance(t, mx.array):
            if float(mx.abs(t).max().item()) > 0:
                names.append(pre)
        elif isinstance(t, dict):
            for k, v in t.items():
                walk(v, f"{pre}.{k}" if pre else str(k))
        elif isinstance(t, (list, tuple)):
            for i, v in enumerate(t):
                walk(v, f"{pre}[{i}]")

    walk(grads, "")
    return sorted(names)


@dataclass
class Baselines:
    """What the composition must not regress, recorded in phase 0.

    Both numbers are held-out and come from the single-channel runs on this
    backbone: the constitution channel's CE to its teacher on the red-team test
    split, and the physics channel's held-out accuracy. A phase is accepted only
    if neither has moved beyond the tolerance, which is set from the campaigns'
    own noise rather than picked: the width campaign's chunk-to-chunk CE movement
    at convergence was 0.001, so 0.003 is three times the drift that convergence
    itself showed.
    """
    const_ce: Optional[float] = None
    phys_acc: Optional[float] = None
    const_tol: float = 0.003
    phys_tol: float = 0.02

    def accept(self, const_ce=None, phys_acc=None) -> Tuple[bool, List[str]]:
        bad = []
        if self.const_ce is not None and const_ce is not None:
            if const_ce > self.const_ce + self.const_tol:
                bad.append(f"constitution CE {const_ce:.4f} > {self.const_ce:.4f}"
                           f" + {self.const_tol}")
        if self.phys_acc is not None and phys_acc is not None:
            if phys_acc < self.phys_acc - self.phys_tol:
                bad.append(f"physics acc {phys_acc:.3f} < {self.phys_acc:.3f}"
                           f" - {self.phys_tol}")
        return (not bad), bad
