"""Two frozen partners on one frozen backbone: the physics bridge and the
constitution bridge, injecting at different depths in the same forward pass.

Both channels already exist in PsiLM and are trained there separately. What is
new here is the composition, and the composition is not neutral: the channels
share a residual stream, so whichever writes lower changes what the other's gate
and attention see. This module makes that interaction explicit rather than
incidental, and its self-test pins down the three properties that let a result be
attributed to the composition rather than to a bug in it:

  1. both channels off      -> bit-identical to the bare staged backbone
  2. physics only           -> bit-identical to psilm.mlx.model.PsiLMMLX
  3. constitution only      -> bit-identical to psilm.mlx.constitution.PsiConstitutionMLX

So any difference in a dual run is the other channel's presence, not a different
code path. PsiDualMLX subclasses PsiLMMLX and overrides only _couple, which means
the physics loss, the no-harm arm and the physics generate path are inherited
verbatim -- there is no second copy of them to drift.

Layer convention, as everywhere in PsiLM: "layer l" is the residual stream
*entering* block l. A channel reads at its l_fwd and writes at its l_rev.

Ordering restriction. Every active channel's read depth must be at or below every
active channel's write depth (max l_fwd <= min l_rev). With that, the reads all
see the unmodified stream and the composition has one unambiguous order; without
it, "read at depth d" and "write at depth d" do not commute and the semantics
would depend on a tie-break nobody should have to remember. The intended Qwen3.5
configuration satisfies it comfortably: both channels read at 13, the
constitution writes at 24 and the physics channel at 26. An interleaved
configuration raises NotImplementedError with this explanation rather than
quietly picking an order.

What this module does NOT settle is how to train the two objectives together.
The physics loss is a five-term supervised objective and the constitution loss is
cross-entropy to a prompted teacher; they have different data, different batch
shapes and different natural learning rates. `loss_physics`, `loss_constitution`
and `loss_noharm` are provided as the three arms a joint trainer would schedule,
and the schedule itself is deliberately left to that trainer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from psilm.mlx.bridges import build_ic_mlx
from psilm.mlx.constitution import (ConstitutionBridgesMLX, ConstitutionModelMLX,
                                    _eos_id_set)
from psilm.mlx.model import PsiLMMLX, cross_entropy_masked
from psilm.mlx.staged import MlxStream

PHYS, CONST = "physics", "constitution"
MODES = ("psilm", "zeroed", "off")


@dataclass
class ChannelState:
    """One channel's trace through a single coupled forward pass."""
    mode: str = "off"
    sigma: Optional[mx.array] = None      # (B, L, 1) gate, always PRE-floor
    ratio: Optional[mx.array] = None      # (B, L) injection RMS / local stream RMS
    tokens: Optional[mx.array] = None     # (B, M, d_model) soft tokens written


@dataclass
class DualState:
    """Everything the last _couple produced, per channel plus the physics readouts.

    The physics slots are None whenever the physics channel was off, which is why
    loss_physics checks the mode before it unpacks them.
    """
    physics: ChannelState = field(default_factory=ChannelState)
    constitution: ChannelState = field(default_factory=ChannelState)
    params_hat: Optional[mx.array] = None
    x0_hat: Optional[mx.array] = None
    x0_logits: Optional[mx.array] = None
    u_hat: Optional[mx.array] = None
    w_x0: Optional[mx.array] = None

    def gate(self, name: str) -> Optional[mx.array]:
        s = getattr(self, name).sigma
        return None if s is None else s[..., 0]


class PsiDualMLX(PsiLMMLX):
    """Frozen backbone + frozen FNO + frozen constitution partner + two bridges.

    Construction takes both channels' pieces and both depth pairs. Either channel
    may be absent (None), in which case this is exactly the single-channel model
    it inherits from or delegates to -- useful for running the three self-test
    baselines through one class.
    """

    def __init__(self, model, tokenizer, *,
                 fno=None, phys_bridges=None, l_fwd_phys=None, l_rev_phys=None,
                 const_model: Optional[ConstitutionModelMLX] = None,
                 const_bridges: Optional[ConstitutionBridgesMLX] = None,
                 l_fwd_const=None, l_rev_const=None,
                 lam_gate: float = 1.0, digit_weight: float = 5.0,
                 lam_x0: float = 0.3, ic_fn=None, param_loss_fn=None):
        n = len(model.model.layers)
        # PsiLMMLX.__init__ wants an fno and bridges; when the physics channel is
        # absent we still want its digit ids, layer defaults and freezing, so pass
        # a stand-in that only has to answer freeze().
        super().__init__(model, tokenizer, fno if fno is not None else _NullFrozen(),
                         phys_bridges, l_fwd=l_fwd_phys, l_rev=l_rev_phys,
                         digit_weight=digit_weight, lam_x0=lam_x0,
                         ic_fn=ic_fn, param_loss_fn=param_loss_fn)
        self.has_phys = phys_bridges is not None and fno is not None
        self.has_const = const_bridges is not None and const_model is not None
        self.const = const_model
        self.cphi = const_bridges
        self.l_fwd_const = l_fwd_const if l_fwd_const is not None else round(n * 10 / 24)
        self.l_rev_const = l_rev_const if l_rev_const is not None else round(n * 15 / 24)
        if self.has_const:
            assert 0 < self.l_fwd_const <= self.l_rev_const <= n, \
                (self.l_fwd_const, self.l_rev_const, n)
            assert const_bridges.m_tokens == const_model.n_readout, \
                (f"bridges expect {const_bridges.m_tokens} readout tokens, "
                 f"model gives {const_model.n_readout}")
            assert const_bridges.d_const == const_model.d_const, \
                (const_bridges.d_const, const_model.d_const)
            const_model.model.freeze()
        self.lam_gate = lam_gate
        self.eos_ids = _eos_id_set(tokenizer)
        # per-channel arm: "psilm" writes, "zeroed" builds the tokens and measures
        # the gate without writing, "off" does not touch the bridge at all. The
        # zeroed arm is what makes a gate observable on a stream it never altered.
        self.modes: Dict[str, str] = {PHYS: "psilm" if self.has_phys else "off",
                                      CONST: "psilm" if self.has_const else "off"}
        self.floors: Dict[str, Optional[float]] = {PHYS: None, CONST: None}
        self.last = DualState()

    # -- arms ---------------------------------------------------------------
    def set_modes(self, physics: Optional[str] = None, constitution: Optional[str] = None):
        """Set either channel's arm, refusing a mode a channel cannot serve."""
        for name, m, present in ((PHYS, physics, self.has_phys),
                                 (CONST, constitution, self.has_const)):
            if m is None:
                continue
            if m not in MODES:
                raise ValueError(f"{name}: mode {m!r} not in {MODES}")
            if m != "off" and not present:
                raise ValueError(f"{name}: no bridge was constructed, only 'off' is available")
            self.modes[name] = m
        return self

    def active(self) -> List[str]:
        return [c for c in (PHYS, CONST) if self.modes[c] != "off"]

    def _plan(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """(reads, writes) as ascending (depth, channel), with the order check."""
        depths = {PHYS: (self.l_fwd, self.l_rev), CONST: (self.l_fwd_const, self.l_rev_const)}
        act = self.active()
        reads = sorted((depths[c][0], c) for c in act)
        writes = sorted((depths[c][1], c) for c in act)
        if reads and writes and reads[-1][0] > writes[0][0]:
            raise NotImplementedError(
                "a channel reads deeper than another writes "
                f"(reads {reads}, writes {writes}). Every active channel's read "
                "depth must be at or below every write depth, so that the reads "
                "see the unmodified stream and the composition has one order. "
                "See this module's docstring.")
        return reads, writes

    # -- the two channels' halves ------------------------------------------
    def _read(self, ch: str, hidden, prompt_mask, x0_span):
        if ch is PHYS or ch == PHYS:
            params_hat, x0_hat, x0_logits, w_x0 = self.phi.fwd(hidden, prompt_mask, x0_span)
            ic = self.ic_fn(params_hat)
            feats = self.fno.features(ic)
            u_field = self.fno.proj(feats).squeeze(-1)
            x0_in = mx.stop_gradient(x0_hat) if self.detach_x0 else x0_hat
            tokens, u_hat = self.phi.rev(feats, u_field, x0_in)
            if getattr(self.phi, "channel", "field") == "value":
                # as in PsiLMMLX: the language side learns to read the physics
                # answer; the lookup stays deep-supervised through loss_u
                tokens = self.phi.val(mx.stop_gradient(u_hat))
            self.last.params_hat, self.last.x0_hat = params_hat, x0_hat
            self.last.x0_logits, self.last.u_hat, self.last.w_x0 = x0_logits, u_hat, w_x0
            return tokens
        soft, _ = self.cphi.fwd(hidden, prompt_mask)
        return self.cphi.rev(self.const.features(soft))

    def _write(self, ch: str, hidden, tokens):
        """Inject, or measure the gate without injecting when the arm is zeroed.

        The floor is set for the call and always restored, so an inference arm
        cannot leak a leaky gate into the next call -- the same discipline as
        PsiConstitutionMLX._apply_inject and bench_common.StagedDecoder._inject.
        """
        inj = (self.phi if ch == PHYS else self.cphi).inject
        inj.gate_floor = self.floors[ch]
        try:
            h_inj, sigma, ratio = inj(hidden, tokens, return_ratio=True)
        finally:
            inj.gate_floor = None
        return (h_inj if self.modes[ch] == "psilm" else hidden), sigma, ratio

    # -- coupling -----------------------------------------------------------
    def _couple(self, stream: MlxStream, prompt_mask, x0_span=None):
        """Override of PsiLMMLX._couple: same signature and same 6-tuple, so every
        inherited loss and generate path works unchanged, with the constitution
        channel riding along in the same pass.

        The returned sigma is the PHYSICS gate, keeping the inherited diagnostics
        about the channel they were written for; the constitution's gate, ratio
        and tokens are on self.last.
        """
        self.last = DualState()
        self.last.physics.mode = self.modes[PHYS]
        self.last.constitution.mode = self.modes[CONST]
        reads, writes = self._plan()
        if not reads:                                  # base arm: backbone alone
            stream.run(0, self.n_layers)
            self._last_ratio = None
            return (None, None, None, None, None, None)

        depth = 0
        for d, ch in reads:
            stream.run(depth, d); depth = d
            getattr(self.last, ch).tokens = self._read(ch, stream.hidden, prompt_mask, x0_span)
        for d, ch in writes:
            stream.run(depth, d); depth = d
            stream.hidden, sigma, ratio = self._write(ch, stream.hidden,
                                                      getattr(self.last, ch).tokens)
            st = getattr(self.last, ch)
            st.sigma, st.ratio = sigma, ratio
        stream.run(depth, self.n_layers)

        # what the inherited physics code expects to find
        self._last_ratio = self.last.physics.ratio
        L = self.last
        return (L.params_hat, L.x0_hat, L.x0_logits, L.u_hat, L.physics.sigma, L.w_x0)

    def logits(self, ids, attn=None, prompt_mask=None, x0_span=None):
        """Full-sequence logits plus the DualState for this pass."""
        s = MlxStream(self.model, ids, attn)
        if prompt_mask is None:
            prompt_mask = mx.ones(ids.shape, dtype=mx.bool_)
        self._couple(s, prompt_mask, x0_span)
        return s.finish(), self.last

    # -- training arms ------------------------------------------------------
    def loss_physics(self, batch, **kw):
        """The inherited physics objective, with whatever the constitution arm is
        set to. Refuses to run with the physics channel off rather than silently
        returning a loss with no physics in it."""
        if self.modes[PHYS] == "off":
            raise ValueError("loss_physics with the physics channel off")
        return self.loss_fn(batch, **kw)

    def loss_constitution(self, batch):
        """CE to the teacher's continuation, plus the gate penalty on no-harm
        batches. Mirrors PsiConstitutionMLX.loss_fn term for term; the difference
        is that the gate penalty covers EVERY open channel, because on an off-task
        prompt neither partner is relevant and either one left open is a leak."""
        if self.modes[CONST] == "off":
            raise ValueError("loss_constitution with the constitution channel off")
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        self._couple(s, batch["prompt_mask"])
        logits = s.finish()
        ce = cross_entropy_masked(logits, batch["p_labels"], None, 1.0)
        valid = batch["p_attn"].astype(mx.float32)
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        pm = batch["prompt_mask"].astype(mx.float32)
        n_valid, n_resp, n_pm = valid.sum() + 1e-6, resp.sum() + 1e-6, pm.sum() + 1e-6
        stats, gate_sum = {}, mx.array(0.0)
        for ch in self.active():
            g = self.last.gate(ch)
            if g is None:
                continue
            gate_sum = gate_sum + (g * valid).sum() / n_valid
            r = getattr(self.last, ch).ratio
            stats[ch] = ((g * valid).sum() / n_valid, (g * resp).sum() / n_resp,
                         (r * resp).sum() / n_resp, (pm * g).sum() / n_pm)
        loss = ce + (self.lam_gate * gate_sum if batch.get("noharm") else 0.0)
        c = stats.get(CONST, (mx.array(0.0),) * 4)
        return loss, (ce, c[0], c[1], c[2], c[3], gate_sum)

    def loss_noharm(self, batch):
        """Off-task prompt through every open channel, trained to reproduce the
        backbone's own continuation, plus lam_gate on the sum of the gates."""
        b = dict(batch); b["noharm"] = True
        return self.loss_constitution(b)

    def trainable_parameters(self) -> Dict[str, Any]:
        """Both bridge modules' parameters under stable names, for one optimizer."""
        out = {}
        if self.has_phys:
            out[PHYS] = self.phi.trainable_parameters()
        if self.has_const:
            out[CONST] = self.cphi.trainable_parameters()
        return out

    def n_trainable(self) -> int:
        def count(t):
            if isinstance(t, mx.array):
                return t.size
            if isinstance(t, dict):
                return sum(count(v) for v in t.values())
            if isinstance(t, (list, tuple)):
                return sum(count(v) for v in t)
            return 0
        return count(self.trainable_parameters())

    def describe(self) -> str:
        parts = []
        if self.has_phys:
            parts.append(f"physics {self.l_fwd}/{self.l_rev} "
                         f"channel={getattr(self.phi, 'channel', 'field')}")
        if self.has_const:
            w = len(self.cphi.write_dims)
            parts.append(f"constitution {self.l_fwd_const}/{self.l_rev_const} "
                         f"write {w} of {self.cphi.d_model} "
                         f"({100.0 * w / self.cphi.d_model:.2f}%)")
        return (f"PsiDualMLX over {self.n_layers} layers | " + " | ".join(parts)
                + f" | {self.n_trainable() / 1e6:.2f}M trainable")


class _NullFrozen:
    """Stand-in for an absent FNO: PsiLMMLX.__init__ only calls freeze() on it."""

    def freeze(self):
        return self


def dual_meta(psi: PsiDualMLX, step: int, extra: Optional[Dict[str, Any]] = None):
    """Checkpoint sidecar: enough to rebuild both channels' shapes exactly.

    The resolved write/read dimension lists are stored, not the --write-dims
    spelling that produced them, for the reason the single-channel loader gives:
    a `random:<seed>:<n>` control would come back different if the seed were lost.
    """
    meta: Dict[str, Any] = {"n_layers": psi.n_layers,
                            "has_physics": psi.has_phys,
                            "has_constitution": psi.has_const,
                            "step": int(step)}
    if psi.has_phys:
        meta["physics"] = {"l_fwd": psi.l_fwd, "l_rev": psi.l_rev,
                           "channel": getattr(psi.phi, "channel", "field"),
                           "gate_bias": float(psi.phi.inject.g2.bias.mean().item()),
                           "inj_cap": psi.phi.inject.inj_cap}
    if psi.has_const:
        c = psi.cphi
        meta["constitution"] = {"l_fwd": psi.l_fwd_const, "l_rev": psi.l_rev_const,
                                "d_model": c.d_model, "d_const": c.d_const,
                                "k_fwd": c.k_fwd, "m_tokens": c.m_tokens,
                                "write_dims": c.write_dims, "read_dims": c.read_dims,
                                "gate_bias": float(c.inject.gate_bias),
                                "inj_cap": c.inject.inj_cap,
                                "readout_norm": c.fwd.readout_norm,
                                "emb_rms": c.fwd.emb_rms,
                                "d_hidden": c.fwd.mlp1.weight.shape[0],
                                "const_model": psi.const.path}
    if extra:
        meta.update(extra)
    return meta


def load_dual_stack(model, tokenizer, *,
                    phys_ckpt=None, fno_path="results/stage2/fno.pt",
                    const_ckpt=None, const_model_path=None,
                    l_rev_phys=None, l_rev_const=None, lam_gate: float = 1.0):
    """Assemble a PsiDualMLX from the two channels' trained checkpoints.

    Both halves are read exactly the way PsiLM's own evaluators read them -- the
    physics side reproduces eval/mlx_stage2_eval.py's assembly (including taking
    the coupling depth from the checkpoint meta rather than the Stage-2 fraction,
    which is wrong for any backbone that injected elsewhere: Qwen3.5 trained at
    26 of 32 where the rule says 20), and the constitution side goes through
    psilm.mlx.constitution.load_constitution_stack, which resolves the write and
    read dimension lists from the meta so a random control cannot come back
    different. Either checkpoint may be omitted to build a single-channel model.

    NOT YET EXERCISED against real checkpoints: at the time of writing the GPU is
    committed and the 9B stack could not be loaded, so this composes two tested
    loaders but the composition itself has only been read, not run. Check the
    printed describe() line against the two source metas the first time it is used.
    """
    from pathlib import Path
    import json

    fno = phys = None
    l_fwd_phys = None
    if phys_ckpt is not None:
        from psilm.mlx.bridges import PsiBridgesMLX, load_bridge_weights
        from psilm.mlx.fno import convert_from_torch, load_fno_safetensors
        ck = Path(phys_ckpt)
        pmeta = json.loads(Path(str(ck) + ".meta").read_text())
        margs = pmeta.get("args", {})
        fno = (load_fno_safetensors(fno_path) if str(fno_path).endswith(".safetensors")
               else convert_from_torch(fno_path))
        phys = PsiBridgesMLX(d_model=model.args.hidden_size,
                             gate_bias=margs.get("gate_bias", -2.0),
                             inj_cap=margs.get("inj_cap"),
                             channel=margs.get("channel", "field"),
                             readout_norm=margs.get("readout_norm", "rms"))
        load_bridge_weights(phys, ck)
        fno.freeze()
        l_fwd_phys = pmeta.get("l_fwd")
        if l_rev_phys is None:
            l_rev_phys = pmeta.get("l_rev")

    const = cphi = None
    l_fwd_const = None
    if const_ckpt is not None:
        from psilm.mlx.constitution import load_constitution_stack
        cphi, const, cmeta = load_constitution_stack(const_ckpt, const_model_path)
        l_fwd_const = cmeta.get("l_fwd")
        if l_rev_const is None:
            l_rev_const = cmeta.get("l_rev")

    psi = PsiDualMLX(model, tokenizer, fno=fno, phys_bridges=phys,
                     l_fwd_phys=l_fwd_phys, l_rev_phys=l_rev_phys,
                     const_model=const, const_bridges=cphi,
                     l_fwd_const=l_fwd_const, l_rev_const=l_rev_const,
                     lam_gate=lam_gate)
    # Fail here rather than mid-training: on Qwen3.5 the constitution channel
    # trained at 13/24 and the physics channel at 13/26, which satisfies the
    # ordering rule; a pair that does not is a configuration error, and _plan()
    # would otherwise only discover it on the first forward pass.
    psi._plan()
    return psi
