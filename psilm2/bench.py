"""A coupler that drives BOTH channels, for the guard-rail bench.

eval/bench_common.py's StagedDecoder is the only decoder here with a KV cache, and
the guard-rail needs it: its GSM8K arm generates 384 tokens per item, which the
recompute-the-whole-sequence decode of psilm2.dual cannot afford. But the bench has
one write point and the dual stack has two --- the constitution channel at layer 24
and the physics channel at 26 --- so the decoder gained one opt-in hook: a coupler
that exposes `l_rev2` and `inject2` gets a second injection on the way up, and any
coupler that does not (every physics run, every single-channel constitution run)
takes the path it always took. That equivalence was checked by running a guard-rail
with the old and new decoder and diffing: identical on every field but wall-clock.

`DualCoupler` is that coupler. It owns both channels' reads and both writes, so
nothing about the composition lives in the shared bench. It mirrors
ConstitutionCoupler.tokens/inject and StagedDecoder._physics_tokens rather than
re-deriving them, because a second implementation of the physics read is a silent
way to feed the trained readout something it was not trained on.

The arm semantics follow the bench's: "psilm" writes, anything else measures the
gate without writing. Which CHANNELS are open is fixed when the coupler is built,
so the bench's three arms become base / this-configuration / zeroed, and a run per
configuration gives the comparison.
"""

from typing import Optional

import mlx.core as mx

MODES = ("both", "physics", "constitution")


class DualCoupler:
    """Both channels behind the two calls StagedDecoder makes of one.

    channels: "both", "physics" (constitution measured but not written) or
    "constitution" (physics measured but not written). Measuring rather than
    omitting the idle channel keeps its gate observable in every configuration,
    which is the whole point of the zeroed arm applied per channel.
    """

    def __init__(self, phys_bridges, fno, const_bridges, const_model,
                 l_rev_phys: int, l_rev_const: int, channels: str = "both"):
        if channels not in MODES:
            raise ValueError(f"channels={channels!r} not in {MODES}")
        if l_rev_const > l_rev_phys:
            raise NotImplementedError(
                f"this coupler writes the constitution channel first (at {l_rev_const}) "
                f"and the physics channel second (at {l_rev_phys}); a configuration with "
                "the physics write shallower needs the two swapped, which the bench's "
                "single l_rev cannot express without another hook.")
        self.phi, self.fno = phys_bridges, fno
        self.cphi, self.const = const_bridges, const_model
        self.channels = channels
        # the bench injects the SHALLOWER write at its own l_rev and this one at l_rev2
        self.l_rev2 = int(l_rev_phys)
        self.l_rev_const = int(l_rev_const)
        self.last = {}

    # -- read: both channels, from the same prompt hidden state ---------------
    def tokens(self, h_prompt, x0_span=None, sub_value=None):
        if sub_value is not None:
            raise ValueError("the dual coupler has no scalar to substitute; run the "
                             "physics guard-rail for the shuffled-value control")
        from psilm.mlx.bridges import build_ic_mlx
        L = h_prompt.shape[1]
        pmask = mx.ones((1, L), dtype=mx.bool_)

        # constitution: the whole prompt is the situation, so no span
        soft, weights = self.cphi.fwd(h_prompt, pmask)
        c_tokens = self.cphi.rev(self.const.features(soft))

        # physics: exactly StagedDecoder._physics_tokens, span included when given
        span = None if x0_span is None else mx.array([list(x0_span)], dtype=mx.int32)
        params_hat, x0_hat, _x0_logits, _w = self.phi.fwd(h_prompt, pmask, span)
        feats = self.fno.features(build_ic_mlx(params_hat))
        u_field = self.fno.proj(feats).squeeze(-1)
        p_tokens, u_hat = self.phi.rev(feats, u_field, x0_hat)
        if getattr(self.phi, "channel", "field") == "value":
            p_tokens = self.phi.val(u_hat)
        mx.eval(c_tokens, p_tokens, u_hat, x0_hat)
        diag = {"channels": self.channels,
                "u_hat": round(float(u_hat[0].item()), 4),
                "x0_hat": round(float(x0_hat[0].item()), 4),
                "const_token_rms": round(float(mx.sqrt((c_tokens * c_tokens).mean()).item()), 4),
                "phys_token_rms": round(float(mx.sqrt((p_tokens * p_tokens).mean()).item()), 4)}
        return (c_tokens, p_tokens), diag

    # -- write: the shallower channel at the bench's l_rev --------------------
    def _apply(self, inject, h, tokens, write: bool, floor):
        inject.gate_floor = floor
        try:
            out = inject(h, tokens)
        finally:
            inject.gate_floor = None
        h_inj, sigma = out[0], out[1]
        return (h_inj if write else h), sigma

    def inject(self, h, tokens, mode, floor=None):
        """The constitution write, at the bench's own l_rev."""
        write = mode == "psilm" and self.channels in ("both", "constitution")
        h, sigma = self._apply(self.cphi.inject, h, tokens[0], write, floor)
        self.last["gate_constitution"] = sigma
        return h, sigma

    def inject2(self, h, tokens, mode, floor=None):
        """The physics write, at l_rev2. The sigma the bench records is the
        constitution channel's, from inject(); this one is kept on `last` so a
        caller can read it without the bench needing a second gate column."""
        write = mode == "psilm" and self.channels in ("both", "physics")
        h, sigma = self._apply(self.phi.inject, h, tokens[1], write, floor)
        self.last["gate_physics"] = sigma
        return h, sigma


def build_dual_coupler(phys_ckpt, const_ckpt, d_model, fno_path, const_model_path=None,
                       gate_bias=None, channels="both"):
    """Load both channels' trained checkpoints and return (coupler, l_fwd, l_rev, meta).

    l_rev is the SHALLOWER of the two write depths, because that is the one the
    bench injects at its own pause point; the deeper one rides on the coupler as
    l_rev2. Both reads happen at l_fwd, which the two checkpoints must agree on --
    the bench has one read point, and on Qwen3.5 both channels trained at 13.
    """
    import json
    from pathlib import Path
    from psilm.mlx.constitution import load_constitution_stack

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "PsiLM" / "eval"))
    from bench_common import load_physics_stack                    # noqa: E402

    fno, phys, pmeta = load_physics_stack(phys_ckpt, d_model, gate_bias, fno_path)
    cphi, const, cmeta = load_constitution_stack(const_ckpt, const_model_path)
    lf_p, lr_p = int(pmeta["l_fwd"]), int(pmeta["l_rev"])
    lf_c, lr_c = int(cmeta["l_fwd"]), int(cmeta["l_rev"])
    if lf_p != lf_c:
        raise NotImplementedError(
            f"the two channels read at different depths ({lf_p} and {lf_c}); the bench "
            "has one read point, so a dual guard-rail needs them to agree")
    coupler = DualCoupler(phys, fno, cphi, const, lr_p, lr_c, channels=channels)
    meta = {"step": f"phys {pmeta.get('step')} / const {cmeta.get('step')}",
            "model": pmeta.get("model"), "physics": pmeta, "constitution": cmeta,
            "channels": channels, "l_rev_constitution": lr_c, "l_rev_physics": lr_p}
    return coupler, lf_p, lr_c, meta
