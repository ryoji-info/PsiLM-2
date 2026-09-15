<p align="center"><b>ΨLM-2 — one frozen language model, two frozen partners: a physics model and a document of values, each reached through small trainable bridges.</b></p>

## What ΨLM-2 is

ΨLM-2 keeps the ΨLM premise — nothing pretrained is fine-tuned — and puts two
partners on the same frozen backbone at once:

- a **physics bridge**, which reads a PDE's inputs out of the backbone's hidden
  states, lets a neural operator solve it, and returns the value as soft tokens;
- a **constitution bridge**, which reads the prompt out of the same stream, lets
  a small partner model that holds [Claude's constitution](https://www.anthropic.com/constitution)
  deliberate over it, and returns *that* as soft tokens.

The backbone is **Qwen3.5 9B** (frozen, 4-bit, 32 layers, d = 4096). Only the
bridges train: a few tens of millions of parameters against nine billion frozen
ones. No text crosses either interface — both partners are reached and answered
entirely in latent space.

The aim is the one the physics bridge already demonstrates for a *quantity*,
applied to a *disposition*: the backbone acquires something it did not have,
through a channel narrow enough to audit and a gate that can be closed. Where
the physics bridge supplies a number the model cannot compute, the constitution
bridge is meant to supply the reasoning about values the model would otherwise
only approximate.

Whether it does is an empirical question, and the honest answer today is
*partly, and not yet at a useful width*. See [docs/status.md](docs/status.md)
before drawing conclusions from anything here.

## Where the constitution bridge writes

The injection site is not arbitrary. Following [arXiv:2602.00986](https://arxiv.org/abs/2602.00986),
a probe ranks residual-stream coordinates by how well they predict whether the
model's own continuation will be correct — its "value neurons" — and the bridge
writes only into those, through a frozen boolean mask. At this backbone's layer
24 the top 1% is 41 of 4096 dimensions.

**That is the design; it is not what the trained stack uses.** At 9B the mask turned
out to buy nothing — every width is harmless because the gate, not the mask, does the
safety work — while a 41-dimension write changed no behaviour at all. So the
constitution channel here writes all 4096 dimensions, and the mask survives as the
mechanism that made the comparison possible rather than as part of the recipe. See
[docs/status.md](docs/status.md).

Two findings from ΨLM qualify the choice of site, and both are load-bearing here:

- The paper's **causal** claim does not reproduce on this backbone. Zeroing the
  41 value neurons costs 3 GSM8K points at p = 0.375; random draws of the same
  size cost nothing. At 0.5B, where zeroing *did* hurt, magnitude-matched
  controls reproduced the damage — so the effect there was activation magnitude,
  not value.
- The **probe** evidence survives, because a probe rescales its own inputs:
  the top 1% reaches AUC 0.788 against 0.821 at full width.

So the value neurons are a defensible place to write into, on probe evidence,
and not because zeroing them breaks the model.

## The code

`psilm2.dual.PsiDualMLX` runs both channels in one forward pass, reading and
injecting at each channel's own depth. It subclasses `psilm.mlx.model.PsiLMMLX`
and overrides only the coupling step, so the physics loss, the no-harm arm and
the physics generate path are inherited rather than copied — there is no second
version of them to drift.

The channels share a residual stream, so whichever writes lower changes what the
other's gate and attention see. That interaction is the point of the composition
and the module is built to expose it rather than absorb it. `PsiDualMLX` takes a
per-channel arm — `psilm` writes, `zeroed` builds the tokens and measures the
gate without writing, `off` does not touch the bridge — which is what makes the
interaction measurable from either side.

`python -m psilm2.dual_self_test` proves nine properties on a tiny random stack
on the CPU, in a few seconds and with no weights. Three of them are bit-identity
with the models ΨLM already trains, so that any difference in a dual run is the
other channel's presence and not a different code path; one of them demands that
the physics gate **does** move when the constitution opens below it. See
[docs/self-test.md](docs/self-test.md).

## Status

**The dual stack is built, verified on the real 9B, and measured in three arms** —
and the headline result is that the joint training phase is not worth running. The
two channels compose for free: the untrained warm start costs +0.00027 of
constitution cross-entropy (95% [−0.00038, +0.00093], spanning zero) against the
channel's own effect of −0.094, and physics holds 1.000 held-out accuracy with the
constitution channel open. Both trained arms pay more than that and fail acceptance.

`load_dual_stack` has been run against the real checkpoints; the three bit-identity
properties hold there as well as on the tiny stack. [docs/status.md](docs/status.md)
records what is measured and what it does not establish — read it before treating
anything here as an outcome, and note that the composition was measured on
cross-entropy and accuracy only, not on refusal behaviour or a guard-rail suite.

## Related

- [ΨLM](https://github.com/ryoji-info/PsiLM) — the physics-bridge work this builds on, including the value-neuron and constitution-bridge experiments at 0.5B and 9B.

## License

Apache 2.0. Claude's constitution is released by Anthropic under CC0 1.0; the
copy used to build the partner model carries its provenance alongside it.
