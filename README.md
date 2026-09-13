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

Two findings from ΨLM qualify that choice, and both are load-bearing here:

- The paper's **causal** claim does not reproduce on this backbone. Zeroing the
  41 value neurons costs 3 GSM8K points at p = 0.375; random draws of the same
  size cost nothing. At 0.5B, where zeroing *did* hurt, magnitude-matched
  controls reproduced the damage — so the effect there was activation magnitude,
  not value.
- The **probe** evidence survives, because a probe rescales its own inputs:
  the top 1% reaches AUC 0.788 against 0.821 at full width.

So the value neurons are a defensible place to write into, on probe evidence,
and not because zeroing them breaks the model.

## Status

Nothing in this repository is a finished result. The two bridges exist and are
trained **separately**; the combined stack that this repository is named for has
not been built yet. [docs/status.md](docs/status.md) records exactly what is
measured, what is running, and what does not exist.

## Related

- [ΨLM](https://github.com/ryoji-info/PsiLM) — the physics-bridge work this builds on, including the value-neuron and constitution-bridge experiments at 0.5B and 9B.

## License

Apache 2.0. Claude's constitution is released by Anthropic under CC0 1.0; the
copy used to build the partner model carries its provenance alongside it.
