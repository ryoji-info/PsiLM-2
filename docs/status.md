# What is measured, what is running, what does not exist

Last updated 2026-09-13. Every number here was produced by the harness in
[ΨLM](https://github.com/ryoji-info/PsiLM); none of it was produced by code in
this repository yet.

## Implemented, not trained (2026-09-15)

**The composition exists as code and has never been trained.** `psilm2.dual`
runs both channels in one pass; nine assertions on a tiny CPU stack pin down that
it reduces exactly to each single-channel model when the other channel is off,
and that the two channels really do share a residual stream — the physics gate
moves when the constitution writes below it, and the physics bridge receives
gradient from the constitution's objective, which contains no physics targets.
See [self-test.md](self-test.md).

What that does *not* settle is everything a run would:

- **The training schedule** was written, run on the 9B in three arms, and the
  answer is **don't run it** — see [schedule.md](schedule.md). The untrained
  composition already costs +0.0003 of constitution CE (95% [−0.0004, +0.0009],
  spanning zero) and clears acceptance; both trained arms fail it, paying +0.0035
  and +0.0066 of absolute CE to remove an interference that was at the noise
  floor. Warm-start both channels and leave the composition alone.
- ~~Whether the gates stay selective when both are open.~~ **Answered:** they do,
  without training. At warm start the two trained gates already separate by task
  about fiftyfold, and the residual interference costs +0.0003 of CE. The
  cross-penalty improves the separation by a further 43× on physics prompts and
  that improvement buys nothing measurable.
- ~~Whether the two injections interfere.~~ **Answered, for this pair:** they do
  not measurably. The constitution channel here writes all 4096 dimensions at layer
  24 and the physics channel writes the whole stream at 26, so one write passes
  straight over the other — and the physics channel still scores 1.000 with the
  constitution open, while the constitution channel's CE moves by +0.0003. Untested
  for the narrow (41-dimension) constitution write, which is the variant the 9B
  width campaign found behaviourally inert anyway.
`load_dual_stack` **has** now been run against the real 9B checkpoints
(`python -m psilm2.verify_qwen35`, 2026-09-15): physics 13/26 with the value
channel, constitution 13/24 writing all 4096 dimensions, 45.17M trainable
parameters, 10.0 GB peak. All three bit-identity properties hold on the real
stack, and the interaction is measurable there too — the physics gate moves from
0.001764 to 0.001757 when the constitution opens below it. On a red-team prompt
the constitution gate sits at 0.0153 against the physics channel's 0.0018, which
is the two gates behaving selectively in the composition, and the only evidence so
far that they will.

The value of the code as it stands is that it makes those four questions
answerable by a run rather than by an argument.

## Measured

**Physics bridge on Qwen3.5 9B.** Trained and published
([ryoji-info/Qwen3.5-9B-PsiLM](https://huggingface.co/ryoji-info/Qwen3.5-9B-PsiLM)).
Its results, guard-rails and caveats are documented in ΨLM's technical notes.

**Value neurons on Qwen3.5 9B** (following [arXiv:2602.00986](https://arxiv.org/abs/2602.00986)).
800 GSM8K-train trajectories at the paper's sampler, Monte-Carlo probe target,
seven candidate depths. Layer 24 chosen by held-out AUC after pruning to 1%:

| | full width (4096) | top 1% (41 dims) | 41 uniform random |
|---|---:|---:|---:|
| AUC at layer 24 | 0.821 | 0.788 | 0.712 |

The gap to random is about 1.5 conservative standard errors (Hanley–McNeil on
119 correct / 41 incorrect held-out trajectories) — suggestive, not established.
The paper's *causal* claim does not reproduce here: zeroing the 41 value neurons
costs 3 GSM8K points at p = 0.375, zeroing the top 5% costs 1, and three random
draws of 41 cost nothing. At 0.5B the same intervention cost 13 points, but
magnitude-matched controls reproduced that damage, so it was activation
magnitude rather than value.

**Constitution bridge on Qwen3.5 9B, narrow write** (`vn`: 41 of 4096 dims at
layer 24, read at layer 13, 28.34M bridge parameters, 1,000 steps). Teacher =
the same frozen backbone with a 3,422-token constitution excerpt as its system
prompt; student = the backbone with a plain system prompt plus the bridges.

| n = 50, held-out test | CE to teacher | top-1 agreement | refusal | gate on generated | injection, % of local stream |
|---|---:|---:|---:|---:|---:|
| base | 0.4829 | 0.8599 | 0.6200 | — | — |
| bridges open | **0.4702** | 0.8595 | 0.6200 | 0.474 | 9.5% |
| bridges zeroed | 0.4829 | 0.8599 | 0.6200 | 0.474 | 9.5% |
| teacher (stored, 128 tok) | — | — | 0.8000 | — | — |
| base (stored, 128 tok) | — | — | 0.6600 | — | — |

Three readings, in order of what they settle:

1. **The channel is exactly inert when gated off.** The zeroed arm reproduces
   base to four decimals on every column, which is the masked-write module's
   correctness claim holding at 4096 dimensions.
2. **It carries real information.** Held-out CE falls 0.4829 → 0.4702, matching
   the validation move of 0.4720 → 0.4589.
3. **It changes no behaviour.** Refusal is identical in both arms and McNemar
   returns b = 0, c = 0, p = 1.0 — not one item of fifty flipped either way,
   while the teacher being distilled flips nine (p = 0.0039). This is not a
   closed gate: the gate sits at 0.474 and the injection runs at 9.5% of the
   local stream, inside its 0.2 cap.

Training had converged: a second 500-step chunk moved held-out CE by 0.0004, so
(3) is a capacity ceiling at 41 coordinates, not undertraining.

## Running

Two further write widths on the same backbone and recipe, to separate width from
location: `all` (the whole 4096-dimension stream) and `vn5` (the top 5%, 205
dims). Each gets the same two held-out evaluations and a four-dataset guard-rail
(red-team, GSM8K, MMLU, BoolQ; base / open / zeroed arms; KL to base). At 0.5B
the full-width write did carry the teacher's behaviour — and cost 8 GSM8K points
for it, with the same over-refusal on ordinary requests the teacher shows. The
open question this answers is whether harm is set by the width of the write or
by which coordinates it lands in; at 0.5B the evidence said width.

## The caveat that governs all of it

At 0.5B, a control with an *untouched* partner model — same architecture, no
constitution in its weights — matched its constitution-partner twin within noise
at every width. So the document in the partner's weights was not what the
channel carried; the constitution entered through the teacher, and the partner
served as a latent scratchpad. Whether a partner that reasons with the document
rather than reciting it behaves differently is untested, and is the question the
combined stack should be designed to answer rather than assume.
