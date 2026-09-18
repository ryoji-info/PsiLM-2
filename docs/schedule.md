# The training schedule for the dual stack

## The signal neither single-channel run could produce

Both campaigns trained their gate to shut on *off-task* prompts, and for both of
them off-task meant GSM8K and MMLU (`data/noharm_qwen35_all.json`). Neither
channel was ever shown **the other channel's on-task prompts as negatives**,
because in a single-channel run there was no other channel. So nothing in either
trained gate knows that:

| prompt | physics gate | constitution gate | trained? |
|---|---|---|---|
| a Burgers question | open | **shut** | no — new |
| a red-team prompt | **shut** | open | no — new |
| GSM8K / MMLU | shut | shut | yes, both |

Two gates each trained to open on "their" prompts and shut on arithmetic have no
reason to stay out of each other's way. That matters because the gate is what keeps
each channel off the other's prompts: at 9B the constitution channel's KL to the
base was **302×** larger on red-team prompts than on arithmetic, and that
divergence selectivity — not the write mask — is why writing all 4096 dimensions
left GSM8K, MMLU and BoolQ within one or two items of the backbone while moving
red-team refusal 0.660 → 0.720. (The paper later showed the selectivity is not
the gate's own doing: measured on the gate, the red-team/arithmetic ratio *falls*
with width, 547×/364×/109×, and the wide write is safe because its divergence
lands on the prompts the document speaks to.) A composition that let both gates
drift open on everything would still discard the property that makes the two
channels observable from either side.

So the schedule's one substantive addition is a **cross-gate penalty**: on each
task's own batches, the *other* channel's mean gate is penalised. It is the
mechanism the no-harm arm already uses, pointed at a negative the single-channel
runs did not have.

```
physics batch:      L = L_physics      + λ_cross · gate_constitution
constitution batch: L = L_constitution + λ_cross · gate_physics
no-harm batch:      L = CE(own cont.)  + λ_gate  · (gate_physics + gate_constitution)
```

One physics, one constitution, one no-harm, in rotation. Each channel therefore
meets pressure to shut on two of every three steps and to open on one. The
single-channel runs ran their no-harm arm every second step, so this is a
comparable duty cycle reached from the other direction.

## Phases

| | phase | trains | why |
|---|---|---|---|
| 0 | parity | nothing | `psilm2.verify_qwen35` confirms the warm-started dual reproduces both single-channel models bit for bit, and their held-out metrics become the baseline the composition must not regress |
| 1 | **coexist** | the two gate MLPs only | the readouts, token heads and injections already work; what is new is each gate's decision function, now that its input stream can carry the other channel's write |
| 2 | finetune | both bridges, lr/3 | only if phase 1 leaves a regression, early-stopped on whichever metric regressed |

Phase 1 before phase 2 is not caution for its own sake — it states the hypothesis
out loud. If adjusting two gate MLPs is enough, then nothing that took 11,500
physics steps to learn was put at risk to find out. It is also cheap for a
structural reason: with only the gates trainable, no gradient flows below the
injection depths, so the backward pass spans layers 24–32 rather than 13–32 and
needs no gradient checkpointing.

600 steps of the 3-task cycle gives each arm 200 steps. That is calibrated to what
the campaigns measured rather than to habit: the constitution channel converged
inside 500 steps at 9B and every width plateaued by chunk 2, so 200 steps is
generous for a phase that only has to move two gate MLPs.

## Hyperparameters

`λ_gate = 1.0` is the value both campaigns used. **`λ_cross` is the one genuinely
new hyperparameter**, started at 1.0 on the grounds that "shut on the other task's
prompts" is the same demand as "shut on arithmetic". Sweep it only if phase 1
misses its acceptance test, and sweep it upward: the failure mode it addresses is a
gate that stays open where it should not.

Acceptance is a no-regression test against phase 0's baselines, with tolerances
taken from the campaigns' own noise rather than picked — the width campaign's
chunk-to-chunk CE movement at convergence was 0.001, so the constitution tolerance
is 0.003, three times the drift convergence itself showed.

## One trap, worth naming

Gate-only is implemented by zeroing the non-gate gradients — and that is only half
of it. **AdamW decays every parameter it owns whether or not the gradient was
zero**, so a 600-step gate-only phase under AdamW would quietly shrink readouts
that took 11,500 physics steps to train, while reporting itself as frozen.
`run_phase` therefore uses plain Adam for gate-only phases, where a zero gradient
leaves a parameter bit-identical.

Assertion 8 of `psilm2.schedule_self_test` holds that promise: after a gate-only
phase, 8 gate tensors have moved and 0 of the other 57 have, bit for bit. Putting
AdamW back makes it fail and name the decayed physics readouts, so the test has
teeth rather than passing vacuously.

## What the self-test establishes

```
PYTHONPATH=../PsiLM python -m psilm2.schedule_self_test
```

1. the interleave is physics/constitution/no-harm, 1:1:1, deterministic
2. the no-harm flag is set on exactly the no-harm steps
3. the gate-only mask keeps 8 leaves, all `inject.g1/g2`, in both channels
4. the cross penalty raises the loss by the other channel's mean gate
5. **one descent step on the cross penalty lowers the physics gate 0.364 → 0.036**
6. the acceptance test tolerates 0.0025 of CE drift and refuses 0.006
7. the physics task path runs inside the schedule, with its cross penalty attached
8. **a gate-only phase moves 8 gate tensors and 0 of 57 others**
9. a chunked run's step count is cumulative across chunks
10. chunk 2 does not trace chunk 1's trajectory (the replay test)
11. an active frozen channel is refused with a named error, not trained for zero steps

Assertion 5 is the design under test rather than the code: if a step on the cross
penalty did not move the other gate, the penalty would be decoration. Assertions
9–11 were added after the first 9B run exposed the bugs they catch
([self-test.md](self-test.md)).

## Running it

```bash
PYTHONPATH=../PsiLM python -m psilm2.train_dual --dry-run --steps 12   # CPU, no weights
PYTHONPATH=../PsiLM PSILM_BACKBONE=<backbone dir> python -m psilm2.train_dual --phase coexist --steps 600
```

Chunked like every PsiLM trainer: `--steps` per invocation, resuming from its own
`bridges.safetensors`, so a Metal watchdog kill costs one chunk rather than the
run. The supervisor loop belongs in a shell script, as it does for the
single-channel campaigns.

## First contact with the 9B (2026-09-15)

Three things the real stack said that the tiny one could not.

**The grad window is real; the speedup I first claimed for it was not.** Qwen3.5's
GatedDeltaNet layers run a Metal kernel with no VJP, so the differentiable ops-path
scan must cover exactly the layers the backward pass touches, and gate-only needs it
from layer **24** (the shallowest injection) rather than 13 (the shallowest read).
That much holds. The "10.4 s/step, 2.6× faster, 600 steps in 1h45m" this document
first reported does not: **10.4 s/step came from the smoke run, which carried the
frozen-constitution bug** described below, so it timed a backward pass that reached
only one bridge. The valid arms ran at **23.1–24.4 s/step** and took **3h52m** and
**3h57m** for 600 steps — against the constitution campaign's ~25 s/step, a real
speedup of roughly **1.1×, not 2.6×**.

A second bug made that harder to see. `run_phase` computed `s/step` as this chunk's
elapsed time divided by the *cumulative* step, so chunk 2 read 2× fast and chunk 3
3× fast, and a flat 23 s/step run appeared to accelerate to 11.6 and then 7.8. Fixed
to divide by the chunk-local step. The corrected per-chunk figures are 23.08 / 23.26
/ 23.37 (λ_cross = 1) and 23.55 / 23.20 / 24.42 (λ_cross = 0).

**The no-harm negatives are in a different schema**, and the schedule reads them in
their own: `data/noharm_qwen35_all.json` comes from `eval/build_noharm.py` and uses
`target_ids`, where the constitution data uses `base_ids`. Both campaigns share the
one file. The source now checks the schema and says what it found if it is wrong.

**And the honest one, which deflates the rationale above.** After six steps of the
smoke run the gates were already sorted by task. (That run carried the
frozen-constitution bug, so its constitution gate is the warm-started one and never
trained — which is what makes it a clean read of the warm start, and why its
*timings* above had to be thrown out:)

| batch | physics gate | constitution gate |
|---|---:|---:|
| physics | 0.731 | 0.014 |
| constitution | 0.008 | 0.418 |
| no-harm | 0.008 | 0.005 |

Fifty-fold separation in both directions, at warm start, before the cross-gate
penalty could have done much. The premise of this schedule — that neither gate has
seen the other channel's on-task prompts as negatives — is literally true, but the
*generalisation* covers it better than the argument assumed: the constitution gate
was trained to shut on GSM8K and MMLU, and a Burgers question looks enough like
arithmetic that it shuts on that too. So the cross-gate penalty may be largely
redundant rather than load-bearing. That is worth stating plainly, because the
alternative is to run the phase, observe selectivity, and credit the mechanism that
happened to be switched on.

What the full run can still settle is whether the separation *holds* under
training, since a joint objective could as easily erode it as preserve it, and
whether either channel's own held-out metric regresses. The λ_cross = 0 ablation is
the comparison that would actually attribute it, and it costs the same ~3h55m.

## The result: don't run it (2026-09-15)

Three arms, each measured by `psilm2.accept` on the same fifty held-out red-team
items and the same fifty held-out physics items, against the single-channel
numbers the same harnesses produced:

| arm | const CE alone | both open | cost of opening the other channel | verdict |
|---|---:|---:|---|---|
| **warm-start, untrained** | **0.38905** | **0.38931** | **+0.00027, 95% [−0.00038, +0.00093]** | **ACCEPTED** |
| phase 1, λ_cross = 1 | 0.39250 | 0.39243 | −0.00007, no interval | regressed |
| phase 1, λ_cross = 0 | 0.39564 | 0.39658 | +0.00094, no interval | regressed |

Only the warm-start arm carries per-item cross-entropies: `accept.py` gained them
after the two trained arms were measured. So neither trained arm's composition cost
has an interval, and neither is claimed to be distinguishable from zero. For scale,
the warm-start arm's per-item differences have SD 0.0024 and SEM 0.00034.

Physics scored 1.000 accuracy in every arm, at MAE 0.0141–0.0150 against the
campaign's own 0.0147, and the bare backbone scores 0.000 — so the channel does
all of that work and none of it is disturbed by the constitution channel being
open.

**The two channels compose for free without any joint training.** The untrained
composition costs +0.0003 of constitution cross-entropy, with a paired bootstrap
interval that spans zero, and its top-1 agreement (0.8686) is the highest of any
arm measured. It clears the acceptance test that both trained arms fail.

**Training the composition makes it worse.** The cross-gate penalty did what it was
designed to do — it drove the constitution gate 39× lower on physics prompts than
the control did (0.00022 against 0.00865; the 43× first reported here came from
reading a 4-decimal log rather than the 5-decimal record), and ~14× lower on
no-harm prompts — but it paid
+0.0035 of absolute CE to remove an interference of +0.0003 that was not
distinguishable from zero to begin with. The control paid +0.0066 and left more
interference than the untrained stack had. Both are bad trades.

So the recommendation this document exists to give is: **warm-start both channels
from their trained checkpoints and do not train the composition.** Phase 0 is the
whole recipe. The schedule below it is correct, tested and unnecessary.

Why it was worth building anyway: nothing above could be known without it. The
cross-penalty arm alone would have shown selective gates and been credited for
them; the control is what showed the no-harm arm was not doing that work; and the
untrained arm is what showed the work did not need doing. The three-arm structure,
not the schedule, is what produced a usable answer — and the same lesson the 0.5B
magnitude-matched draws taught, met again at a different level.

**The penalty worked in one direction only.** It tightened the constitution gate as
intended, and over the same 600 steps the *physics* gate became less selective in
both arms — on constitution batches 0.0110 → 0.0161 and on no-harm batches 0.0113 →
0.0292 under the penalty, which is the term the penalty was pushing down. The
control drifted the same way. Absolute values stay small (0.77 on-task against 0.016
off-task is still ~48× selective), and these are single-batch snapshots rather than
averages, but "it did what it was designed to do" is half the story: the gate with
room to improve improved, and the gate that was already tight eroded.

One caveat on the premise. The argument for a cross-gate penalty was that neither
gate has seen the other channel's on-task prompts as negatives. That is true, and
the penalty measurably acts on it. What is false is the assumption that it
mattered: the gates generalise across tasks well enough on their own that the
residual interference is at the noise floor. A schedule can be well-motivated,
correctly implemented, effective at its stated mechanism, and still not worth
running.
