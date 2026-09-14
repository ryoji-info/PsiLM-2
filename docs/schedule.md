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
reason to stay out of each other's way. That matters because the gate is what does
the safety work: at 9B the constitution channel's KL to the base was **302×**
larger on red-team prompts than on arithmetic, and that selectivity — not the
write mask — is why writing all 4096 dimensions cost nothing on any benchmark. A
composition that let both gates drift open on everything would discard exactly the
property the campaign established.

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

Assertion 5 is the design under test rather than the code: if a step on the cross
penalty did not move the other gate, the penalty would be decoration.

## Running it

```bash
PYTHONPATH=../PsiLM python -m psilm2.train_dual --dry-run --steps 12   # CPU, no weights
PYTHONPATH=../PsiLM python -m psilm2.train_dual --phase coexist --steps 600
```

Chunked like every PsiLM trainer: `--steps` per invocation, resuming from its own
`bridges.safetensors`, so a Metal watchdog kill costs one chunk rather than the
run. The supervisor loop belongs in a shell script, as it does for the
single-channel campaigns.

**This has not been run on the 9B.** The schedule is designed, implemented and
tested on a tiny stack; whether the cross-gate penalty actually holds two gates
apart on a real backbone is the first thing a run would tell us, and it is not
something the design can settle by itself.
