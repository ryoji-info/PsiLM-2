# What the dual-bridge self-test establishes

```
PYTHONPATH=../PsiLM python -m psilm2.dual_self_test
```

Everything below runs on a random 6-layer Qwen2 at d_model 64 on the **CPU**, so
it never queues work on the shared GPU and needs no weights. It takes a few
seconds.

The first three assertions exist for attribution. A dual run differs from a
single-channel run for two possible reasons — the other channel is present, or
the dual code path is not the same code path — and only the first is
interesting. These rule out the second:

| | assertion | why it matters |
|---|---|---|
| 1 | both channels off is **bit-identical** to the bare staged backbone | the composition adds nothing when closed |
| 2 | physics only is **bit-identical** to `psilm.mlx.model.PsiLMMLX` | the physics results carry over unchanged |
| 3 | constitution only is **bit-identical** to `psilm.mlx.constitution.PsiConstitutionMLX` | so do the constitution results |

The rest characterise the composition:

| | assertion |
|---|---|
| 4 | both open is a different function from either alone, and from neither |
| 5 | a `zeroed` arm leaves the stream exactly alone while still measuring its gate |
| 6 | one channel writing while the other is only measured is bit-identical to the writer alone — so `zeroed` cannot leak |
| 7 | **the physics gate moves when the constitution opens below it** (up to 0.0077 on the tiny stack) |
| 8 | an interleaved depth configuration raises `NotImplementedError` rather than silently picking an order |
| 9 | a 3-dimension masked constitution write touches exactly those 3 dimensions, with the physics channel writing the whole stream in the same pass |

Assertion 7 is the one worth staring at. It is the composition's only real
coupling: the constitution writes at a lower layer than the physics channel, so
the physics gate reads a stream the constitution has already modified. The
assertion demands that this be **non-zero** — if the physics gate were identical
with and without the constitution write below it, the two channels would not be
sharing a residual stream and the whole exercise would be two independent models
in a trench coat.

The size line reads two numbers on purpose. The stack holds **73.52M** bridge
parameters — physics 45.18M plus constitution 28.34M — and only 45.17M of them are
trainable *as loaded*, because `load_constitution_stack` freezes its bridge and the
verification path does not ask for `trainable=True`. An earlier version of this
document reported 45.17M as the stack's size, which was the physics bridge alone;
that number is the visible signature of the freeze bug that invalidated the first
training run, so `describe()` now prints both and a trainer's own log prints 73.52M.

## The schedule's assertions

`psilm2.schedule_self_test` asserts eleven properties, not the eight this document
first listed. The three added after the first run failed are the ones that catch its
bugs: **9** a chunked run's step count is cumulative and its weights continue, **10**
chunk 2 does not trace chunk 1's trajectory (the replay symptom), and **11** an
active frozen channel is refused with a named error rather than trained for zero
steps.

## The gradient test

Separately, `gradient_test()` checks that both bridges are reachable by an
optimizer, because a dual model where only one is would train happily and be
silently single-channel. From one `loss_constitution` call:

```
physics      34 of 36 parameter tensors have a non-zero gradient
constitution 23 of 23 parameter tensors have a non-zero gradient
```

The physics bridge receiving gradient from the *constitution's* objective — which
has no physics targets in it — is the same coupling assertion 7 measures, seen
from the backward pass.

The two exceptions are named rather than merely counted: `rev.u_head.weight` and
`rev.u_head.bias` produce the scalar `u_hat`, which feeds only the physics
deep-supervision term `loss_u` and never the token path (in the `value` channel
form it sits behind a `stop_gradient` as well). A pure cross-entropy objective
correctly reaches everything else and not those. The test asserts that exact set,
so a future change that silently detaches a *different* tensor fails here instead
of training a dead parameter.

## On the real stack

`python -m psilm2.verify_qwen35` re-checks the three attribution identities on the
actual Qwen3.5 9B backbone and the actual trained checkpoints, where a loader
mistake, a depth read from the wrong meta or a dtype difference would show up
instead of a logic error:

```
PsiDualMLX over 32 layers | physics 13/26 channel=value
                          | constitution 13/24 write 4096 of 4096 (100.00%)
                          | 73.52M bridge parameters (45.17M currently trainable)
  both off      vs bare staged backbone  IDENTICAL
  physics only  vs PsiLMMLX              IDENTICAL
  constitution  vs PsiConstitutionMLX    IDENTICAL
  physics gate alone 0.001764 -> with the constitution open 0.001757
  constitution gate on this prompt: 0.0153
peak 10.0 GB
```

Inference only, a few seconds per arm, and it needs the GPU free.
