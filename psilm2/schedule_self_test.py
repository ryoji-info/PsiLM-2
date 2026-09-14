"""Self-test for the dual training schedule, on the tiny stack, CPU only.

    PYTHONPATH=../PsiLM python -m psilm2.schedule_self_test

Plumbing first (the interleave, the gradient mask), then the one assertion that
tests the design rather than the code: a step under the cross-gate penalty must
push the OTHER channel's gate DOWN. If it did not, the penalty would be decoration
and the two gates would have no reason to stay out of each other's way.
"""

import json

import mlx.core as mx
import mlx.nn as nn

from psilm.mlx.constitution import pad_batch, synthetic_items

from .dual import CONST, PHYS
from .schedule import (Baselines, DualSchedule, GATE_KEYS, PHASES, Phase,
                       gate_only_grads, trainable_leaf_names)
from .dual_self_test import build_tiny_dual, _ids


def _phys_batch(B=2, n_prompt=7, n_cont=4, vocab=256, seed=9):
    """A physics batch with every key PsiLMMLX.loss_fn reads, from random values.

    Enough to drive the loss and its gradient; the targets are meaningless, which
    is fine because nothing here checks that the physics task LEARNS -- only that
    its loss runs inside the schedule and that the penalties attach to it.
    """
    mx.random.seed(seed)
    L = n_prompt + n_cont
    import numpy as np
    rng = np.random.default_rng(seed)
    ids = rng.integers(3, vocab, (B, L)).astype(np.int32)
    lb = np.full((B, L), -100, dtype=np.int32); lb[:, n_prompt:] = ids[:, n_prompt:]
    pm = np.zeros((B, L), bool); pm[:, :n_prompt] = True
    return {"p_ids": mx.array(ids), "p_attn": mx.array(np.ones((B, L), np.int32)),
            "p_labels": mx.array(lb), "prompt_mask": mx.array(pm),
            "x0_span": mx.array(np.tile([[1, 4]], (B, 1)).astype(np.int32)),
            "params": mx.array(rng.normal(size=(B, 3)).astype(np.float32)),
            "x0": mx.array(rng.random(B).astype(np.float32)),
            "x0_bins": mx.array(rng.integers(0, 64, B).astype(np.int32)),
            "u_true": mx.array(rng.normal(size=B).astype(np.float32))}


def self_test(verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    S = build_tiny_dual(write_idx=[1, 7, 13], seed=4)
    psi = S["dual"]
    ids, pm = _ids()
    const_batch = pad_batch(synthetic_items(2, n_prompt=7, n_cont=4), pad_id=0)
    sources = {"physics": lambda: _phys_batch(),
               "constitution": lambda: dict(const_batch),
               "noharm": lambda: dict(const_batch)}

    # 1. the interleave is 1:1:1 and deterministic
    sch = DualSchedule(PHASES[0], sources)
    seen = [sch.next()[0] for _ in range(9)]
    assert seen == ["physics", "constitution", "noharm"] * 3, seen
    assert sch.counts == {"physics": 3, "constitution": 3, "noharm": 3}, sch.counts
    say("1. interleave       physics/constitution/noharm, 1:1:1, deterministic")

    # 2. a no-harm batch is flagged so the model's own gate penalty engages
    sch2 = DualSchedule(PHASES[0], sources)
    flags = [sch2.next()[1].get("noharm", False) for _ in range(6)]
    assert flags == [False, False, True, False, False, True], flags
    say("2. noharm flag      set on exactly the no-harm steps")

    # 3. gate_only_grads keeps the two gate MLPs in BOTH channels and nothing else
    psi.set_modes(physics="psilm", constitution="psilm")

    def loss_of(params, task="constitution", phase=PHASES[0]):
        psi.phi.update(params[PHYS]); psi.cphi.update(params[CONST])
        return DualSchedule(phase, sources).loss(psi, task, dict(const_batch))[0]

    p0 = {PHYS: psi.phi.trainable_parameters(), CONST: psi.cphi.trainable_parameters()}
    loss, grads = mx.value_and_grad(loss_of)(p0)
    mx.eval(loss, grads)
    masked = gate_only_grads(grads)
    mx.eval(masked)
    kept = trainable_leaf_names(masked)
    assert kept, "the mask zeroed everything"
    for n in kept:
        assert ".inject.g1." in n or ".inject.g2." in n, f"mask kept a non-gate leaf: {n}"
    assert any(n.startswith(PHYS) for n in kept), "no physics gate survived the mask"
    assert any(n.startswith(CONST) for n in kept), "no constitution gate survived the mask"
    say(f"3. gate-only mask   keeps {len(kept)} leaves, all inject.g1/g2, both channels")

    # 4. the cross penalty is real: it changes the loss, and only on task batches
    with_cross = Phase("t", 1, 3e-4, True, lam_cross=1.0, lam_gate=1.0)
    no_cross = Phase("t", 1, 3e-4, True, lam_cross=0.0, lam_gate=1.0)
    lc = DualSchedule(with_cross, sources).loss(psi, "constitution", dict(const_batch))[0]
    l0 = DualSchedule(no_cross, sources).loss(psi, "constitution", dict(const_batch))[0]
    mx.eval(lc, l0)
    assert float(lc.item()) > float(l0.item()), "the cross penalty did not raise the loss"
    say("4. cross penalty    raises the constitution loss by %.5f (the physics gate's mean)"
        % (float(lc.item()) - float(l0.item())))

    # 5. THE DESIGN ASSERTION. One optimizer step on the cross penalty alone must
    #    push the PHYSICS gate down, on a constitution batch. This is what makes
    #    the two gates stay out of each other's way rather than both drifting open.
    def cross_only(params):
        psi.phi.update(params[PHYS]); psi.cphi.update(params[CONST])
        psi.loss_constitution(dict(const_batch))          # forward, for psi.last
        g = psi.last.gate(PHYS)
        valid = const_batch["p_attn"].astype(mx.float32)
        return (g * valid).sum() / (valid.sum() + 1e-6)

    before = float(cross_only(p0).item())
    _, cg = mx.value_and_grad(cross_only)(p0)
    mx.eval(cg)
    step = 0.5
    moved = {PHYS: dict(p0[PHYS]), CONST: p0[CONST]}
    inj = dict(moved[PHYS]["inject"])
    for k in GATE_KEYS:                                  # descend on the gate only
        inj[k] = {kk: vv - step * cg[PHYS]["inject"][k][kk] for kk, vv in inj[k].items()}
    moved[PHYS]["inject"] = inj
    after = float(cross_only(moved).item())
    assert after < before, (f"a descent step on the cross penalty did not lower the "
                            f"physics gate: {before:.6f} -> {after:.6f}")
    say("5. cross penalty    one descent step lowers the physics gate %.6f -> %.6f"
        % (before, after))

    # 6. the acceptance test refuses a regression and passes a tie
    b = Baselines(const_ce=0.3890, phys_acc=1.0)
    assert b.accept(const_ce=0.3890, phys_acc=1.0)[0]
    assert b.accept(const_ce=0.3915, phys_acc=1.0)[0], "0.0025 is inside the 0.003 tolerance"
    ok, why = b.accept(const_ce=0.3950, phys_acc=1.0)
    assert not ok and "constitution CE" in why[0], why
    ok, why = b.accept(const_ce=0.3890, phys_acc=0.90)
    assert not ok and "physics acc" in why[0], why
    say("6. acceptance        tolerates 0.0025 of CE drift, refuses 0.006 and a 10-point"
        " physics drop")

    # 7. the physics task path runs inside the schedule too
    lp, sp = DualSchedule(with_cross, sources).loss(psi, "physics", _phys_batch())
    mx.eval(lp)
    assert mx.isfinite(lp).item(), "the physics task loss is not finite"
    assert "cross" in sp, "no cross penalty was applied on a physics batch"
    say("7. physics task      loss %.4f finite, with the constitution gate penalised"
        % float(lp.item()))

    # 8. GATE-ONLY MUST REALLY FREEZE. Zeroing the non-gate gradients is only half
    #    of it: AdamW decays every parameter it owns whether or not the gradient
    #    was zero, so a 600-step gate-only phase under AdamW would quietly shrink
    #    readouts that took 11,500 physics steps to train. run_phase uses Adam for
    #    gate-only phases for exactly that reason, and this asserts the promise --
    #    every non-gate parameter bit-identical after a phase.
    from mlx.utils import tree_flatten
    from .train_dual import run_phase

    T = build_tiny_dual(write_idx=[1, 7, 13], seed=6)
    tp = T["dual"]
    tp.set_modes(physics="psilm", constitution="psilm")
    before = {f"{c}.{k}": v for c, m in ((PHYS, tp.phi), (CONST, tp.cphi))
              for k, v in tree_flatten(m.parameters())}
    mx.eval(list(before.values()))
    before = {k: mx.array(v) for k, v in before.items()}
    import pathlib, tempfile
    run_phase(tp, Phase("coexist", 9, 3e-4, True, 1.0, 1.0), sources,
              out_dir=pathlib.Path(tempfile.mkdtemp()), verbose=False, save_every=9)
    after = {f"{c}.{k}": v for c, m in ((PHYS, tp.phi), (CONST, tp.cphi))
             for k, v in tree_flatten(m.parameters())}
    mx.eval(list(after.values()))
    moved = [k for k in before if not bool(mx.all(before[k] == after[k]).item())]
    gates = [k for k in moved if ".inject.g1" in k or ".inject.g2" in k]
    others = [k for k in moved if k not in gates]
    assert not others, ("a gate-only phase moved non-gate parameters -- weight decay? "
                        f"{others[:6]}")
    assert gates, "a gate-only phase moved no gate parameters at all"
    say(f"8. gate-only phase  moved {len(gates)} gate tensors and 0 of "
        f"{len(before)-len(gates)} others, bit-identical")

    # 9. CHUNKED RESUME. Every chunk is a fresh process that re-warm-starts both
    #    channels from their trained checkpoints, so without a resume step three
    #    chunks of 200 are three independent 200-step runs and the final artifact
    #    holds 200 while the log claims 600. That is what chunking is for, so it
    #    gets an assertion: the step count must be cumulative and the weights must
    #    continue rather than reset.
    R = build_tiny_dual(write_idx=[1, 7, 13], seed=7)
    rp = R["dual"]
    rp.set_modes(physics="psilm", constitution="psilm")
    d = pathlib.Path(tempfile.mkdtemp())
    ph = Phase("coexist", 3, 3e-4, True, 1.0, 1.0)
    run_phase(rp, ph, sources, out_dir=d, verbose=False, save_every=3)
    m1 = json.loads((d / "bridges.safetensors.meta").read_text())["step"]
    g1 = mx.array(rp.cphi.inject.g2.bias)
    mx.eval(g1)
    run_phase(rp, ph, sources, out_dir=d, verbose=False, save_every=3)
    m2 = json.loads((d / "bridges.safetensors.meta").read_text())["step"]
    assert (m1, m2) == (3, 6), f"step count not cumulative across chunks: {m1} then {m2}"
    # and a fresh model resuming from that checkpoint must land on those weights,
    # not on the warm-start ones
    F = build_tiny_dual(write_idx=[1, 7, 13], seed=7)["dual"]
    F.set_modes(physics="psilm", constitution="psilm")
    fresh = mx.array(F.cphi.inject.g2.bias); mx.eval(fresh)
    from .train_dual import resume as do_resume
    got = do_resume(F, d)
    after = mx.array(F.cphi.inject.g2.bias); mx.eval(after)
    assert got == 6, f"resume reported step {got}, expected 6"
    assert not bool(mx.all(fresh == after).item()), "resume did not change the weights"
    say(f"9. chunked resume   step count cumulative ({m1} -> {m2}) and weights continue")

    # 10. THE REPLAY TEST. Bug B was: run_phase captured `params` before resume(),
    #     and loss_for writes them back every step, so step 1 overwrote the loaded
    #     weights and the chunk reproduced the previous one exactly -- same seed,
    #     same batches, same starting point. Cumulative step counts and a direct
    #     resume() call both looked fine, which is why this needs its own assertion:
    #     two consecutive chunks must NOT trace the same trajectory.
    Q = build_tiny_dual(write_idx=[1, 7, 13], seed=8)
    qp = Q["dual"]
    qp.set_modes(physics="psilm", constitution="psilm")
    dq = pathlib.Path(tempfile.mkdtemp())
    phq = Phase("coexist", 4, 3e-3, True, 1.0, 1.0)
    _, r1 = run_phase(qp, phq, sources, out_dir=dq, verbose=False, save_every=4)
    _, r2 = run_phase(qp, phq, sources, out_dir=dq, verbose=False, save_every=4)
    g1s = [r["gate_" + CONST] for r in r1["constitution"]]
    g2s = [r["gate_" + CONST] for r in r2["constitution"]]
    assert g1s != g2s, ("two chunks traced an identical constitution-gate trajectory "
                        f"{g1s} -- the resume was overwritten by stale params")
    say(f"10. no replay       chunk 2 diverges from chunk 1 ({g1s[0]:.4f} -> {g2s[0]:.4f})")

    # 11. a frozen channel is refused, not silently trained for zero steps. Bug A was
    #     load_constitution_stack's bridges.freeze(), which made the optimizer a
    #     no-op while every log line still printed a plausible batch-varying gate.
    Z = build_tiny_dual(write_idx=[1, 7, 13], seed=9)["dual"]
    Z.set_modes(physics="psilm", constitution="psilm")
    Z.cphi.freeze()
    try:
        run_phase(Z, Phase("coexist", 1, 3e-4, True, 1.0, 1.0), sources,
                  out_dir=pathlib.Path(tempfile.mkdtemp()), verbose=False)
    except ValueError as e:
        assert "no trainable parameters" in str(e), e
        say("11. frozen channel  refused with a named error, not trained for zero steps")
    else:
        raise AssertionError("a frozen channel was accepted")

    say("\nall schedule assertions passed")
    return True


if __name__ == "__main__":
    self_test()
