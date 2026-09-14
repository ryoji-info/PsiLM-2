#!/usr/bin/env python3
"""Run the dual stack's training schedule.

    PYTHONPATH=../PsiLM python -m psilm2.train_dual --phase coexist --steps 600

Warm-starts both channels from their trained checkpoints, then runs one phase of
psilm2.schedule.DualSchedule: a physics batch, a constitution batch and a no-harm
batch in rotation, with the other channel's gate penalised on each task batch.
See psilm2/schedule.py for why that penalty is the point, and docs/schedule.md for
the phase order.

Chunked like every other PsiLM trainer: --steps per invocation, resuming from its
own bridges.npz, so a Metal watchdog kill costs one chunk rather than the run. The
supervisor loop belongs in a shell script, as it does for the single-channel
campaigns.

--dry-run exercises the whole loop on a tiny random stack, on the CPU, without
touching the 9B or the GPU. That is what the self-tests run; use it after editing
anything here.
"""

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .dual import CONST, PHYS, dual_meta
from .schedule import (Baselines, DualSchedule, PHASES, Phase, gate_only_grads)


def grad_window(psi, phase) -> int:
    """The shallowest layer the backward pass must reach.

    Qwen3.5's GatedDeltaNet layers run a Metal kernel with no VJP, so the
    differentiable ops-path scan has to be switched on for exactly the layers the
    backward pass touches and no more -- it keeps its whole recurrence on the tape,
    which costs 25.2 GB at batch 2 across all 32 layers against 12.9 GB for the top
    twelve.

    In a gate-only phase the only trainable parameters are the gate MLPs, which read
    the stream AT the injection depths, so nothing below the shallowest injection
    needs a gradient: the window is min(l_rev). In a full phase the forward bridges
    are trainable too and the window has to reach min(l_fwd). On this backbone that
    is 24 against 13 -- the concrete form of phase 1 being the cheap one.
    """
    lo = []
    for present, l_fwd, l_rev in ((psi.has_phys, psi.l_fwd, psi.l_rev),
                                  (psi.has_const, psi.l_fwd_const, psi.l_rev_const)):
        if present:
            lo.append(l_rev if phase.gate_only else l_fwd)
    return min(lo) if lo else psi.n_layers


def clip_grad_norm(grads, max_norm):
    leaves = [v for _, v in tree_flatten(grads)]
    total = mx.sqrt(sum((mx.sum(g * g) for g in leaves), mx.array(0.0)))
    scale = mx.minimum(mx.array(1.0), max_norm / (total + 1e-9))
    return tree_map(lambda g: g * scale, grads), total


# -- batch sources ----------------------------------------------------------
def constitution_source(path, pad_id, batch, rng):
    """HH-RLHF prompts with the constitution-conditioned teacher's continuation."""
    from psilm.mlx.constitution import pad_batch
    items = json.loads(Path(path).read_text())
    return lambda: pad_batch(rng.sample(items, batch), pad_id)


def noharm_source(path, pad_id, batch, rng):
    """Off-task prompts (GSM8K/MMLU) with the backbone's OWN continuation.

    The key is "target_ids", not the constitution data's "base_ids": these
    negatives come from eval/build_noharm.py, which predates the constitution work
    and writes {prompt_ids, target_ids, target_text, source}. Both campaigns share
    this one file, so the schedule reads it in its own schema rather than
    normalising it.
    """
    from psilm.mlx.constitution import pad_batch
    items = json.loads(Path(path).read_text())
    missing = [k for k in ("prompt_ids", "target_ids") if k not in items[0]]
    if missing:
        raise ValueError(f"{path}: no-harm items lack {missing}; "
                         f"found {sorted(items[0])}")
    return lambda: pad_batch(rng.sample(items, batch), pad_id, tgt_key="target_ids",
                             noharm=True)


def physics_source(hf_tok, path, batch, rng):
    """The Stage-2 Burgers QA task, via the same builder the physics trainer uses.

    Imported from eval/mlx_stage2_train.py rather than reimplemented: to_mlx_batch
    holds the x0-bin quantisation (x0 * 100, clipped to 0..99) that the trained
    pointer head expects, and a second copy of that would be a silent way to feed
    the warm-started readout targets it was never trained on.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "PsiLM" / "eval"))
    from mlx_stage2_train import to_mlx_batch                      # noqa: E402
    from psilm.stage2.qa import QABuilder, make_batch as torch_make_batch
    items = json.loads(Path(path).read_text())
    builder = QABuilder(hf_tok)
    return lambda: to_mlx_batch(torch_make_batch(builder, rng.sample(items, batch), "cpu"))


# -- the loop ---------------------------------------------------------------
def run_phase(psi, phase: Phase, sources, *, out_dir: Path, lr=None, clip="module",
              log_every=25, save_every=100, baselines: Baselines = None, verbose=True):
    """One phase. Returns the step count reached and the last stats."""
    sch = DualSchedule(phase, sources)
    rate = lr if lr is not None else phase.lr
    # One optimizer PER CHANNEL: mlx keeps a single state tree per optimizer, and
    # the two bridges have different shapes, so a shared optimizer raises
    # KeyError on the second channel's update.
    #
    # And no weight decay in a gate-only phase. Gate-only is implemented by
    # zeroing the non-gate gradients, and AdamW would still decay those
    # parameters every step -- 600 steps of decay on readouts that took 11,500
    # physics steps to train is not "frozen". Adam with a zero gradient leaves a
    # parameter bit-identical, which is what the phase promises.
    make_opt = (lambda: optim.Adam(learning_rate=rate)) if phase.gate_only else \
               (lambda: optim.AdamW(learning_rate=rate))
    opts = {PHYS: make_opt(), CONST: make_opt()}
    params = {PHYS: psi.phi.trainable_parameters() if psi.has_phys else {},
              CONST: psi.cphi.trainable_parameters() if psi.has_const else {}}

    def loss_for(p, task, batch):
        if psi.has_phys:
            psi.phi.update(p[PHYS])
        if psi.has_const:
            psi.cphi.update(p[CONST])
        return sch.loss(psi, task, batch)

    model = psi.model
    if getattr(model, "needs_train_mode_for_grad", False):
        w = grad_window(psi, phase)
        model.set_grad_window(w)
        if verbose:
            print(f"[backbone] differentiable SSM scan from layer {w} up "
                  f"({'gate-only: injections' if phase.gate_only else 'full: reads'})")

    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "supervisor.log").open("a")
    run = {}
    t0 = time.time()
    for step in range(1, phase.steps + 1):
        task, batch = sch.next()
        (loss, stats), grads = mx.value_and_grad(
            lambda p: loss_for(p, task, batch))(params)
        if phase.gate_only:
            grads = gate_only_grads(grads)
        if clip == "module":
            grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
        else:
            grads, _ = clip_grad_norm(grads, 1.0)
        if psi.has_phys:
            opts[PHYS].update(psi.phi, grads[PHYS])
        if psi.has_const:
            opts[CONST].update(psi.cphi, grads[CONST])
        mx.eval(loss, psi.phi.parameters() if psi.has_phys else [],
                psi.cphi.parameters() if psi.has_const else [],
                opts[PHYS].state, opts[CONST].state)
        params = {PHYS: psi.phi.trainable_parameters() if psi.has_phys else {},
                  CONST: psi.cphi.trainable_parameters() if psi.has_const else {}}

        rec = {"step": step, "task": task, "loss": round(float(loss.item()), 4)}
        for k, v in stats.items():
            if isinstance(v, mx.array):
                rec[k] = round(float(v.item()), 5)
        run.setdefault(task, []).append(rec)
        if verbose and step % log_every == 0:
            line = (f"[{phase.name}] step {step}/{phase.steps} "
                    + " | ".join(f"{t}: loss {run[t][-1]['loss']:.4f} "
                                 f"gp {run[t][-1].get('gate_'+PHYS, float('nan')):.4f} "
                                 f"gc {run[t][-1].get('gate_'+CONST, float('nan')):.4f}"
                                 for t in run)
                    + f" | {(time.time()-t0)/step:.2f}s/step")
            print(line); log.write(line + "\n"); log.flush()
        if step % save_every == 0 or step == phase.steps:
            save(psi, out_dir, step, phase)
    log.close()
    return step, run


def save(psi, out_dir: Path, step: int, phase: Phase):
    flat = {}
    if psi.has_phys:
        flat.update({f"physics.{k}": v for k, v in tree_flatten(psi.phi.parameters())})
    if psi.has_const:
        flat.update({f"constitution.{k}": v for k, v in tree_flatten(psi.cphi.parameters())})
    mx.save_safetensors(str(out_dir / "bridges.safetensors"), flat)
    meta = dual_meta(psi, step, extra={"phase": phase.name, "lam_cross": phase.lam_cross,
                                       "lam_gate": phase.lam_gate,
                                       "gate_only": phase.gate_only, "lr": phase.lr})
    (out_dir / "bridges.safetensors.meta").write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="coexist", choices=[p.name for p in PHASES])
    ap.add_argument("--steps", type=int, default=None, help="override the phase's default")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lam-cross", type=float, default=None)
    ap.add_argument("--out", default="results/dual_qwen35")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny random stack on the CPU: exercises the loop, touches no GPU")
    ap.add_argument("--model", default="/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx")
    ap.add_argument("--phys-ckpt", default="results/stage2_qwen35/bridges.npz")
    ap.add_argument("--fno", default="results/stage2/fno.pt")
    ap.add_argument("--const-ckpt", default="results/stage2c_qwen35_all/bridges.npz")
    ap.add_argument("--const-model",
                    default="results/constitution_model/qwen2.5-0.5b-constitution")
    ap.add_argument("--const-data", default="data/constitution_train_qwen35.json")
    ap.add_argument("--noharm-data", default="data/noharm_qwen35_all.json")
    ap.add_argument("--phys-data", default="data/stage2_qa_train.json")
    a = ap.parse_args()

    phase = next(p for p in PHASES if p.name == a.phase)
    if a.steps is not None:
        phase = Phase(phase.name, a.steps, phase.lr, phase.gate_only,
                      phase.lam_cross, phase.lam_gate, phase.cycle)
    if a.lam_cross is not None:
        phase = Phase(phase.name, phase.steps, phase.lr, phase.gate_only,
                      a.lam_cross, phase.lam_gate, phase.cycle)
    rng = random.Random(a.seed)

    if a.dry_run:
        from psilm.mlx.constitution import pad_batch, synthetic_items
        from .dual_self_test import build_tiny_dual
        from .schedule_self_test import _phys_batch
        psi = build_tiny_dual(write_idx=[1, 7, 13], seed=4)["dual"]
        psi.set_modes(physics="psilm", constitution="psilm")
        items = synthetic_items(8, n_prompt=7, n_cont=4)
        sources = {"physics": lambda: _phys_batch(),
                   "constitution": lambda: pad_batch(rng.sample(items, 2), 0),
                   "noharm": lambda: pad_batch(rng.sample(items, 2), 0,
                                               tgt_key="base_ids", noharm=True)}
        out = Path("/tmp/psilm2_dryrun")
    else:
        from transformers import AutoTokenizer
        from psilm.mlx.gemma_loader import load_backbone_any
        from .dual import load_dual_stack
        model, _stock, tok = load_backbone_any(a.model)
        psi = load_dual_stack(model, tok, phys_ckpt=a.phys_ckpt, fno_path=a.fno,
                              const_ckpt=a.const_ckpt, const_model_path=a.const_model,
                              lam_gate=phase.lam_gate)
        psi.set_modes(physics="psilm", constitution="psilm")
        hf = AutoTokenizer.from_pretrained(a.model)
        pad = hf.pad_token_id or hf.eos_token_id
        sources = {"physics": physics_source(hf, a.phys_data, a.batch, rng),
                   "constitution": constitution_source(a.const_data, pad, a.batch, rng),
                   "noharm": noharm_source(a.noharm_data, pad, a.batch, rng)}
        out = Path(a.out)
        print(psi.describe())

    print(f"phase {phase.name}: {phase.steps} steps, lr {phase.lr}, "
          f"gate_only={phase.gate_only}, lam_cross={phase.lam_cross}, "
          f"lam_gate={phase.lam_gate}, cycle {phase.cycle}")
    step, run = run_phase(psi, phase, sources, out_dir=out, lr=a.lr,
                          log_every=a.log_every, save_every=a.save_every)
    print(f"\nphase {phase.name} reached step {step}; wrote {out}/bridges.safetensors")
    for task, rows in run.items():
        g = rows[-1]
        print("  %-13s last loss %.4f  gate phys %s  gate const %s"
              % (task, g["loss"], g.get("gate_" + PHYS), g.get("gate_" + CONST)))


if __name__ == "__main__":
    main()
