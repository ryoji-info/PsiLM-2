#!/usr/bin/env python3
"""Phase acceptance: did the composition cost either channel its own performance?

    PYTHONPATH=../PsiLM python -m psilm2.accept --ckpt results/dual_qwen35

The gate traces a phase prints are diagnostics, not results. A schedule can drive
both gates to beautiful task-selectivity and still have quietly broken one of the
channels it was composing, and the trace would look the same. This script measures
the two things that decide it, each against the single-channel number the same
harness produced:

  constitution   teacher-forced CE and top-1 agreement on the held-out red-team
                 split, against the constitution-only run's 0.3890 / 0.8678

  physics        held-out accuracy on the Burgers QA validation set, against the
                 physics-only run's own figure

and it measures each with the OTHER channel both open and off, which is the
comparison that attributes any movement to the composition rather than to the
phase's training. Four arms per channel, one load of the backbone.

Nothing here trains, so it runs at inference memory (~10 GB) and needs the GPU
free.
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .dual import CONST, PHYS, load_dual_stack
from .schedule import Baselines


def teacher_forced(psi, prompt_ids, cont_ids):
    """(CE, top-1 agreement) on the teacher's continuation, in one pass.

    Mirrors PsiConstitutionMLX.teacher_forced: the prompt mask covers the prompt
    only, which is what the forward bridges pooled over in training, and the
    logits are sliced so position i predicts cont_ids[i].
    """
    ids = list(prompt_ids) + list(cont_ids)
    t = mx.array([ids], dtype=mx.int32)
    pmask = mx.array([[True] * len(prompt_ids) + [False] * len(cont_ids)])
    lg, _ = psi.logits(t, None, pmask)
    lo, hi = len(prompt_ids) - 1, len(ids) - 1
    lg = lg[:, lo:hi].astype(mx.float32)
    tgt = mx.array([list(cont_ids)], dtype=mx.int32)
    ce = nn.losses.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                 reduction="mean")
    agree = (lg.argmax(-1) == tgt).astype(mx.float32).mean()
    mx.eval(ce, agree)
    return float(ce.item()), float(agree.item())


def eval_constitution(psi, items, n, arms):
    out = {}
    for label, (pm, cm) in arms.items():
        psi.set_modes(physics=pm, constitution=cm)
        ces, ags = [], []
        for it in items[:n]:
            if not it.get("teacher_ids"):
                continue
            ce, ag = teacher_forced(psi, it["prompt_ids"], it["teacher_ids"])
            ces.append(ce); ags.append(ag)
        out[label] = {"n": len(ces), "ce": sum(ces) / len(ces),
                      "agree": sum(ags) / len(ags)}
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return out


#: absolute, matching eval/mlx_stage2_eval.py's score(): abs(pred - true) <= TOL.
#: A relative tolerance would not be the number the baseline was measured with.
PHYS_TOL = 0.05


def parse_answer(text):
    """The PsiLM arm's parser, verbatim from eval/mlx_stage2_eval.py.

    The trained reply is "u at x = {x0} equals {u}." and only the number after
    "equals" counts -- a reply that stops at the x0 echo is a miss. This is NOT the
    text arms' "Answer:" parser: pointing that one at a coupled reply finds nothing
    at all and scores a flat zero, which is a false catastrophic regression rather
    than a measurement.
    """
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def eval_physics(psi, builder, items, n, arms, tol=PHYS_TOL, max_new=24):
    """Held-out accuracy on the same protocol the physics campaign measured:
    psi.generate(builder, item) at its default 24 tokens, parse_answer, and an
    ABSOLUTE tolerance of 0.05 on u."""
    out = {}
    for label, (pm, cm) in arms.items():
        psi.set_modes(physics=pm, constitution=cm)
        ok = tot = 0
        errs = []
        for it in items[:n]:
            txt = psi.generate(builder, it, max_new=max_new)
            v = parse_answer(txt)
            tot += 1
            if v is not None:
                errs.append(abs(v - it["u"]))
                ok += int(abs(v - it["u"]) <= tol)
        out[label] = {"n": tot, "acc": ok / max(1, tot),
                      "mae": (sum(errs) / len(errs)) if errs else None,
                      "parsed": len(errs) / max(1, tot)}
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/dual_qwen35",
                    help="a phase output directory holding bridges.safetensors")
    ap.add_argument("--model", default="/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx")
    ap.add_argument("--phys-ckpt", default="results/stage2_qwen35/bridges.npz")
    ap.add_argument("--fno", default="results/stage2/fno.pt")
    ap.add_argument("--const-ckpt", default="results/stage2c_qwen35_all/bridges.npz")
    ap.add_argument("--const-model",
                    default="results/constitution_model/qwen2.5-0.5b-constitution")
    ap.add_argument("--const-test", default="data/constitution_test_qwen35.json")
    ap.add_argument("--phys-val", default="data/stage2_qa_val.json")
    ap.add_argument("--n-const", type=int, default=50)
    ap.add_argument("--n-phys", type=int, default=50)
    # the single-channel numbers this composition must not regress, from the runs
    # that produced them (results/constitution/summary_qwen35.json, the physics
    # campaign's held-out record)
    ap.add_argument("--base-const-ce", type=float, default=0.3890)
    ap.add_argument("--base-phys-acc", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from psilm.mlx.gemma_loader import load_backbone_any
    from psilm.stage2.qa import QABuilder

    model, _stock, tok = load_backbone_any(a.model)
    psi = load_dual_stack(model, tok, phys_ckpt=a.phys_ckpt, fno_path=a.fno,
                          const_ckpt=a.const_ckpt, const_model_path=a.const_model)
    ck = Path(a.ckpt)
    from .train_dual import resume
    step = resume(psi, ck)
    if not step:
        raise SystemExit(f"{ck}/bridges.safetensors holds no step: nothing to accept")
    meta = json.loads((ck / "bridges.safetensors.meta").read_text())
    print(psi.describe())
    print(f"phase {meta.get('phase')} step {step}, lam_cross={meta.get('lam_cross')}, "
          f"gate_only={meta.get('gate_only')}")

    # a backbone whose SSM scan was left in a training window would score on the
    # wrong numerics; put it back to the deployment path for every measurement
    if getattr(model, "needs_train_mode_for_grad", False):
        model.set_grad_window(psi.n_layers)

    citems = json.loads(Path(a.const_test).read_text())
    pitems = json.loads(Path(a.phys_val).read_text())
    hf = AutoTokenizer.from_pretrained(a.model)
    builder = QABuilder(hf)

    carms = {"both open": ("psilm", "psilm"), "constitution only": ("off", "psilm"),
             "base": ("off", "off")}
    parms = {"both open": ("psilm", "psilm"), "physics only": ("psilm", "off"),
             "base": ("off", "off")}
    print(f"\nconstitution, held-out red-team n={a.n_const} (teacher-forced):")
    cres = eval_constitution(psi, citems, a.n_const, carms)
    for k, v in cres.items():
        print("  %-20s CE %.4f   agree %.4f" % (k, v["ce"], v["agree"]))
    print(f"\nphysics, held-out QA n={a.n_phys} (greedy, {'24'} tokens):")
    pres = eval_physics(psi, builder, pitems, a.n_phys, parms)
    for k, v in pres.items():
        print("  %-20s acc %.3f   parsed %.2f   MAE %s"
              % (k, v["acc"], v["parsed"],
                 ("%.4f" % v["mae"]) if v["mae"] is not None else "-"))

    b = Baselines(const_ce=a.base_const_ce, phys_acc=a.base_phys_acc)
    ok, why = b.accept(const_ce=cres["both open"]["ce"], phys_acc=pres["both open"]["acc"])
    print("\nacceptance against the single-channel baselines "
          f"(CE {a.base_const_ce} +{b.const_tol}, acc {a.base_phys_acc} -{b.phys_tol}):")
    print("  " + ("ACCEPTED" if ok else "REGRESSED: " + "; ".join(why)))
    print("\nwhat opening the other channel costs:")
    print("  constitution CE  %.4f alone -> %.4f with physics open  (%+.4f)"
          % (cres["constitution only"]["ce"], cres["both open"]["ce"],
             cres["both open"]["ce"] - cres["constitution only"]["ce"]))
    print("  physics acc      %.3f alone -> %.3f with the constitution open (%+.3f)"
          % (pres["physics only"]["acc"], pres["both open"]["acc"],
             pres["both open"]["acc"] - pres["physics only"]["acc"]))

    rec = {"ckpt": str(ck), "step": step, "meta": meta, "constitution": cres,
           "physics": pres, "accepted": ok, "why": why}
    dest = Path(a.out) if a.out else ck / "accept.json"
    dest.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
