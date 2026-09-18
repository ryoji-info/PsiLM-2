"""The dual-bridge identities, on the real Qwen3.5 9B stack rather than a tiny one.

    PYTHONPATH=../PsiLM python -m psilm2.verify_qwen35 --model <backbone dir> ...   (or PSILM_BACKBONE=<backbone dir>)

psilm2.dual_self_test proves nine properties on a random 6-layer stack, which is
where a logic error shows up. This script re-checks the three that matter for
attribution on the actual backbone and the actual trained checkpoints, where a
loader mistake, a depth read from the wrong meta or a dtype difference would show
up instead. It loads both channels, runs one prompt through four arm settings and
compares against the two single-channel models PsiLM trains.

Inference only, about 10 GB peak and a few seconds per arm; it needs the GPU free.
"""

import argparse
import os

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PSILM_BACKBONE"),
                    help="the NVFP4 Qwen3.5 9B backbone: a local directory or Hub id (default: $PSILM_BACKBONE)")
    ap.add_argument("--phys-ckpt", default="results/stage2_qwen35/bridges.npz")
    ap.add_argument("--fno", default="results/stage2/fno.pt")
    ap.add_argument("--const-ckpt", default="results/stage2c_qwen35_all/bridges.npz")
    ap.add_argument("--const-model",
                    default="results/constitution_model/qwen2.5-0.5b-constitution")
    ap.add_argument("--prompt", default="How do I pick a lock that isn't mine?")
    a = ap.parse_args()
    if a.model is None:
        ap.error("--model or $PSILM_BACKBONE is required: the NVFP4 backbone directory, e.g. an `hf download ryoji-info/Qwen3.5-9B-PsiLM --local-dir ...` copy")

    from transformers import AutoTokenizer
    from psilm.mlx.constitution import PsiConstitutionMLX
    from psilm.mlx.gemma_loader import load_backbone_any
    from psilm.mlx.model import PsiLMMLX
    from psilm.mlx.staged import MlxStream
    from .dual import CONST, PHYS, load_dual_stack

    model, _stock, tok = load_backbone_any(a.model)
    psi = load_dual_stack(model, tok, phys_ckpt=a.phys_ckpt, fno_path=a.fno,
                          const_ckpt=a.const_ckpt, const_model_path=a.const_model)
    print(psi.describe())
    print("depths: physics %d/%d | constitution %d/%d of %d layers"
          % (psi.l_fwd, psi.l_rev, psi.l_fwd_const, psi.l_rev_const, psi.n_layers))

    hf = AutoTokenizer.from_pretrained(a.model)
    ids = mx.array([hf.encode(a.prompt)])
    pm = mx.ones(ids.shape, dtype=mx.bool_)

    def dual(p, c):
        psi.set_modes(physics=p, constitution=c)
        lg, st = psi.logits(ids, None, pm)
        mx.eval(lg)
        return lg, st

    d_off, _ = dual("off", "off")
    d_p, st_p = dual("psilm", "off")
    d_c, _ = dual("off", "psilm")
    _, st_b = dual("psilm", "psilm")

    s = MlxStream(model, ids); s.run(0, psi.n_layers); bare = s.finish()
    ref_p = PsiLMMLX(model, tok, psi.fno, psi.phi, l_fwd=psi.l_fwd, l_rev=psi.l_rev)
    s = MlxStream(model, ids); ref_p._couple(s, pm, None); r_p = s.finish()
    ref_c = PsiConstitutionMLX(model, tok, psi.const, psi.cphi,
                               l_fwd=psi.l_fwd_const, l_rev=psi.l_rev_const)
    s = MlxStream(model, ids); ref_c._couple(s, pm, "psilm"); r_c = s.finish()
    mx.eval(bare, r_p, r_c)

    ok = True
    print("\nbit-identity against what PsiLM already trains:")
    for label, x, y in (("both off      vs bare staged backbone", d_off, bare),
                        ("physics only  vs PsiLMMLX", d_p, r_p),
                        ("constitution  vs PsiConstitutionMLX", d_c, r_c)):
        same = bool(mx.all(x == y).item())
        ok &= same
        print("  %-38s %s" % (label, "IDENTICAL" if same else
              "DIFFERS, max %.3e" % float(mx.abs(x - y).max().item())))

    g1, g2 = st_p.gate(PHYS), st_b.gate(PHYS)
    mx.eval(g1, g2)
    d = float(mx.abs(g2 - g1).max().item())
    print("\nthe interaction: the physics gate reads the stream at %d, the constitution"
          % psi.l_rev)
    print("writes at %d, so opening the constitution must move it." % psi.l_rev_const)
    print("  physics gate alone %.6f -> with the constitution open %.6f (max |delta| %.3e)"
          % (float(g1.mean().item()), float(g2.mean().item()), d))
    print("  constitution gate on this prompt: %.4f" % float(st_b.gate(CONST).mean().item()))
    assert d > 0.0, "the physics gate did not move: the channels are not sharing a stream"
    assert ok, "a bit-identity check failed on the real stack"
    print("\npeak %.1f GB | all real-stack checks passed" % (mx.get_peak_memory() / 1e9))


if __name__ == "__main__":
    main()
