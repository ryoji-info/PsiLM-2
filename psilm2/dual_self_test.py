"""Self-test for the dual-bridge composition, on a tiny random stack, CPU only.

Run:  python -m psilm2.dual_self_test

The point of these assertions is attribution. A dual run differs from a
single-channel run for two possible reasons -- the other channel is present, or
the dual code path is not the same code path -- and only the first is
interesting. Assertions 1-3 rule out the second by demanding bit-identity with
the two models PsiLM already trains and with the bare backbone. Assertions 4-7
then characterise what the composition actually does.

Everything runs on a random 4-layer Qwen2 at d_model 64 on the CPU, so this never
queues work on the shared GPU and needs no weights.
"""

import mlx.core as mx

from psilm.mlx.bridges import PsiBridgesMLX
from psilm.mlx.constitution import (ConstitutionBridgesMLX, ConstitutionModelMLX,
                                    FakeTok, PsiConstitutionMLX)
from psilm.mlx.fno import FNO1dMLX
from psilm.mlx.model import PsiLMMLX
from psilm.mlx.staged import MlxStream

from .dual import CONST, PHYS, PsiDualMLX


def build_tiny_dual(d_model=64, n_layers=6, vocab=256, heads=4,
                    l_fwd_phys=2, l_rev_phys=5, l_fwd_const=2, l_rev_const=4,
                    write_idx=None, seed=0):
    """A random backbone, a random constitution partner, a tiny FNO, both bridges."""
    mx.set_default_device(mx.cpu)
    mx.random.seed(seed)
    from mlx_lm.models.qwen2 import Model, ModelArgs

    def args():
        return ModelArgs(model_type="qwen2", hidden_size=d_model,
                         num_hidden_layers=n_layers, intermediate_size=2 * d_model,
                         num_attention_heads=heads, rms_norm_eps=1e-6,
                         vocab_size=vocab, num_key_value_heads=heads,
                         max_position_embeddings=512, rope_theta=10000.0,
                         tie_word_embeddings=True)

    model = Model(args()); model.freeze()
    tok = FakeTok()
    fno = FNO1dMLX(width=32, modes=4, layers=1)          # width 32: ReverseBridgeMLX's default
    phys = PsiBridgesMLX(d_model, gate_bias=-0.5, inj_cap=0.2)
    const = ConstitutionModelMLX.from_loaded(Model(args()), tok, m_tokens=4, path="<tiny>")
    cphi = ConstitutionBridgesMLX(d_model, const.d_const, write_idx=write_idx,
                                  k_fwd=3, m_tokens=const.n_readout, gate_bias=0.0,
                                  inj_cap=0.2, readout_norm="dim",
                                  emb_rms=const.emb_rms, d_hidden=32)
    # At init to_out is 1e-3 and g2 exactly zero -- the right training start, but it
    # makes every "does the channel do anything" assertion vacuous. Give both
    # channels a material injection, as the constitution self-test does.
    for b in (phys, cphi):
        b.inject.to_out.weight = 0.3 * mx.random.normal(b.inject.to_out.weight.shape)
        b.inject.g2.weight = 0.02 * mx.random.normal(b.inject.g2.weight.shape)
    mx.eval(model.parameters(), const.model.parameters(), fno.parameters(),
            phys.parameters(), cphi.parameters())
    dual = PsiDualMLX(model, tok, fno=fno, phys_bridges=phys,
                      l_fwd_phys=l_fwd_phys, l_rev_phys=l_rev_phys,
                      const_model=const, const_bridges=cphi,
                      l_fwd_const=l_fwd_const, l_rev_const=l_rev_const)
    return dict(model=model, tok=tok, fno=fno, phys=phys, const=const, cphi=cphi,
                dual=dual, l=(l_fwd_phys, l_rev_phys, l_fwd_const, l_rev_const))


def _ids(B=2, L=9, vocab=256, seed=5):
    mx.random.seed(seed)
    ids = mx.random.randint(3, vocab, (B, L))
    pmask = mx.concatenate([mx.ones((B, L - 3), dtype=mx.bool_),
                            mx.zeros((B, 3), dtype=mx.bool_)], axis=1)
    return ids, pmask


def _dual_logits(dual, ids, pmask, physics, constitution):
    dual.set_modes(physics=physics, constitution=constitution)
    lg, st = dual.logits(ids, None, pmask)
    mx.eval(lg)
    return lg, st


def self_test(verbose=True):
    S = build_tiny_dual()
    dual, ids, = S["dual"], None
    ids, pmask = _ids()
    lp, lr, lc, lcr = S["l"]

    def say(*a):
        if verbose:
            print(*a)

    # 1. both channels off == the bare staged backbone, bit for bit
    s = MlxStream(S["model"], ids); s.run(0, dual.n_layers)
    bare = s.finish(); mx.eval(bare)
    off, _ = _dual_logits(dual, ids, pmask, "off", "off")
    assert bool(mx.all(bare == off).item()), "both-off != bare backbone"
    say("1. both off        == bare staged backbone            bit-identical")

    # 2. physics only == PsiLMMLX over the same objects
    ref_p = PsiLMMLX(S["model"], S["tok"], S["fno"], S["phys"], l_fwd=lp, l_rev=lr)
    sp = MlxStream(S["model"], ids); ref_p._couple(sp, pmask, None)
    ref_pl = sp.finish(); mx.eval(ref_pl)
    only_p, st_p = _dual_logits(dual, ids, pmask, "psilm", "off")
    assert bool(mx.all(ref_pl == only_p).item()), "physics-only != PsiLMMLX"
    say("2. physics only    == PsiLMMLX                         bit-identical")

    # 3. constitution only == PsiConstitutionMLX over the same objects
    ref_c = PsiConstitutionMLX(S["model"], S["tok"], S["const"], S["cphi"],
                               l_fwd=lc, l_rev=lcr)
    sc = MlxStream(S["model"], ids); ref_c._couple(sc, pmask, "psilm")
    ref_cl = sc.finish(); mx.eval(ref_cl)
    only_c, st_c = _dual_logits(dual, ids, pmask, "off", "psilm")
    assert bool(mx.all(ref_cl == only_c).item()), "constitution-only != PsiConstitutionMLX"
    say("3. constitution only == PsiConstitutionMLX             bit-identical")

    # 4. both open is a different function from either alone
    both, st_b = _dual_logits(dual, ids, pmask, "psilm", "psilm")
    assert not bool(mx.all(both == only_p).item()), "both == physics only"
    assert not bool(mx.all(both == only_c).item()), "both == constitution only"
    assert not bool(mx.all(both == off).item()), "both == neither"
    say("4. both open       != either alone, and != neither")

    # 5. the zeroed arm measures a gate on a stream it never wrote to
    for phys_m, const_m, ref, label in (("zeroed", "off", off, "physics"),
                                        ("off", "zeroed", off, "constitution"),
                                        ("zeroed", "zeroed", off, "both")):
        z, stz = _dual_logits(dual, ids, pmask, phys_m, const_m)
        assert bool(mx.all(z == ref).item()), f"zeroed {label} changed the stream"
        for ch in (PHYS, CONST):
            if getattr(stz, ch).mode == "zeroed":
                assert getattr(stz, ch).sigma is not None, f"zeroed {ch}: no gate measured"
    say("5. zeroed arms     leave the stream exactly alone, gates still measured")

    # 6. mixing arms: physics writing while the constitution is only measured must
    #    equal physics alone -- otherwise 'zeroed' is leaking into the stream
    mixed, _ = _dual_logits(dual, ids, pmask, "psilm", "zeroed")
    assert bool(mx.all(mixed == only_p).item()), "constitution-zeroed leaked into physics"
    mixed2, _ = _dual_logits(dual, ids, pmask, "zeroed", "psilm")
    assert bool(mx.all(mixed2 == only_c).item()), "physics-zeroed leaked into constitution"
    say("6. one writing, one measured == the writer alone       bit-identical")

    # 7. THE INTERACTION. The constitution writes at 4, the physics gate reads the
    #    stream at 5, so opening the constitution changes what the physics gate
    #    sees. This is the composition's one real coupling and it must be visible.
    gp_alone = st_p.gate(PHYS); gp_both = st_b.gate(PHYS)
    mx.eval(gp_alone, gp_both)
    d = float(mx.abs(gp_both - gp_alone).max().item())
    assert d > 0.0, ("the physics gate is identical with and without the "
                     "constitution write below it: the channels are not sharing a stream")
    say(f"7. physics gate moves by up to {d:.4f} when the constitution opens below it")

    # 8. the ordering restriction is refused, not silently resolved
    bad = PsiDualMLX(S["model"], S["tok"], fno=S["fno"], phys_bridges=S["phys"],
                     l_fwd_phys=5, l_rev_phys=5,        # reads at 5
                     const_model=S["const"], const_bridges=S["cphi"],
                     l_fwd_const=2, l_rev_const=4)      # writes at 4, below that read
    try:
        bad.logits(ids, None, pmask)
    except NotImplementedError as e:
        assert "read depth must be at or below" in str(e)
        say("8. interleaved depths raise NotImplementedError, not a silent order")
    else:
        raise AssertionError("an interleaved configuration was accepted")

    # 9. a narrow constitution write touches only its own dimensions, with the
    #    physics channel writing the whole stream in the same pass
    W = [1, 7, 13]
    N = build_tiny_dual(write_idx=W, seed=1)
    nd, (npf, npr, ncf, ncr) = N["dual"], N["l"]
    s1 = MlxStream(N["model"], ids)
    nd.set_modes(physics="psilm", constitution="off"); nd._couple(s1, pmask, None)
    h_p = s1.hidden
    s2 = MlxStream(N["model"], ids)
    nd.set_modes(physics="psilm", constitution="psilm"); nd._couple(s2, pmask, None)
    h_b = s2.hidden
    mx.eval(h_p, h_b)
    # the constitution writes at 4 and the physics channel at 5, so by the final
    # layer the difference has spread; compare instead at the constitution's own
    # write depth, where its mask is still exact
    a = MlxStream(N["model"], ids)
    nd.set_modes(physics="off", constitution="off"); a.run(0, ncr)
    base_h = a.hidden
    b = MlxStream(N["model"], ids)
    nd.set_modes(physics="off", constitution="psilm")
    b.run(0, ncf)
    tk = nd._read(CONST, b.hidden, pmask, None)
    b.run(ncf, ncr)
    wrote, _, _ = nd._write(CONST, b.hidden, tk)
    mx.eval(base_h, wrote)
    delta = mx.abs(wrote - base_h).max(axis=(0, 1))
    mx.eval(delta)
    touched = [i for i, v in enumerate(delta.tolist()) if v > 0]
    assert touched == W, f"masked write touched {touched}, expected {W}"
    say(f"9. a {len(W)}-dimension constitution write touches exactly those dims")

    say("\n" + nd.describe())
    say("all dual-bridge assertions passed")
    return True




def gradient_test(verbose=True):
    """Both channels must receive gradients from the joint objective.

    A dual model where only one bridge is reachable by the optimizer would train
    perfectly happily and silently be a single-channel model, so this is checked
    separately from the forward-identity assertions above: every leaf of both
    bridges' trainable parameters gets a finite, and for the reachable ones a
    non-zero, gradient from one loss_constitution call.
    """
    from psilm.mlx.constitution import pad_batch, synthetic_items
    import mlx.nn as nn

    S = build_tiny_dual(write_idx=[1, 7, 13], seed=2)
    dual = S["dual"]
    dual.set_modes(physics="psilm", constitution="psilm")
    batch = pad_batch(synthetic_items(3, n_prompt=7, n_cont=4), pad_id=0)

    def flat(t, prefix=""):
        if isinstance(t, mx.array):
            yield prefix, t
        elif isinstance(t, dict):
            for k, v in t.items():
                yield from flat(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(t, (list, tuple)):
            for i, v in enumerate(t):
                yield from flat(v, f"{prefix}[{i}]")

    def loss_only(params):
        dual.phi.update(params[PHYS])
        dual.cphi.update(params[CONST])
        return dual.loss_constitution(batch)[0]

    p0 = {PHYS: dual.phi.trainable_parameters(), CONST: dual.cphi.trainable_parameters()}
    loss, grads = mx.value_and_grad(loss_only)(p0)
    mx.eval(loss, grads)
    assert mx.isfinite(loss).item(), f"loss is not finite: {loss}"

    report = {}
    for ch in (PHYS, CONST):
        leaves = dict(flat(grads[ch]))
        assert leaves, f"{ch}: no gradient leaves at all"
        nonzero, nonfinite = 0, []
        for name, g in leaves.items():
            if not bool(mx.all(mx.isfinite(g)).item()):
                nonfinite.append(name)
            if float(mx.abs(g).max().item()) > 0:
                nonzero += 1
        assert not nonfinite, f"{ch}: non-finite gradients at {nonfinite[:4]}"
        assert nonzero > 0, f"{ch}: every gradient is exactly zero -- channel unreachable"
        # Named, not merely counted: rev.u_head produces the scalar u_hat, which
        # feeds only the physics deep-supervision term loss_u and never the token
        # path (in the "value" channel form it is behind a stop_gradient as well),
        # so a pure cross-entropy objective reaches everything else and not it.
        # Asserting the exact set means a future change that silently detaches a
        # DIFFERENT tensor fails here instead of training a dead parameter.
        zeros = sorted(n for n, g in leaves.items()
                       if float(mx.abs(g).max().item()) == 0.0)
        expected = {PHYS: ["rev.u_head.bias", "rev.u_head.weight"], CONST: []}[ch]
        assert zeros == expected, f"{ch}: zero-gradient tensors {zeros}, expected {expected}"
        report[ch] = (nonzero, len(leaves))
    if verbose:
        print("gradients from one joint loss_constitution call (loss %.4f):" % float(loss.item()))
        for ch, (nz, n) in report.items():
            print("  %-12s %d of %d parameter tensors have a non-zero gradient" % (ch, nz, n))

    # the physics channel is reached only through the shared stream: the
    # constitution objective has no physics targets, so if this number were zero
    # the two channels would be training independently rather than jointly
    assert report[PHYS][0] > 0, "the physics bridge is not reachable from the constitution loss"
    if verbose:
        print("  -> the physics bridge IS reachable from the constitution objective,")
        print("     i.e. the two channels are coupled through the residual stream")
        print("gradient test passed")
    return True


if __name__ == "__main__":
    self_test()
    print()
    gradient_test()
