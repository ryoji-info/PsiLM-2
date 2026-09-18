<p align="center"><img src="assets/psilm2-banner.png" width="720"></p>

<p align="center"><b>ΨLM-2 — one frozen language model, two frozen partners: a physics model and a document of values, each reached through small trainable bridges.</b></p>

## What ΨLM-2 is

ΨLM-2 keeps the [ΨLM](https://github.com/ryoji-info/PsiLM) premise — nothing
pretrained is fine-tuned during bridge training; the backbone and both partners
stay frozen, the constitution partner having been produced by a full fine-tune
beforehand — and puts two partners on the same frozen backbone at once:

- a **physics bridge**, which reads a PDE's inputs out of the backbone's hidden
  states, lets a neural operator solve it, and returns the value as soft tokens;
- a **constitution bridge**, which reads the prompt out of the same stream, lets
  a small frozen partner model that holds [Claude's constitution](https://www.anthropic.com/constitution)
  deliberate over it, and returns *that* as soft tokens.

The backbone is **Qwen3.5 9B** (frozen, NVFP4, 32 layers, d = 4096). Only the
bridges train: 73.52M parameters (physics 45.18M, constitution 28.34M) against
nine billion frozen ones. No text crosses either interface.

The question the paper asks is whether a channel like the physics one — narrow
enough to audit, gated so it can be closed — can carry a *disposition* rather
than a *quantity*, and where in the backbone's hidden state it should be written.
The paper is [`paper/psilm2.pdf`](paper/psilm2.pdf) (source beside it); the
companion physics paper is [`paper/psilm.pdf`](paper/psilm.pdf).

## What was found

1. **The value-neuron probe reproduces at both scales, but its causal
   justification does not.** Following [arXiv:2602.00986](https://arxiv.org/abs/2602.00986),
   a probe ranks residual-stream coordinates by how well they predict whether the
   model's own continuation will be correct. At 0.5B the damage from zeroing them
   tracks activation magnitude and magnitude-matched controls reproduce it in
   full; at 9B the ablation is a flat null.
2. **The probe's signal occupies a roughly fixed number of coordinates, about
   100–200, at d = 896 and at d = 4096.** As models grow it is a falling fraction.
3. **Width was budget, and the probe's coordinates cost collateral.** At the
   saturated default cap the narrow writes (41 and 205 coordinates) leave the
   keyword refusal count where it was (1:1 and 2:3 flips) and carry the
   teacher's manner rather than its judgment. At matched energy, 410 coordinates carry 62% of full width's
   cross-entropy gain and 55% of its divergence. The probe's own 410 fit the
   teacher 1.3× better than four magnitude-matched controls (z ≈ 3.9) and pay
   98× their MMLU divergence for it. On 400 red-team prompts, adjudicated on
   substance by blind judges against a written rubric, the full-width write
   withholds the requested assistance on 21 pairs and supplies it on 2
   (p = 0.0001); both 410 writes at parity do the same (16:5, 15:2). What is
   withheld is legitimate information 40 times, dual-use material 7 times and
   harmful specifics 4 times, and the writes also release content the backbone
   had refused: racist tropes on one prompt (the wide write and the probe-best
   410) and a partial methamphetamine precursor list on another (both 410
   writes). A content-free injection of the same size and gate
   changes 8 decisions one way and 4 the other, so the withholding is the
   document's content: a blunter refusal, paid for mostly in helpfulness.
4. **Two bridges of different kinds compose on one backbone without joint
   training, and joint training makes the composition worse.** Neither channel
   costs the other its payload (physics accuracy stays at 1.000, and joint
   training costs constitution cross-entropy without buying anything); the
   composition is not inert on the backbone (MMLU divergence 0.072 where the
   constitution channel alone stayed at 0.005 and the physics channel alone sits
   at 0.041, with no significant accuracy change: MMLU 75 → 77, 3:1, p = 0.63).

[docs/status.md](docs/status.md) records every measurement, what is still
running, and what none of it establishes. Read it before treating anything here
as an outcome.

## The code

`psilm2.dual.PsiDualMLX` runs both channels in one forward pass, reading and
injecting at each channel's own depth. It subclasses `psilm.mlx.model.PsiLMMLX`
and overrides only the coupling step, so the physics loss, the no-harm arm and
the physics generate path are inherited rather than copied. `PsiDualMLX` takes a
per-channel arm — `psilm` writes, `zeroed` builds the tokens and measures the
gate without writing, `off` does not touch the bridge — which is what makes the
channels' interaction measurable from either side.

`python -m psilm2.dual_self_test` proves nine properties on a tiny random stack
on the CPU, in seconds and with no weights; three are bit-identity with the
single-channel models ΨLM trains ([docs/self-test.md](docs/self-test.md)).
`python -m psilm2.verify_qwen35` re-runs the three bit-identity properties, plus
the gate-interaction check, on the real 9B checkpoints (it needs `--model` or
`PSILM_BACKBONE` pointing at the backbone). The training chains, the evaluation harness, the value-neuron
tooling and every result file live in the ΨLM checkout, which this package
imports.

## Artifacts

| where | what |
|---|---|
| [ryoji-info/Qwen3.5-9B-PsiLM](https://huggingface.co/ryoji-info/Qwen3.5-9B-PsiLM) | the NVFP4 backbone and the physics bridge |
| [ryoji-info/PsiLM-2](https://huggingface.co/ryoji-info/PsiLM-2) | the constitution partner model, every trained constitution bridge (9B and 0.5B), the physics bridge in the same layout, and the guard-rail task caches |
| [ryoji-info/PsiLM](https://github.com/ryoji-info/PsiLM) | code, data, chains, and every evaluation record (`results/bench/*_summary.json`, `results/constitution/`) |

## Evaluating it yourself

```bash
pip install -U huggingface_hub                      # the `hf` downloader (`hf auth login` first while a repo is still private)
git clone https://github.com/ryoji-info/PsiLM && git clone https://github.com/ryoji-info/PsiLM-2
hf download ryoji-info/Qwen3.5-9B-PsiLM --local-dir hub/backbone     # 8 GB: backbone + physics bridge + FNO
hf download ryoji-info/PsiLM-2 --local-dir hub/psilm2                 # partner, bridges, caches
cd PsiLM && python -m venv .venv && .venv/bin/pip install -e . -e ../PsiLM-2
# the constitution channel alone, on the recorded 100 items
.venv/bin/python eval/bench_guardrail.py --tag mine_all --n 100 \
    --tasks-cache ../hub/psilm2/caches/tasks_const_qwen35_n100.json \
    --model ../hub/backbone --hf-tokenizer ../hub/backbone --bridge-kind constitution \
    --ckpt ../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json --max-new-mmlu 256 \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed --kl --seed 0
# both channels (the FNO beside the backbone loads without torch)
.venv/bin/python eval/bench_guardrail.py --tag mine_both --n 100 \
    --tasks-cache ../hub/psilm2/caches/tasks_const_qwen35_n100.json \
    --model ../hub/backbone --hf-tokenizer ../hub/backbone --bridge-kind dual --dual-channels both \
    --phys-ckpt ../hub/psilm2/physics/qwen3.5-9b/bridges.safetensors \
    --fno ../hub/backbone/physics/fno_burgers_singlemode.safetensors \
    --ckpt ../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json --max-new-mmlu 256 \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed --kl --seed 0
```

Every flag above is one the recorded runs used (`--max-new-mmlu 256` included;
the task cache checks them, and identifies the tokenizer by a fingerprint of its
behaviour rather than by its path, so any copy of the backbone restores it). The
`base` arm is deterministic and must reproduce the recorded rows item for item;
`zeroed` must equal `base` to four decimals; `psilm` is the arm under test.
Compare against `results/bench/const_qwen35_all_guardrail_summary.json` and
`dual_qwen35_both_guardrail_summary.json` in the ΨLM checkout. Everything runs on
one Apple M2 (24 GB); a four-dataset guard-rail takes about four hours. The
tracked `results/stage2/fno.pt` is the same FNO in torch form and needs
`pip install -e ".[stage1]"`; rebuilding a task cache from scratch needs
`".[bench]"`.

## License

Apache 2.0. Claude's constitution is released by Anthropic under CC0 1.0; the
copy used to build the partner model carries its provenance alongside it.

If this work is useful to you: [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).
