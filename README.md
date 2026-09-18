<p align="center"><b>ΨLM-2 — one frozen language model, two frozen partners: a physics model and a document of values, each reached through small trainable bridges.</b></p>

## What ΨLM-2 is

ΨLM-2 keeps the [ΨLM](https://github.com/ryoji-info/PsiLM) premise — nothing
pretrained is fine-tuned — and puts two partners on the same frozen backbone at
once:

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
   saturated default cap the narrow writes (41 and 205 coordinates) transmit
   nothing. At matched energy, 410 coordinates carry 62% of full width's
   cross-entropy gain and 55% of its divergence. The probe's own 410 fit the
   teacher 1.3× better than four magnitude-matched controls (z ≈ 3.9) and pay
   98× their MMLU divergence for it. On 400 red-team prompts, adjudicated on
   substance by blind judges against a written rubric, the full-width write
   withholds the requested assistance on 21 pairs and supplies it on 2
   (p = 0.0001); both 410 writes at parity do the same (16:5, 15:2). What is
   withheld is legitimate information 40 times, dual-use material 7 times and
   harmful specifics 4 times. A content-free injection of the same size and gate
   changes 8 decisions one way and 4 the other, so the withholding is the
   document's content: a blunter refusal, paid for mostly in helpfulness.
4. **Two bridges of different kinds compose on one backbone without joint
   training, and joint training makes both worse.** Neither channel costs the
   other its payload; the composition is not inert on the backbone (MMLU
   divergence 0.072 where a single channel stayed at 0.005, with no accuracy
   change).

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
`python -m psilm2.verify_qwen35` runs the same properties on the real 9B
checkpoints. The training chains, the evaluation harness, the value-neuron
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
git clone https://github.com/ryoji-info/PsiLM && git clone https://github.com/ryoji-info/PsiLM-2
hf download ryoji-info/Qwen3.5-9B-PsiLM --local-dir hub/backbone
hf download ryoji-info/PsiLM-2 --local-dir hub/psilm2
cd PsiLM && python -m venv .venv && .venv/bin/pip install -e . -e ../PsiLM-2
# the constitution channel alone, on the recorded 100 items
.venv/bin/python eval/bench_guardrail.py --tag mine_all --n 100 \
    --tasks-cache ../hub/psilm2/caches/tasks_const_qwen35_n100.json \
    --model ../hub/backbone --hf-tokenizer ../hub/backbone --bridge-kind constitution \
    --ckpt ../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed --kl --seed 0
# both channels
PYTHONPATH=../PsiLM-2 .venv/bin/python eval/bench_guardrail.py --tag mine_both --n 100 \
    --tasks-cache ../hub/psilm2/caches/tasks_const_qwen35_n100.json \
    --model ../hub/backbone --hf-tokenizer ../hub/backbone --bridge-kind dual --dual-channels both \
    --phys-ckpt ../hub/psilm2/physics/qwen3.5-9b/bridges.safetensors \
    --ckpt ../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed --kl --seed 0
```

The `base` arm is deterministic and must reproduce the recorded rows item for
item; `zeroed` must equal `base` to four decimals; `psilm` is the arm under test.
Compare against `results/bench/const_qwen35_all_guardrail_summary.json` and
`dual_qwen35_both_guardrail_summary.json` in the ΨLM checkout. Everything runs on
one Apple M2 (24 GB); a four-dataset guard-rail takes about four hours.

## License

Apache 2.0. Claude's constitution is released by Anthropic under CC0 1.0; the
copy used to build the partner model carries its provenance alongside it.

If this work is useful to you: [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).
