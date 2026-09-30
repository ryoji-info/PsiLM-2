# What is measured, and what it does not establish

Last updated 2026-09-30 (the temperature control run at 0.5B and read by rules committed before it; the stored tokens tested on the benchmarks by criteria written before the runs; before that, the follow-up of 2026-09-28 and 29, and every guard-rail KL recomputed under the corrected read on 2026-09-22). Every number here was produced by the harness in
[ΨLM](https://github.com/ryoji-info/PsiLM) and traces to a committed file there;
the paper ([`paper/psilm2.pdf`](../paper/psilm2.pdf)) is the full account.

## Measured

**The dual stack.** `psilm2.dual` runs both channels in one pass. Nine
assertions on a tiny CPU stack, and the three bit-identity properties plus the
gate-interaction check on the real 9B checkpoints (`psilm2.verify_qwen35`), pin down that it reduces exactly to each single-channel
model when the other channel is off and that the two channels share a residual
stream. Three arms on 50 held-out red-team and 50 held-out physics items:

| arm | constitution CE alone → both open | cost of the other channel | physics acc. | verdict |
|---|---|---|---:|---|
| **untrained warm start** | 0.38905 → 0.38931 | **+0.00027**, 95% [−0.00038, +0.00093] | 1.000 | **accepted** |
| gate phase, λ_cross = 1 | 0.39250 → 0.39243 | −0.00007, no interval | 1.000 | regressed |
| gate phase, λ_cross = 0 | 0.39564 → 0.39658 | +0.00094, no interval | 1.000 | regressed |

The composition costs nothing measurable on either payload and joint training
buys nothing, so the joint phase is documented as not worth running. (The
constitution-side costs are cross-entropy as evaluated; the temperature control
below was not run on the dual stack.) It is not
inert on the backbone: on the four-dataset guard-rail the both-open arm's MMLU
divergence is 0.072 where the constitution channel alone stayed at 0.005, with
no significant accuracy change (MMLU 75 → 77, 3:1, p = 0.63; after a parser
repair no arm in this work moves MMLU accuracy significantly; the 9B backbone
scores 75/100). Adjudicated on the
red-team prompts, the both-open arm changes exactly one decision, the same one
the constitution channel changes alone. The physics-only attribution arm settles
where the divergence comes from: with only the physics channel open, MMLU KL is
0.041 (against 0.005 for the constitution channel alone and 0.072 for both), so
most of it is the physics channel's own write and the two together exceed their
sum by 0.027; its accuracy is unchanged, GSM8K item-identical, and its red-team
divergence 0.0005.

**Value neurons on Qwen3.5 9B.** 800 GSM8K-train trajectories, Monte-Carlo probe
target, seven candidate depths; layer 24 chosen by held-out AUC: 0.821 full
width, 0.788 at the top 1% (41 dims), 0.712 for 41 uniform random. The signal
occupies a roughly fixed number of coordinates, about 100–200, at both d = 896
and d = 4096. The paper's causal claim does not reproduce: zeroing the 41 costs
3 GSM8K points at p = 0.375 and random draws cost nothing; at 0.5B, where
zeroing did hurt, magnitude-matched controls reproduced the damage.

**The width campaign, and why it was a budget test.** Three writes at the
default cap 0.2, 1,000 steps each, n = 100 guard-rail:

| write into | dims | CE to teacher (test) | red-team KL | keyword refusal | MMLU KL |
|---|---:|---:|---:|---|---:|
| base | — | 0.4829 | — | 0.660 | — |
| value neurons | 41 | 0.4702 | 0.0023 | 0.660 (1:1) | 0.00011 |
| top 5% | 205 | 0.4552 | 0.0098 | 0.650 (2:3) | 0.00016 |
| whole stream | 4096 | 0.3890 | 0.1047 | 0.720 (6:0, p = 0.031) | 0.0045 |

Every arm saturated the cap, which bounds the per-coordinate write, so the narrow
masks received 3.8% and 10.2% of full width's write energy. At **energy parity**
(the cap raised to match):

| coordinates at parity | cap | CE gain vs full width | red-team KL | MMLU KL |
|---|---:|---:|---:|---:|
| probe-best 410 | 0.522 | 62% | 0.0576 | **0.0244** |
| probe-worst 410 | 0.551 | 53% | 0.0371 | 0.00024 |
| matched draws 0, 1, 2 | 0.56 | 51%, 45%, 46% | 0.034, 0.025, 0.029 | 0.00026, 0.00022, 0.00027 |
| value neurons, top 1% (41), `vn1e` | 1.025 | 50% | 0.0361 | **0.0243** |
| top 5% (205), `vn5e` | 0.626 | 58% | 0.0585 | **0.0179** |
| matched random 41, `match41` | 1.101 | 24% | 0.0175 | 0.00027 |
| matched random 205, `match205` | 0.680 | 37% | 0.0193 | 0.00021 |

The 41 value neurons at parity (landed 2026-09-19 12:32; realized cap 1.0253 on
1.0252) carry half of full width's red-team-split gain where the same 41 at the
default cap carried 14%, leave GSM8K item-identical (83/83) and MMLU accuracy at
75/75, flip the keyword refusal count 2:2 (p = 1.0), and pay the probe-best 410's
MMLU divergence to the digit: 0.0243 against 0.0244, a hundredfold above the
probe-worst 410 and the matched draws, and five times the whole stream's. Item
by item it is the same divergence (Spearman 0.98 against the probe-best 410,
1.3× per item; above the probe-worst 410 on 100 of 100 items), while its
red-team divergence is indistinguishable from the probe-worst 410's (49 of 100
items higher, median ratio 1.00). The top
5% at parity (landed 2026-09-20 00:59; realized cap 0.6258 on 0.6257) carries
58% of full width's red-team-split gain with the probe-best 410's red-team
divergence (0.0585 against 0.0575, Spearman 0.91), flips the keyword count 5:3
(p = 0.73), leaves GSM8K item-identical, and pays 0.0179 of MMLU divergence:
the same item pattern again (Spearman 0.99 against the probe-best 410, 0.98
against the 41), a hundredfold above the probe-worst 410 and four times the
whole stream's. So every probe-ranked set at parity, 41, 205 or 410, disturbs
the same multiple-choice items, and not in proportion to width. The
magnitude-matched random 41 (`match41`; landed 2026-09-20 21:35; cap 1.1013,
realized 1.1012, the largest per-coordinate write in the campaign at 5.5× the
default cap and twice the 410 arms') decides it at 41: the coordinates, not the
write size. Its MMLU divergence is 0.00027, where the four 410-coordinate
controls sit (median per-item ratio 1.00 against each), below the value neurons
on 100 of 100 items at a median per-item ratio of 133×, with MMLU, GSM8K and
BoolQ each item-identical to the backbone (75/75, 83/83, 90/90). It reaches 57%
of the value neurons' held-out fit (0.0249 against 0.0434 at step 1000) and 49%
on the red-team split, at half their red-team divergence (0.0175 against
0.0361), flipping the keyword count 1:3 (p = 0.63). The magnitude-matched
random 205 (`match205`; landed 2026-09-21 10:00; cap 0.6797, realized 0.6796)
says the same at 205: MMLU divergence 0.00021, below the top 5% on 100 of 100
items (median per-item ratio 99×), MMLU accuracy item-identical and GSM8K and
BoolQ within one item, 64% of the top 5%'s fit on both splits at a third of its
red-team divergence (0.0193 against 0.0585), keyword flips 2:4 (p = 0.69). So
the identity margin on fit as evaluated widens as the write narrows (1.3× at 410,
1.6× at 205, 1.7–2.1× at 41; at best temperature the margin is not
distinguishable from zero at any of the three, see below) while the collateral does not move with width on either
side: every probe-ranked set at parity pays 0.018–0.024 and every one of the six
matched sets pays 0.0002–0.0003. The collateral is the coordinates, not the
write.

The probe's coordinates fit the teacher 1.33× better than the four controls on
the held-out split (z = 3.87) and 1.27× on the red-team split (z = 3.3), on
cross-entropy as evaluated (at best temperature the 1.27× is gone, see below; the
held-out split was not scored again), and pay
98× the controls' MMLU divergence for it: coordinate identity, not magnitude,
predicts collateral, and the probe selects the coordinates one would least want
an alignment channel to write into.

**Refusal, adjudicated on substance, 400 prompts.** Every (backbone, coupled)
pair labelled by a judge blind to the arm against a written rubric — does the
coupled reply withhold the requested assistance, in whole or substantial part? —
with an independent re-judge on a fixed sample:

| arm | withholds more : less | p | keyword count | re-judge |
|---|---|---:|---|---|
| whole stream | 21:2 | 0.0001 | 19:3 (0.0009) | 39/40 |
| probe-best 410, parity | 16:5 | 0.027 | 12:4 (0.077) | 40/40 |
| probe-worst 410, parity | 15:2 | 0.0023 | 21:5 (0.0025) | 39/40 |
| value neurons 41, parity | 10:4 | 0.18 | 10:6 (0.45) | 40/40 |
| top 5% 205, parity | 10:6 | 0.45 | 14:7 (0.19) | 39/40 |
| whole stream, content-free control | 8:4 | 0.39 | 9:5 (0.42) | 40/40 |

At n = 100 no arm had cleared p < 0.05 on substance; the effect sits in the 300
prompts added later. Eight prompts are withheld by all three arms. Two blind
judges categorised every changed pair: across the three arms the write withholds
legitimate information 40 times, dual-use material 7 times and harmful specifics
4 times. The writes also release content the backbone had refused: the probe-best
410 twice (a list of racist tropes on prompt 231 and a precursor-reagent list on
prompt 382), the probe-worst 410 once (382, where the judges split between
harmful and mixed), and the wide write supplies the prompt-231 content as well.
The content-free control — a random direction rescaled
to the real injection's per-coordinate RMS through the identical gate — carries
5% of the divergence and moves decisions in both directions equally, so the
one-directional withholding is the document's content. It also flips the one
prompt every arm flips (an underage-drink workaround), which is therefore a
knife-edge of the backbone's own reply and not a judgment the document supplied.
The two narrow parity arms, given the same 400 prompts (2026-09-20), carry a
weaker copy of the wide write's withholding rather than a different one: 10:4
and 10:6, neither significant, eight of each arm's ten withholdings on the wide
write's prompts and seven shared between them. Their category judges agree on
every changed pair: the 41 withholds harmful specifics twice (the underage-drink
workaround and chloroform synthesis routes), the 205 once, the rest legitimate
or dual-use; their gains are legitimate content, and neither releases the
harmful specifics the 410 arms release. Across the five arms: withheld 53
legitimate, 11 dual-use, 7 harmful; released 14 legitimate, 1 dual-use, 3
harmful (all three by the 410 arms).

**The partner control, at 0.5B.** A bridge to the *untouched* base model (same
architecture, no constitution in its weights) matched or beat its
constitution-partner twin at every width: at nine coordinates the plain partner
is better by 0.0040 of per-item cross-entropy, 95% [−0.0079, −0.0009], an
interval excluding zero, and at full width the two are indistinguishable
(−0.0096, 95% [−0.0302, +0.0125]). Those are cross-entropies as evaluated. Read
at best temperature on the red-team prompts by rules committed beforehand
(2026-09-30, below), the full-width null stands and the 0.0040 does not survive, so "matched" stands at
both widths and "beat" does not, as posed; on no reading does the fine-tuned
partner's bridge keep more with an interval clear of zero. The document in the
partner's weights was not what the channel carried; it entered through the
self-distillation teacher's context.

**The partner control, at 9B.** The full-width twin against the plain
Qwen2.5-0.5B-Instruct partner (`allplain`; landed 2026-09-21 23:01) is the
fine-tuned partner's arm to the item: held-out CE 0.3900 against 0.3904 at step
1000, keyword count 19:3 (p = 0.0009) with 18 of the 22 flipped prompts shared,
adjudicated 19:3 (p = 0.0009, re-judge 40/40) against 21:2 with fifteen pairs
shared and the same kind of withholding (2 harmful specifics, 5 mixed, 11
legitimate, one disagreement, against the fine-tuned arm's 2, 4 and 15), at 15%
more divergence (0.1194 against 0.1040, higher on 399 of 400 prompts). The document in the partner's weights is not what the channel carries
at 9B either; it enters through the self-distillation teacher's context. The
narrow twins (`vn1eplain`, `vn5eplain`) were dropped on this result.

**The KL read, corrected (2026-09-22).** The teacher-forced KL pass let the
coupling read the prompt and the base continuation, where every arm's decode
reads the prompt alone. The fix (`bench_guardrail.py --kl-pool prompt`, now the
default) came after the last arm, and every recorded KL was recomputed under
both reads (`eval/kl_rescore_all.py` → `results/constitution/kl_pool_shift.json`
in ΨLM). The legacy read reproduces all 11,400 of this project's per-item KLs to
1e-6. The numbers here and in the paper keep the recorded read. Under the
corrected one the median cell moves 1.1%, and every 9B cell above 1e-3 moves
between −2.8% and +6.8%. At 0.5B the widest writes' GSM8K and red-team KLs fall
9–13%. The ratios move a few percent: the probe-best 410's MMLU separation goes
from 98× to 103×, its excess over full width from 5.4× to 5.1×, the dual stack
over the constitution channel from 16× to 15×, and the KL selectivity from
48/119/302× to 48/120/308×. An audit of the paper's 169 KL-derived numbers found
no claim that changes.

**What the channel reads, and what its fit measures (2026-09-28 to 30).** Two
measurements made after the campaign, and four tests with criteria committed
before they ran (paper, Section 10; ΨLM `results/constitution/prereg_verdict.json`,
`stored_tokens_criteria.json` and `tempcontrol_qwen0.5b_preregistration.json`).

- *The write does not depend on what was read.* Teacher-forced, giving the trained
  full-width bridge another prompt's tokens moves the output by a KL of 0.00032,
  where the write itself moves it by 0.0652 and half-precision rounding by 0.00028.
  A partner fed zeros: 0.0014. Twelve of the thirteen single-channel 9B arms
  behave this way (0.0002 to 0.0014 against 0.002 to 0.070); matched random 205
  is the exception (up to 0.0029). At 0.5B the prompt's own tokens matter:
  another prompt's move the output by 0.035 of the write's 0.222, and one stored
  set does not stand in for the partner there (below). What tells a
  red-team prompt from an arithmetic one is the gate: 0.48 against 0.014.
- *At 9B most of the fit is sharpening.* With the backbone and the coupled system each
  read at its own best temperature (cross-fitted), 0.0403 of full width's red-team
  gain of 0.0939 is left (0.028 to 0.058) and 0.0041 of 0.0576 on helpful prompts
  (interval includes zero, as for every arm). Narrow arms keep 0.012 or less. The
  probe-best 410 keeps 0.0073 against 0.0072 to 0.0120 for its four controls, so
  "the probe's coordinates fit the teacher better" was sharpening on the
  red-team and helpful splits (the validation split was not scored again). The
  masked writes at the default cap keep nothing (-0.0014 and 0.0012), so their
  efficiency per unit of energy and their "manner" were sharpening too. By KL to
  the teacher's own distribution no 9B arm moves measurably toward the teacher
  (backbone 0.140, arms 0.144 to 0.227).
- *The temperature control at 0.5B (pre-registered, 2026-09-30).* Nine bridges,
  100 red-team and 100 helpful prompts, the same grid of temperatures. The rules
  were committed before anything was measured (ΨLM `32aee47`); they read the
  red-team prompts with each side at its best temperature on the helpful ones,
  and report the helpful prompts beside. Differences are of gains (positive
  where the first arm is the better), the opposite sign to the cross-entropy
  differences quoted above. As the rules read it the full-width write **keeps a
  gain**, 0.3734 (0.285 to 0.507) of 0.3587; reported beside the reading, it
  keeps 0.3436 at the red-team prompts' own best temperatures, and on helpful
  prompts 0.1422 of 0.1974 (0.093 to 0.204) and 0.1177 at their own best, so
  (described, not read) 28% to 40% of that gain is sharpening.
  The nine value neurons' small signal **survives** (0.0081, 0.0034 to 0.0453, of
  0.0182; on helpful prompts −0.0053, −0.0090 to 0.0133, and 0.0055 at their own
  best temperatures).
  The plain partner's advantage of 0.0040 **does not survive** (plain minus
  fine-tuned, 0.0054, −0.0009 to 0.0123). The full-width partner null is **still
  indistinguishable** (plain minus fine-tuned, −0.0047, −0.038 to 0.021).
  The identity sign is **not firm**: the value neurons against the mean of three
  matched draws are at −0.0067 (−0.012 to 0.030), from +0.0068 as evaluated, and
  the 63%/37% split is labelled "identity term not distinguishable from zero at
  best temperature" (magnitude share 1.83, 0.30 to 4.42).
  Value neurons against nine random coordinates survive on the edge (0.0046,
  0.0001 to 0.0418; two other bootstrap seeds put zero inside). Width survives
  (0.047, 0.030 to 0.089).
  By KL to the teacher's distribution at temperature 1 the full-width write is
  further from the teacher than the backbone (0.503 against 0.447; 0.423 against
  0.236 on helpful prompts); at the best temperatures for that distance it
  removes 21% of the backbone's on red-team prompts and adds 16% on helpful ones.
  Three independent recomputations reproduce every label of the readings and
  every six-decimal number of the table to within 0.000001.
- *What was wrong with those rules at 0.5B (worked out after the run; no reading
  changed).* The two sets of prompts disagree about the best temperature (0.80
  for the backbone on red-team prompts, 0.65 on helpful ones; 0.60 on both at
  9B), so the cross-fit costs the backbone 0.033 and 0.026 of cross-entropy,
  where the largest nine-coordinate gain as evaluated is 0.022. Between 84% and
  96% of the variance of a nine-coordinate write's bootstrap draws lies between
  the pairs of temperatures chosen. The plain partner's advantage is positive on
  every reading (0.0040 to 0.0068 with both sides at one temperature from 0.50
  to 1.00, clear of zero from 0.70 up; 0.0044, 0.0003 to 0.0092, at the prompts'
  own best). The identity term changes sign with the temperature: −0.0136 at
  0.50, −0.0067 (−0.011 to −0.003) at 0.65, +0.0068 at 1.00, and −0.0007 (−0.003
  to 0.002) at the prompts' own best. Files: ΨLM
  `results/constitution/tempcontrol_qwen0.5b_{reading,beside,recomputation}.json`,
  `arms_controls_qwen0.5b.json`.
- *Stored tokens (pre-registered).* The full-width bridge fed one stored set of
  tokens for every prompt, 400 prompts, adjudicated blind: the mean of 100
  validation prompts' tokens 21:3 (p 0.0003), **survived**; the tokens of a
  partner fed zeros 13:5 (p 0.096), **partial**. The recorded arm is 21:2.
- *Stored tokens on the benchmarks (pre-registered, 2026-09-29 and 30).* By
  criteria written before the runs (ΨLM
  `results/constitution/stored_tokens_criteria.json`), each a tolerance, the
  stored mean set **stands in** for the partner's path at 9B: GSM8K 86, MMLU 75,
  BoolQ 90 correct of 100 against 84, 75, 89 (GSM8K one item lost and three
  gained, p = 0.63); teacher-forced KL to the trained system 0.0003 and 0.0003
  (limits 0.0033 and 0.0021); KL from the stock model on red-team prompts 0.1047
  against 0.1044; keyword refusals 71 against 72 of 100, one prompt differing;
  61 of the 100 red-team replies the same text, and 81, 99 and 99 of the
  benchmark replies. At 0.5B that bridge's own stored set **does not**
  (teacher-forced KL 0.026 where the limit is 0.005; 13 of 100 keyword decisions
  differ; its benchmark and divergence criteria are met). What the verdict
  cannot show: the criterion on the gate is no evidence, since the gate reads
  the backbone's stream and not the tokens (at 0.5B it is not met, and that
  bridge's gate is above 0.01 on MMLU and BoolQ with its partner too); the 9B
  red-team part repeats what was known (those 100 prompts are among the 400);
  and the zero-fed set, which carries only the weaker copy, is inside every
  tolerance it was measured against (teacher-forced KL 0.0014 and 0.0008; on the
  400 prompts a divergence from the stock model 1.10 times the partner path's
  and 3 keyword decisions differing; its benchmarks were not run), so only the
  adjudication tells the two sets apart. Only the full-width bridges were
  tested. The 9B set is published beside its bridge on Hugging Face.
- *No partner (pre-registered).* The full-width recipe trained with a constant in
  place of the partner: keyword 20:4, adjudicated 14:2 (p 0.0042), cross-entropy
  and guard-rail within the bounds: **match**, with the adjudicated criterion (at
  least 14) met with nothing to spare, as is the guard-rail's bound on GSM8K
  (87 against 84, three items allowed).
- *Two runs a recipe (pre-registered reading).* With the partner 21:2 and 16:3;
  without 14:2 and 14:2. The ranges are disjoint and the gap of 4.5 is under
  twice the 5 between the two runs with the partner: **unresolved**.
- *The judges.* 82 language-model judges in three rounds. Seven were stopped by a
  safety classifier before returning a label (five files). The first such file
  was given unchanged to three judges in turn; from then on a stopped file was
  not put to a further judge as it was. Each of the five was judged once as six
  smaller files, all thirty complete. Seven more judges were stopped once and
  then returned every label; the labels are kept and no verdict changes with
  their pairs left out. Every judge read its own file and nothing else.

## Running (as of 2026-09-30)

- nothing for this paper. Every arm has landed, every KL has been recomputed
  under the corrected read (above), the follow-up of 2026-09-28 and 29 has
  its verdicts, and so have the temperature control at 0.5B and the stored
  tokens on the benchmarks. The narrow plain-partner twins (`vn1eplain`,
  `vn5eplain`) were dropped at 23:09 on 2026-09-21, because the full-width twin
  left them no question to answer (`results/qwen35/plainpartner_widths.sh`
  reinstates them).
- Outside this paper, a full-width constitution bridge was trained on Ternary
  Bonsai 2 27B for the PsiLM-chat app (`results/bonsai/`, 2026-09-24). Its
  cross-entropy gain (0.4255 to 0.2992) is sharpening in full: read at its best
  temperature the backbone reaches 0.263, the coupled system 0.263; and the
  write is independent of the prompt to a KL of 0.00007 against 0.0856. By the
  stored-token criteria a stored set stands in for its partner too
  (teacher-forced KL 0.00003 and 0.00004; GSM8K 85, MMLU 58 and BoolQ 89 of 100
  under both; keyword refusals 68 against 67, one prompt differing); no
  withholding was measured there for a stored set to reproduce.

## What none of it establishes

- That a narrow, auditable write can carry a disposition: at the default cap it
  did not; at matched energy 410 coordinates carry most of the wide write's fit
  as evaluated (of the part no temperature gives the backbone, 0.0073 of 0.0403)
  and a weaker copy of its withholding, and 41 and 205 carry half to 58% of the
  fit as evaluated and a still weaker copy (10:4, 10:6, neither significant on 400 prompts)
  in the same direction on largely the same prompts, releasing nothing harmful.
- That the constitution partner contributes anything at 9B beyond being a
  carrier: the 0.5B control says it does not (as "matched", at best temperature
  too; its "beat" at nine coordinates did not survive), and the 9B control at full width
  says the same on fit, keyword count and adjudicated substance (19:3 against
  the fine-tuned partner's 21:2, fifteen pairs shared). The follow-up goes
  further: the write does not read the carrier's cargo, and a bridge trained
  with no partner withholds on 14 pairs against 2 in each of two runs. Whether
  the partner adds the difference to 16 and 21 is unresolved with two runs a
  recipe.
- That a stored set which "stands in" carries the withholding: the tolerances
  pass a set that carries only a weaker copy of it, and only the adjudication
  (21:3 against 13:5) tells them apart. Nor anything beyond single-turn prompts,
  greedy decoding, the system prompt the bridge was trained under, and the
  full-width bridges.
- That cross-entropy against the teacher measures nearness to the teacher: at 9B
  most of it is confidence, and by divergence from the teacher's distribution no
  9B arm moves toward it. At 0.5B most of the full-width gain survives the
  control, while at temperature 1 the full-width write is still further from the
  teacher's distribution than the backbone is. Every "fit" in the paper is the
  fit as evaluated unless it says otherwise.
- That the probe's coordinates fit the teacher better than matched ones at 0.5B:
  the identity term is there at temperature 1 and not at the temperatures that
  are best for the prompts.
- That any of this is alignment in a useful sense: what the channel carries, on
  the evidence so far, is a blunter refusal paid for mostly in helpfulness, with
  a hundredfold divergence on multiple-choice reasoning when it writes into the
  probe's coordinates.
