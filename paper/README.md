# Two papers

`psilm2.tex` — **this repository's paper.** The constitution bridge and dual-bridge
operation: where in a frozen backbone's hidden state a disposition has to be written,
how much of that state it has to disturb, and whether two bridges from unrelated
partners coexist. Builds with `latexmk -pdf psilm2.tex`.

`psilm.tex` — the **companion** paper, on coupling a frozen language model to a frozen
physics model, which is where this interface comes from. Its home is the
[PsiLM](https://github.com/ryoji-info/PsiLM) repository; the copy here is vendored so
the two build against one `refs.bib` and one `figs/`, and it is cited as
`furui2026psilm` rather than extended. **Do not revise it here** — revise it in PsiLM.

Both share `refs.bib`.

## Status

`psilm2.tex` carries `\pending{...}` markers, rendered in red, wherever a number is
not yet in the committed logs. Grep for them before treating the manuscript as
finished:

```
grep -n 'pending{' psilm2.tex
```
