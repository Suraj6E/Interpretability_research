# Planning: Interpretability Research

This is a living document. Items marked **(H)** are hypotheses we plan to test, not established facts.

---

## 1. Vision

Build an interactive, visual map of an LLM's insides: for any question, show which parts fired, which pushed the answer, and which were necessary. Start small (GPT-2 small) and build up to larger models and finer concepts.

## 2. Goals and non-goals

**Goals**
- **G1:** A visual model map for every question in the question set.
- **G2:** Check our methods against circuits already published for GPT-2 small.
- **G3:** Compare skills: which components are shared and which are specific.
- **G4:** Beginner-friendly and reproducible on free Google Colab.
- **G5:** A foundation to build up from (sparse-autoencoder features, larger models, attribution graphs).

**Non-goals (for now)**
- Analyzing closed models (ChatGPT, Claude). Their internals aren't accessible.
- Claiming a full explanation of any model.
- "Activating every part" of the model. Neuron-level coverage is easy to reach and meaningless, and the number of concept-level features has no fixed size.
- Training our own models or running a benchmark leaderboard.

## 3. Research questions

Short summary only. The full specification of each question (hypothesis, metric, control, statistical test, and what result would disprove it) is in `docs/RESEARCH_PROTOCOL.md` §4. If the two documents disagree, the protocol wins.

| ID | Question | Hypothesis |
|---|---|---|
| RQ1 | **Sparsity:** how many components does each skill depend on? | **(H1)** For each skill, the top 10% of components (ranked by denoising recovery), patched together, recover at least 80% of the logit difference on average. |
| RQ2 | **Sharing:** which components are shared across skills? | **(H2)** For some pairs of skills the overlap is larger than chance, and for others it isn't. Skills are neither fully shared nor fully separate. |
| RQ3 | **Agreement:** do correlational views (activation, DLA) predict causal importance? | **(H3a)** Activation strength agrees only weakly with causal rankings (median Spearman ρ < 0.3 per skill). **(H3b)** DLA agrees with causal rankings better than activation strength does. |
| RQ4 | **Scale:** does that agreement change with model size? | **(H4)** Agreement between DLA and causal rankings changes steadily with size across the Pythia suite. The direction is not predicted. |
| RQ5 (supporting) | Can a cheap screening method (attribution patching) be trusted on larger models? | Method check only, not a new contribution. It overlaps with Kramár et al. 2024 and Zhang & Wang 2026. |

RQ3 is especially useful because it directly tests the project's central caveat that "lit up" is not the same as "used."

Pythia is used for RQ4 because every size was trained on the same data in the same order, so size is the main thing that changes (`docs/COMPUTE.md` §7). It is one model family among several, **not** the project's final model (see §8 Phase 5 and §13).

**Later, exploratory (not yet in the protocol).** These need their own full specification before they can become claims.
- **Two-step questions:** does a larger model reason step by step or use a memorized shortcut? **(H)** It depends on the question. Planned for Phase 5 on a larger model.
- **Where shared parts sit:** **(H)** early-layer and late-layer components are shared more often; middle-layer components are more skill-specific. This refines RQ2.

## 4. Definitions (metrics)

| Metric | Definition | Meaning |
|---|---|---|
| **Task metric M** | **Logit difference (LD)** = logit(correct) − logit(wrong) for most skills; for greater-than, **probability difference (PD)** = P(year > YY) − P(year ≤ YY) (Hanna et al. 2023) | How strongly the model prefers the right answer. The patching formulas below use M wherever they say LD |
| **Activation strength** | Size (L2 norm) of a head's or MLP's output at the final token | How strongly a part fired (**correlational**) |
| **Direct logit attribution (DLA)** | A part's output projected onto the "correct minus wrong" answer direction, after the final layer norm | How much a part directly pushed toward the right answer (**correlational**; ignores indirect effects through later parts) |
| **Denoising recovery** | (LD_patched − LD_twin) / (LD_clean − LD_twin), patching the clean activation into the twin run | 0 = no effect, 1 = restoring this part alone brings back the full answer (sufficiency-like, **causal**) |
| **Noising damage** | (LD_clean − LD_patched) / (LD_clean − LD_twin), patching the twin activation into the clean run | 0 = no effect, 1 = corrupting this part alone drops the answer to the twin's level (necessity-like, **causal**) |

**FACT** (Heimersheim & Nanda 2024): the two patching directions answer different questions and can give different results. We report both. The word "necessary" is used only for noising results. The agreement metrics (Spearman ρ, and precision@k with k = 5% of components) are defined in `docs/RESEARCH_PROTOCOL.md` §2.4.

## 5. Question set design

**Rules** (exact inclusion rules: `docs/RESEARCH_PROTOCOL.md` §3)
1. The answer must be a **single token**, so the task metric is simple to compute.
2. A question and its twin differ in **one detail only**.
3. A question and its twin have the **same number of tokens**, so positions line up for patching.
4. Keep only questions the model answers correctly: the correct answer must be its **top-1 guess** (and the wrong one its top-1 on the twin). A wrong answer has no mechanism to explain.
5. Aim for 30–50 questions per skill, generated from templates.

**Generation:** `python -m interp.questions --config configs/questions_v0.yaml` writes every generated pair to `questions/skills_v0.json` (no model involved). `python -m interp.run` then keeps, per model, the questions that pass the rules. Templates and word lists live in the config; details and limitations in `docs/DATASHEET.md`.

**Format (one entry)**

```json
{
  "id": "ioi_001",
  "skill": "indirect_object",
  "template_id": "ioi_t1",
  "clean": "When Mary and John went to the store, John gave a drink to",
  "twin": "When Mary and John went to the store, Mary gave a drink to",
  "metric": "logit_diff",
  "correct": " Mary",
  "wrong": " John",
  "reference": "Wang et al. 2023"
}
```

Greater-than entries have `"metric": "prob_diff"` and `"year": 32` instead of `correct`/`wrong`. Fact-recall entries have a `"group"`: several twin candidates share one clean prompt, and only one is kept.

**Skills (version 1):** induction, indirect object, greater-than, fact recall, grammar agreement. Two-step reasoning is added in Phase 5, since it needs a larger model.

**Question set v0 on GPT-2 small** (kept / generated): induction 59/60, indirect object 65/160, greater-than 58/60, fact recall 27 of 127 clean prompts, agreement 42/60. Counts by exclusion rule: `results/gpt2-small/filter_report.json`.

## 6. Architecture

One Python package does all model work; one viewer draws everything. The viewer has two modes that share the same JSON format.

```
questions/*.json   configs/*.yaml
        │                │
        ▼                ▼
interp/  (Python + TransformerLens; the only code that runs the model)
  model.py     load a model at a pinned revision
  metrics.py   activation strength, DLA, denoising recovery, noising damage
  run.py       BATCH:  python -m interp.run --config configs/<exp>.yaml
  serve.py     LIVE:   python -m interp.serve   (local server for the viewer)
        │                         │
        ▼                         ▼
results/<model>/<id>.json    same JSON, sent over HTTP
        │                         │
        └────────────┬────────────┘
                     ▼
viewer/  (Svelte + D3, built into static files)
  static mode: GitHub Pages; loads saved results (question picker or drag-and-drop)
  live mode:   served by interp/serve.py; maps redraw when the prompt changes
```

Live mode keeps a single implementation of the model, so numbers in the viewer, the results files and the paper all come from the same code.

**Result format (one question)**

```json
{
  "schema_version": 2,
  "id": "ioi_001",
  "skill": "indirect_object",
  "model": "gpt2-small",
  "clean": "...", "twin": "...", "correct": " Mary", "wrong": " John",
  "tokens": ["<|endoftext|>", "When", " Mary", "..."],
  "n_tokens": { "clean": 15, "twin": 15 },
  "metric": { "type": "logit_diff", "clean": 3.36, "twin": -2.48 },
  "top1": { "clean": " Mary", "twin": " John" },
  "activation": { "heads": [[0.0]], "mlp": [0.0] },
  "dla":        { "heads": [[0.0]], "mlp": [0.0] },
  "attention":  [[[0.0]]],
  "top_predictions": [{ "token": " Mary", "prob": 0.677 }],
  "meta": { "git_commit": "...", "model": { "revision": "..." }, "processing": {}, "versions": {} }
}
```

`heads` is a layers × heads grid (12 × 12 for GPT-2 small) and `mlp` has one value per layer. `attention` is [layer][head][token]: how much the last token attends to each token, rounded to 4 decimals. All 144 heads are stored for GPT-2; for large models, store only the top-k heads (`docs/COMPUTE.md` §4). `denoising` and `noising` maps are added in Phase 3. Live mode returns the same shape without the question fields.

## 7. Visualizations

| ID | View | Design |
|---|---|---|
| V1 | **Model map** | Grid: rows = layers 0–11, columns = heads 0–11 plus one MLP column. Toggle between activation, DLA, denoising and noising. |
| V2 | **Clean vs. twin diff** | Same grid, colored by the difference between the two runs |
| V3 | **Attention viewer** | Click a head to see which earlier tokens it attends to |
| V4 | **Skill fingerprints** | Small grids side by side, one per skill |
| V5 | **Activation vs. causal scatter** | Each dot is one component: x = activation strength (or DLA), y = noising damage (or denoising recovery). Answers RQ3. |
| V6 | **Disagreement view** | Activation, DLA and causal grids for the same question, side by side. Cells in the top-k (k = 5% of components) of one view but not the other (the disagreement set, `docs/RESEARCH_PROTOCOL.md` §2.4) are outlined. The user picks which causal measure to compare against (noising or denoising). This is the core of contribution C1, and usage scenario S2 relies on it. |

**Design rules**
- Use a colorblind-safe palette. DLA and patching use a diverging scale (negative / zero / positive); activation uses a single-hue scale.
- Every cell shows its exact value and component name on hover.
- The public site needs no server: the built viewer is plain static files on GitHub Pages showing saved results. Typing new prompts needs the local server (`python -m interp.serve`).
- When the prompt changes, cells animate to their new values, so the eye can follow what changed.
- Clicking any cell opens a details panel for that component. Panels are added as needs become clear; first candidates: attention pattern, the output tokens it pushes up or down, its scores and ranks in every view, per-position activation, and the logit lens.
- Every view has a short "how to read this" note beside it.

## 8. Phases and milestones

Steps are listed in order. Timing depends on available time, so no dates are set.

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Setup** | README, PLANNING, LICENSE, question set v0 | Repository is public and the question set is drafted |
| **1. Activation map** | Recording script and V1 (activation only) | Maps render for all version-1 skills |
| **2. Contribution map** | DLA and V3 attention viewer | DLA shown in V1; attention viewer works |
| **3. Causal map** | Activation patching (noising and denoising), V2, validation | Known results reproduced qualitatively (e.g., induction heads appear in layers 5–7; IOI name-mover heads appear in late layers) |
| **4. Comparison and scale** | V4, V5, V6, scale study across Pythia sizes (RQ4), screening check (RQ5), written summary of RQ1–RQ5 | Findings written up, including what did *not* work |
| **5. Build up** | Larger models beyond Pythia (e.g., Gemma-2-2b; see §13), sparse-autoencoder features (SAELens), attribution graphs (circuit-tracer), exploratory two-step study (§3) | First concept-level map published |

Which model runs on which hardware in each phase: `docs/COMPUTE.md` §7.

## 9. Validation plan (not fooling ourselves)

1. **Known-circuit check:** reproduce published GPT-2 results before trusting new findings.
2. **Random baseline:** patch or ablate random heads as a control.
3. **Ablation choice:** main runs patch from the twin; mean ablation over the question set is the robustness check (`docs/RESEARCH_PROTOCOL.md` §6). Report both if they disagree.
4. **Many prompts:** report averages and spread over the whole question set, not the best example.
5. **Backup mechanisms:** remember that removing one part can let others take over, so a small effect doesn't always mean "unimportant."
6. **Report failures:** negative results go into the write-up too.

## 10. Risks and honest difficulties

| Risk | Impact | Mitigation |
|---|---|---|
| Activation maps get read as "importance" | Misleading conclusions | All three views side by side, with a caveat on every view |
| GPT-2 findings don't carry over to bigger models | Limited generality | State this clearly; Phase 4 tests a range of sizes (Pythia); Phase 5 adds larger models from other families |
| Twin questions differ in token length | Patching positions don't line up | Check token lengths automatically and drop mismatches |
| GPT-2 small can't do some skills | Nothing to explain | Filter to questions with LD > 0; leave hard skills to Phase 5 |
| Neurons are polysemantic | Hard to label parts | Treat labels as hypotheses; add sparse-autoencoder features later |

## 11. Community and outreach

- Publish the viewer on GitHub Pages so anyone can try it without installing anything.
- Add a 1280×640 social preview image showing a real model map.
- Open "good first issue" tasks (new question sets, new visualizations).
- Write a short post for each milestone, with screenshots and honest findings.

## 12. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-12 | Start with GPT-2 small | Many published circuits to check against; runs on free Colab |
| 2026-10-12 | Use TransformerLens | Standard library with easy access to every activation |
| 2026-10-12 | Static HTML viewer | No server; free hosting on GitHub Pages |
| 2026-10-12 | Three views (activation, contribution, causal) | "Activated" alone is misleading |
| 2026-10-12 | License: MIT (proposed) | Simple and permissive; encourages reuse |
| 2026-10-14 | Align §3, §4, §6–§10 with `docs/RESEARCH_PROTOCOL.md` v0.1: RQ1–RQ5, noising and denoising both reported, "necessary" only for noising, patching from the twin as the main ablation. Two-step reasoning moved to "later, exploratory" | The protocol is the newer, fuller spec; two conflicting versions must not exist before the protocol is frozen |
| 2026-10-12 (logged 2026-10-14) | Top-k uses k = 5% of components, rounded up | A share instead of a fixed count, so models with different numbers of components can be compared (`docs/RESEARCH_PROTOCOL.md` §2.4) |
| 2026-10-12 (logged 2026-10-14) | Drop questions with LD_clean − LD_twin < 1.0 | Recovery and damage ratios become unstable when the denominator is small (`docs/RESEARCH_PROTOCOL.md` §3) |
| 2026-10-14 | Add V6, a disagreement view (three views side by side, disagreements outlined) | It is the core of contribution C1, and no planned view showed it |
| 2026-10-14 | Code as Python scripts only; no notebooks for the pipeline | A real project is easier to manage and track; notebooks crash and get hard to follow; effort should go to the visualizations |
| 2026-10-14 | Viewer gets data by drag-and-drop (or file picker) of result JSON | Works when the page is opened by double-click, with no server and no export step |
| 2026-10-14 | Dependencies: pip + venv, `pyproject.toml` plus an exact-pinned `requirements.txt` | No new tools; Colab installs the same way; matches `docs/REPRODUCIBILITY.md` |
| 2026-10-14 | Viewer has two modes: **live** (a local Python server runs the model and the maps redraw as the prompt changes) and **static** (GitHub Pages, saved results). Replaces "Static HTML viewer" (2026-10-12) | The visualization must change when the input changes, which needs the model to run. A server keeps one model implementation (viewer numbers = paper numbers), makes patching easy, and scales to bigger models; running the model in the browser would not |
| 2026-10-14 | Frontend: Svelte + D3, with a build step (Vite). Replaces "no build step" | Live input and linked panels need automatic redraws; same stack as Transformer Explainer; the built output still hosts free on GitHub Pages. Drag-and-drop stays for loading saved results; opening the page by double-click is no longer a goal |
| 2026-10-14 | Code layout: shared `interp/` package; each step is one command (`python -m interp.<step> --config ...`) | Each piece is written once and tested; the live server reuses it; matches `docs/REPRODUCIBILITY.md` §6 |
| 2026-10-14 | Colab: one tiny launcher notebook with no logic (download, install, run) | Keeps goal G4 without notebook-based development |
| 2026-10-14 | Environment: Python 3.13, torch 2.14.1 (CUDA 13.0 build locally), TransformerLens 4.2.0 | Latest versions that support this machine (RTX 4070 Laptop, 8 GB); exact pins live in `requirements.txt` |
| 2026-10-14 | Pythia is the size suite for the scale study (RQ4), not the final model | Same data and training order at every size, so size is the main thing that changes; the final larger model is still open (§13) |
| 2026-10-14 | Load models as TransformerLens `TransformerBridge` with compatibility mode and default weight processing (fold LayerNorm, center writing weights and unembedding, fold value biases); recorded in every result's `meta.processing` | TransformerLens 4.0 (2026-09-21) removed `HookedTransformer`. Centering changes component output norms (it removes each output's mean, which the next LayerNorm removes anyway), so the setting must be fixed and reported |
| 2026-10-15 | Inclusion rule 3 made stricter: the correct answer must be the model's top-1 guess on the clean prompt, and the wrong answer its top-1 on the twin | LD alone kept "The Eiffel Tower is in the city of" although GPT-2's top guess was " London"; we explain the answer the model actually gives |
| 2026-10-15 | Greater-than scored with Hanna et al.'s probability difference (PD); the protocol's formulas now use a per-skill "task metric" M (LD or PD). Rule 4 gap for PD: ≥ 0.2, provisional | Greater-than has many correct answers, so one token pair would be a noisy stand-in; PD keeps results comparable with Hanna et al. 2023 |
| 2026-10-15 | DLA for greater-than projects onto mean W_U(years > YY) − mean W_U(years ≤ YY) | PD is not linear in the logits, so this direction is an approximation; stated wherever greater-than DLA is shown |
| 2026-10-15 | Question set v0: induction uses common words instead of random tokens; grammar templates start "I think that the …" / "Everyone knows that the …"; fact recall offers up to 12 twin candidates per clean prompt and keeps the first that passes | Words survive tokenization; a GPT-2 pilot gave the verb as top guess 92% of the time vs 41% for "The …"; equal-length twins are rare, so several candidates are needed |
| 2026-10-15 | Raised generated counts after a first GPT-2 run (IOI 80 → 160; fact recall +45 capitals, 4 → 12 twin candidates); no further tuning once fact recall reached 27 | Reach the planned 30–50 per skill without lowering the bar; stopped there to avoid fitting the question set to GPT-2's knowledge. No research-question result was looked at |
| 2026-10-15 | Saved results store attention from the last token for all heads (rounded to 4 decimals) and are written as compact JSON (about 23 KB per question) | Lets the viewer show any head's attention for saved questions; size stays small for GPT-2. For large models, store only top-k heads (`docs/COMPUTE.md` §4) |

## 13. Open questions

- ~~Colab notebook first, or a local Python package first?~~ Decided 2026-10-14: scripts only (§12).
- Which larger open model(s) are the final target after Pythia? (Phase 5 candidates: Gemma-2-2b, or another open model with pretrained sparse autoencoders.)
- Keep the repository name `Interpretability_research`, or switch to hyphens (see the README notes)?