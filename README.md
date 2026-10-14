# Interpretability Research: A Visual Map of What Activates Inside an LLM

**See which layers, attention heads and MLPs light up, and which ones actually matter, when a language model answers a question.**
An open-source mechanistic interpretability project with interactive visualizations, starting with GPT-2 small.

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Status: early stage](https://img.shields.io/badge/status-early%20stage-orange.svg)

> **Status: early stage (planning).** This repository is just getting started. Sections marked 🚧 are planned, not finished. Nothing here is a research result yet. Star or watch the repo to follow along as the maps get built.

---

## Table of contents

- [What is this?](#what-is-this)
- [Why "activated" is not enough](#why-activated-is-not-enough)
- [The three maps](#the-three-maps)
- [The question set](#the-question-set)
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [How to read the maps (honest caveats)](#how-to-read-the-maps-honest-caveats)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [Related work and references](#related-work-and-references)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## What is this?

A large language model (LLM) is a stack of layers. GPT-2 small, for example, has **12 layers**, and each layer has **12 attention heads** and **1 MLP block**. That makes 144 heads and 12 MLPs in total.

When you ask the model a question, every one of these parts does some work. This project:

1. Asks the model a set of carefully designed questions.
2. Records what every layer, head and MLP does.
3. Draws it as an **interactive map of the model**, so you can see the inside of an LLM instead of just reading numbers.

Think of it like a brain scan for a neural network. The goal is to make LLM internals understandable for everyone, from curious beginners to interpretability researchers.

## Why "activated" is not enough

A part of the model can **light up** without affecting the answer, and a quiet part can be **essential**. So this project shows three different views of the same question:

| View | Question it answers | Analogy |
|---|---|---|
| **Activation** | Which parts fired strongly? | Which rooms have their lights on |
| **Contribution** | Which parts pushed the model toward its answer? | Who spoke up in the meeting |
| **Causal** | Which parts are *necessary* for the answer? | Who, if removed, would change the decision |

Only the causal view tells you what the model actually *uses*. The other two are cheaper and give useful hints.

## The three maps

🚧 *Screenshots coming soon.*

| Map | Method | Cost |
|---|---|---|
| **1. Activation map** | Output size (norm) of each head and MLP | 1 forward pass |
| **2. Contribution map** | Direct logit attribution: how much each part adds to the correct answer versus a wrong one | 1 forward pass |
| **3. Causal map** | Activation patching in both directions: copy one part's output from a "clean" question into its "twin" and measure how much of the answer comes back, and copy it from the twin into the clean question and measure how much of the answer breaks | Many forward passes |

Extra views (planned):

- **Attention viewer:** which earlier words each head looks at.
- **Clean vs. twin diff:** what changes inside the model when one word in the question changes.
- **Skill fingerprints:** side-by-side maps showing which parts are shared across skills and which are skill-specific.
- **Activation vs. causal scatter:** does "lit up" predict "necessary"?
- **Disagreement view:** the three maps side by side, with the parts where they disagree outlined.

## The question set

Questions are grouped by **skill**. Every question has a **twin**: the same sentence with one detail changed, so the correct answer changes. The twins are what make causal tests possible.

| Skill | Example question | Twin | Published reference to check against |
|---|---|---|---|
| Copying (induction) | Random tokens, repeated twice | Not repeated | Olsson et al. 2022 |
| Indirect object (IOI) | "When Mary and John went to the store, John gave a drink to" → **Mary** | Names swapped | Wang et al. 2023 |
| Greater-than | "The war lasted from 1732 to 17" → a number **> 32** | "…from 1701 to 17" | Hanna et al. 2023 |
| Fact recall | "The Eiffel Tower is in the city of" → **Paris** | "The Colosseum is in the city of" → Rome | Meng et al. 2022 (ROME) |
| Grammar agreement | "The keys to the cabinet" → **are** | "The key to the cabinets" → is | — |
| Two-step reasoning | "The capital of the state containing Dallas is" → **Austin** | Chicago → Springfield | Anthropic 2025 (bigger models only) |

Skills with published answers act as a **sanity check**. If our maps don't find the known circuits where earlier papers found them, our pipeline is wrong, and we fix it before trusting anything new.

## How it works

```
questions/*.json  →  run model (TransformerLens)  →  record activations
                  →  compute 3 maps               →  results/*.json
                  →  interactive viewer (HTML/JS, no server needed)
```

## Quickstart

🚧 **Not working yet.** This shows the planned usage.

```bash
git clone https://github.com/Suraj6E/Interpretability_research.git
cd Interpretability_research
pip install -r requirements.txt
python -m interp.run --model gpt2-small --questions questions/skills_v1.json
```

Then open `viewer/index.html` in a browser. A Google Colab notebook (free GPU, nothing to install) is also planned.

## Repository structure

🚧 Planned layout:

```
Interpretability_research/
├── README.md
├── planning.md
├── questions/        # question sets with twins (JSON)
├── interp/           # model loading, recording, metrics
├── notebooks/        # Colab notebooks
├── results/          # output JSON per question
├── viewer/           # interactive model map (HTML/JS)
└── docs/images/      # screenshots
```

## How to read the maps (honest caveats)

- **Activation is not importance.** A bright cell means "fired strongly," not "mattered."
- **Small models are not big models.** Findings on GPT-2 may not carry over to frontier LLMs.
- **Labels are hypotheses.** Calling a head a "name mover" is an interpretation until a causal test supports it.
- **A map covers one question set, not the whole model.** A part that is dark on our questions may be crucial for other tasks.
- **Neurons are not concepts.** One neuron can respond to many unrelated things (superposition). Finer, concept-level views using sparse autoencoders are on the roadmap.

## Roadmap

Full details are in [planning.md](planning.md).

- [x] Project plan and question-set design
- [ ] Phase 1: Activation map for GPT-2 small
- [ ] Phase 2: Contribution map and attention viewer
- [ ] Phase 3: Causal map, checked against known circuits
- [ ] Phase 4: Skill fingerprints, the disagreement view, an "activation vs. causal" study, and a model-size comparison (Pythia suite)
- [ ] Phase 5: Larger models beyond Pythia (e.g., Gemma-2-2b), sparse-autoencoder features, attribution graphs

## FAQ

**What is mechanistic interpretability?**
The study of *how* a neural network computes its answers, by reverse-engineering its internal parts, rather than only measuring *what* it outputs.

**Can I see inside ChatGPT or Claude with this?**
No. Closed models don't give outsiders access to their internal activations. This project uses open-weight models such as GPT-2 and, later, Gemma.

**Do I need a GPU?**
GPT-2 small is small enough for the free Google Colab GPU, and it also runs (slowly) on a regular CPU.

**How is this different from TransformerLens, Neuronpedia or circuit-tracer?**
This project is built on top of those excellent tools. Its focus is a beginner-friendly, question-set-driven visual map that puts three views (activation, contribution, causal) side by side.

## Related work and references

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): the library this project uses to read model internals
- [Neuronpedia](https://www.neuronpedia.org): an interactive platform for features and attribution graphs
- [circuit-tracer](https://github.com/safety-research/circuit-tracer): an open-source library for attribution graphs
- [ARENA](https://www.arena.education): a free mechanistic interpretability curriculum
- Olsson et al. (2022), [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Wang et al. (2023), [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- Hanna et al. (2023), [How does GPT-2 compute greater-than?](https://arxiv.org/abs/2305.00586)
- Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- Heimersheim & Nanda (2024), [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
- Elhage et al. (2022), [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Templeton et al. (2024), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- Lindsey et al. (2025), [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

## Contributing

Contributions are welcome at any level:

- Add a new question set (a new skill with twin questions).
- Improve or add a visualization.
- Reproduce a known circuit and compare the result with the paper.
- Fix documentation or explain a concept more simply.

Please open an issue first to discuss larger changes.

## Citation

```bibtex
@software{interpretability_research_2026,
  author = {Neupane, Suraj},
  title  = {Interpretability Research: A Visual Map of What Activates Inside an LLM},
  year   = {2026},
  url    = {https://github.com/Suraj6E/Interpretability_research}
}
```

## License

MIT. See [LICENSE](LICENSE).