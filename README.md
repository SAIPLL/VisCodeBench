# VisCodeBench

**VisCodeBench** is an end-to-end Text-to-Vis benchmark that reflects real-world practice: given a natural-language request and some data, a system must write the code that produces the requested chart. Its **2,672 samples** come from real users in three domains (educational, scientific and general-audience), and predictions are scored with data-centric metrics that compare the data a chart actually displays.

- 📄 **Paper:** [VisCodeBench: a new benchmark dataset for Text-to-Vis reflecting real-world practice](https://openreview.net/forum?id=noorBEVR4k) (COLM 2026)
- 🏆 **Leaderboard:** [codabench.org/competitions/16125](https://www.codabench.org/competitions/16125/)
- 🚀 **Tutorial:** [docs/tutorial.md](docs/tutorial.md), from a fresh clone to a leaderboard submission

## News

- **2026-11:** We updated the [running tutorial](docs/tutorial.md). It now runs a coding agent ([OpenAI Codex CLI](https://github.com/openai/codex) or [Claude Code](https://code.claude.com/docs)) inside a Docker container on every sample, handles usage limits, keeps each model's answers separate, and builds the submission file in one command.
- **Coming soon:** leaderboard results for new models (e.g. Codex-gpt-6-astra).

## Dataset

| Subset | Size | Domain | Source |
| --- | --- | --- | --- |
| **GitHub-T2V** | 843 | Educational | Visualisation tasks from assignment notebooks in online data-science, data-analysis and ML courses shared on GitHub. |
| **arXiv-T2V** | 805 | Scientific | Charts contributed by authors of 2024 arXiv papers through an ethics-approved survey, with their code, data and a newly written request. Longer outputs, more plot types and more subplots. On the leaderboard it is split into **easy** and **hard**. |
| **OWID-T2V** | 1,024 | General audience | Charts and datasets curated from [Our World in Data](https://ourworldindata.org/). The originals were drawn in custom JavaScript, so the references are the displayed data points and the original chart images. |

All samples were annotated with LLM assistance and reviewed by experts with 4+ years of coding and academic-visualisation experience. Compared with earlier benchmarks, VisCodeBench has longer inputs and outputs, more libraries per sample, more subplots per chart, and more varied data formats.

The sample inputs (data, context code and request) can be downloaded from the leaderboard. The reference solutions are kept private and used only for scoring, so that they don't leak into LLM training data.

## Getting started

Follow the **[tutorial](docs/tutorial.md)**. In short:

1. Clone this repository and download the sample inputs into `data/codabench`.
2. Build the Docker image, which includes the evaluation library and both coding agents.
3. Generate predictions with `run_agent.sh`, or with your own Text-to-Vis system.
4. Log the data your charts display with `python -m t2v_eval.batch_log`, and upload the resulting zip to the leaderboard.

## Evaluation library

The `t2v_eval` package in this repository is the scoring engine behind the leaderboard. It records the data that matplotlib, seaborn and geopandas actually draw, then compares predictions with references using **Displayed Data Similarity** and **Chart Type Accuracy**. Installation, the API and the supported plotting functions are documented in **[docs/t2v_eval.md](docs/t2v_eval.md)**.

## License

[MIT](LICENSE). See [docs/acknowledgements.md](docs/acknowledgements.md) for the work this project builds on.

## Citation

```bibtex
@inproceedings{
nguyen2026viscodebench,
title={VisCodeBench: a new benchmark dataset for Text-to-Vis reflecting real-world practice},
author={Hy Nguyen and Thomas James Elton and Ella Wilson and Cecile Paris and Andrew Reeson and Josiah Poon and Jonathan K. Kummerfeld},
booktitle={Third Conference on Language Modeling},
year={2026},
url={https://openreview.net/forum?id=noorBEVR4k}
}
```
