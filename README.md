# VisCodeBench

**VisCodeBench** is an end-to-end Text-to-Vis benchmark that reflects real-world
practice. It contains **2,672 samples** drawn from three domains — educational,
scientific, and general-audience — with varying levels of complexity, and it
pairs those samples with data-centric evaluation metrics for scoring generated
visualisations.

> **Paper:** *VisCodeBench: a new benchmark dataset for Text-to-Vis reflecting
> real-world practice* (COLM 2026).
> Hy Nguyen, Thomas James Elton, Ella Wilson, Andrew Reeson, Cécile Paris,
> Josiah Poon, Jonathan K. Kummerfeld.
> The University of Sydney · CSIRO's Data61.

> **🏆 Leaderboard:** https://www.codabench.org/competitions/16125/

This repository provides the **evaluation code**, **representative samples**, and
**prompts**. The full benchmark dataset is kept private and accessed via the
leaderboard to prevent web scraping and use in LLM training.

## What is Text-to-Vis?

Text-to-Vis is the task of taking a human-language request for a visualisation,
together with some input data, and generating code that produces the
corresponding chart. VisCodeBench evaluates systems on realistic requests: the
charts, data, and requests originate from real users rather than synthetic
templates, and many samples require substantial data processing and
customisation.

## Dataset Overview

VisCodeBench is built from three sources. After collection, the raw data went
through an annotation process combining LLM assistance and expert human review
(annotators with 4+ years of coding and academic-visualisation experience).

| Subset | Size | Domain | Description |
| --- | --- | --- | --- |
| **GitHub-T2V** | 843 | Educational | Visualisation tasks from assignment notebooks in online data-science, data-analysis, and ML courses shared on GitHub. Each sample includes underlying data, context code, a natural-language request, the reference code, and the resulting chart. |
| **arXiv-T2V** | 805 | Scientific | Charts contributed by authors of 2024 arXiv papers via an ethics-approved survey, with the corresponding code, underlying data, and a newly written request. Exhibits longer outputs, more plot types, and more subplots. |
| **OWID-T2V** | 1,024 | General audience | Charts and datasets curated from [Our World in Data](https://ourworldindata.org/). Original charts were produced in custom JavaScript, so instead of code we provide reference displayed data points and the original chart images for evaluation. |

Compared with existing benchmarks, VisCodeBench places higher demands on models:
longer inputs and outputs, more distinct libraries per sample, multiple subplots
per chart, and greater flexibility of data format — reflecting genuine
real-world visualisation work.

## Leaderboard & Submissions

Because the full dataset is held out, evaluation is run through the Codabench
competition rather than by releasing test data:

**➡️ https://www.codabench.org/competitions/16125/**

To participate:

1. Open the competition page and read the task description and rules.
2. Register for the competition on Codabench.
3. Generate visualisation code for the provided inputs using your Text-to-Vis
   system.
4. Package your outputs in the required submission format (see the competition
   page) and upload them.
5. Your submission is scored automatically with the metrics in this repository,
   and results appear on the leaderboard.

The representative samples and prompts in this repository illustrate the input
format and let you validate your pipeline locally before submitting.

## Evaluation Library (`t2v_eval`)

The `t2v_eval` package is the scoring engine behind the leaderboard. It logs the
data that plotting libraries actually render, then compares predicted
visualisations against reference visualisations using data-centric metrics such
as **Displayed Data Similarity** and an image-level **Accessibility** score.

Installation, logging, scoring, the full API reference, supported plotting
libraries, and tests are documented in **[docs/t2v_eval.md](./docs/t2v_eval.md)**.

Quick start:

```bash
pip install -e .
```

```python
from t2v_eval.evaluation import calculate_similarity_score

similarity = calculate_similarity_score(
    pred_log_dir="logs/pred",
    gt_log_dir="logs/reference",
)
print(similarity)  # float in [0, 1], higher is better
```

## License

See [LICENSE](LICENSE).

## Acknowledgements

We would like to acknowledge the following references that have contributed to
the development of this project: [References](./docs/acknowledgements.md).
