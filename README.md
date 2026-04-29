---
title: surgical-postop-diagnosis-prediction
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏥 Surgical Post-op Diagnosis Prediction</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=Post-operative+ICD-10+Classification;RAG+%2B+Llama-3+Operative+Diagnosis+Generation;Bio-ClinicalBERT+%C2%B7+FAISS+%C2%B7+LightGBM+%C2%B7+XGBoost" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient+Boosting-4f46e5?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ClinicalBERT-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
[![Ollama](https://img.shields.io/badge/Ollama-Llama--3-3b82f6?style=for-the-badge)](https://ollama.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🏥 Surgical Post-op Diagnosis Prediction** — A multi-task framework that predicts post-operative diagnoses from pre-operative surgical data. Task A classifies ICD-10 codes using Bio-ClinicalBERT embeddings with LightGBM/XGBoost. Task B generates free-text operative diagnoses via FAISS-based RAG retrieval augmented with Llama-3, evaluated with an ablation study across feature combinations.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Pipeline Stages](#-pipeline-stages)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Outputs & Artifacts](#-outputs--artifacts)
- [Reproducibility](#-reproducibility)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🏷️ <b>ICD-10 Classification (Task A)</b></td>
    <td>Multi-class prediction of most-responsible post-operative diagnosis code using Bio-ClinicalBERT + LightGBM/XGBoost</td>
  </tr>
  <tr>
    <td>📝 <b>Operative Text Generation (Task B)</b></td>
    <td>Free-text operative diagnosis generation via RAG + Llama-3 (Ollama), conditioned on pre-op features and retrieved similar cases</td>
  </tr>
  <tr>
    <td>🔍 <b>FAISS Retrieval-Augmented Generation</b></td>
    <td>Fold-isolated FAISS indexes for k-NN retrieval at inference — no cross-fold leakage</td>
  </tr>
  <tr>
    <td>🧪 <b>Ablation Study (Task P5)</b></td>
    <td>Systematic comparison across text-only, tabular-only, TF-IDF, and RAG-K variants</td>
  </tr>
  <tr>
    <td>📊 <b>Comprehensive Metrics</b></td>
    <td>Top-1/Top-3/Top-5 accuracy, macro F1, AUC-ROC for Task A · BLEU, ROUGE for Task B</td>
  </tr>
  <tr>
    <td>📄 <b>Publication-Ready Outputs</b></td>
    <td>LaTeX tables and PDF figures written directly to <code>results/</code> and <code>overleaf/</code></td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│          Surgical Post-op Diagnosis Prediction                     │
│                                                                    │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │  Pre-op    │───▶│ Bio-ClinicalBERT│───▶│  Task A          │  │
│  │  Features  │    │  (768-d)        │    │  ICD-10 Code     │  │
│  └────────────┘    └────────┬────────┘    └──────────────────┘  │
│                             │                                      │
│                    ┌────────▼────────┐    ┌──────────────────┐   │
│                    │  FAISS Index    │───▶│  Task B          │   │
│                    │  (k=5 RAG)      │    │  Operative Dx    │   │
│                    └────────┬────────┘    │  (Llama-3 + RAG) │   │
│                             │             └──────────────────┘   │
│                    ┌────────▼────────┐                            │
│                    │  Task P5        │                            │
│                    │  Ablation Study │                            │
│                    └─────────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- [Ollama](https://ollama.com/) with `llama3` model (for Task B)
- `data/casetime.csv` (source dataset — not included in repo)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/surgical-postop-diagnosis-prediction.git
cd surgical-postop-diagnosis-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scipy scikit-learn lightgbm xgboost torch \
            transformers faiss-cpu matplotlib ollama
```

### Run the Pipelines

```bash
# Full paper pipeline (Stages P1–P5)
python code/paper_pipeline.py

# Or run individual stages
python code/run_p4a.py      # Task A only (ICD-10 classification)
python code/run_all.py      # All tasks end-to-end

# Generate publication figures and LaTeX tables
python code/figures.py
```

> **Note:** Task B (Llama-3 generation) requires Ollama running locally with `llama3` pulled: `ollama pull llama3`

---

## 📊 Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| P1 Data Preparation | Merge post-op targets, encode ICD-10 labels, persist PaperClean table | `data/paper_data.db` |
| P2 BERT Encoding | Bio-ClinicalBERT (768-d) encoding of `scheduled_procedure` text | `data/paper_bert_cache.npy` |
| P3 Feature Engineering | Tabular feature matrices + fold-isolated FAISS indexes (k=5) | `data/paper_faiss/` |
| P4A ICD-10 Classification | LightGBM + Logistic Regression, top-k accuracy, macro F1 | `data/outputs/p4a_results.csv` |
| P4B Operative Dx Generation | RAG retrieval + Llama-3 (Ollama) free-text generation | `data/outputs/p4b_results.csv` |
| P5 Ablation Study | Text-only / tabular-only / TF-IDF / RAG-K variants | `data/outputs/p5_ablation.csv` |

---

## 🧠 ML Models

```python
# Task A — ICD-10 Multi-class Classification
classifiers = {
    "LightGBM": {
        "objective":        "multiclass",
        "metric":           "multi_logloss",
        "num_leaves":       31,
        "learning_rate":    0.1,
        "feature_fraction": 0.7,
        "rounds":           200,
        "early_stopping":   20,
    },
    "LogisticRegression": {
        "C":         1.0,
        "max_iter":  1000,
        "solver":    "lbfgs",
        "multi_class": "auto",
    },
}

# Task B — Operative Diagnosis Text Generation
generator = {
    "model":    "llama3:latest (Ollama)",
    "strategy": "RAG k=5 nearest neighbours (FAISS cosine)",
    "prompt":   "Retrieved similar cases + pre-op features → operative_dx",
}
```

---

## 📁 Project Structure

```
surgical-postop-diagnosis-prediction/
│
├── 📂 code/
│   ├── 📄 paper_pipeline.py     # Main pipeline (Stages P1–P5)
│   ├── 📄 pipeline.py           # Base duration pipeline (Stage 01 prerequisite)
│   ├── 📄 run_p4a.py            # Task A runner (ICD-10 classification only)
│   ├── 📄 run_all.py            # Full end-to-end runner
│   ├── 📄 figures.py            # Figure and LaTeX table generation
│   └── 📄 flowchart.py          # Pipeline flowchart generation
│
├── 📂 data/
│   ├── 📂 raw/                  # Source data — read only, never modify
│   ├── 📂 processed/            # BERT cache, FAISS indexes, label encoders
│   └── 📂 outputs/              # Per-task result CSVs (p4a, p4b, p5_ablation)
│
├── 📂 results/                  # Logs, PDF figures, and LaTeX tables
│
├── 📂 overleaf/                 # LaTeX manuscript
│   └── 📄 *.tex                 # Paper sections
│
├── 📂 flowchart/                # Pipeline flowchart assets
└── 📄 sync.ps1                  # Git sync utility
```

---

## 📦 Outputs & Artifacts

- **Databases & Caches** (local only, not in repo)
  - `data/paper_data.db` — cleaned post-op dataset with ICD-10 labels and fold indices
  - `data/paper_bert_cache.npy` — Bio-ClinicalBERT embeddings (768-d per case)
  - `data/paper_faiss/fold{0–4}_index.faiss` — fold-isolated FAISS indexes for RAG
- **Result CSVs**
  - `data/outputs/p4a_results.csv` — top-1/3/5 accuracy and macro F1 per fold
  - `data/outputs/p4b_results.csv` — generated operative diagnoses with BLEU/ROUGE
  - `data/outputs/p5_ablation.csv` — ablation comparison across feature strategies
- **Figures** — model comparison and ablation PDFs in `results/` and `overleaf/`

---

## 🔁 Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) across all stages.
- FAISS indexes are built per fold from training data only — no test-set leakage.
- BERT embeddings are cached after the first run; subsequent runs reuse the cache.
- ICD-10 classes with fewer than 50 cases are grouped as `OTHER` to stabilise evaluation.
- Stages P1–P3 auto-skip when outputs are already present.

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional medical advice of any kind. All datasets used are subject to institutional data-use agreements — no patient-identifiable information is included in this repository. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/surgical-postop-diagnosis-prediction?style=social)](https://github.com/mnoorchenar/surgical-postop-diagnosis-prediction)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/surgical-postop-diagnosis-prediction?style=social)](https://github.com/mnoorchenar/surgical-postop-diagnosis-prediction/fork)

<sub>This project is developed purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
