#  Mental Health Monitoring on Social Media

A **DistilBERT-based text classifier** fine-tuned to detect and categorize mental health conditions from Reddit posts. Achieves **95% validation accuracy** and **83% test accuracy** across 7 mental health categories.

---

##  Overview

Mental health discussions on social media offer a unique, real-time window into people's psychological states. This project leverages transformer-based NLP to automatically classify Reddit posts into mental health categories — enabling potential early-warning systems, population health insights, and research applications.

**Key highlight:** Fine-tuned `distilbert-base-uncased` on a multi-class Reddit mental health dataset, with full training pipeline, evaluation metrics, and a Gradio inference demo.

---

##  Dataset

- **Source:** Reddit posts from mental health subreddits (e.g., r/depression, r/anxiety, r/SuicideWatch)
- **Classes (7):**
  - `Normal`
  - `Depression`
  - `Anxiety`
  - `Bipolar`
  - `PTSD`
  - `Personality Disorder`
  - `Suicidal Ideation`
- **Split:** ~80% train / 10% validation / 10% test

> **Note:** Dataset sourced from publicly available Kaggle compilations of Reddit mental health posts. Not for clinical use.

---

##  Model Architecture

| Component | Detail |
|---|---|
| Base Model | `distilbert-base-uncased` (HuggingFace) |
| Task | Multi-class text classification |
| Classifier Head | Linear(768 → 7) |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Scheduler | Linear warmup with decay |

DistilBERT was chosen for its balance of performance and efficiency — 40% smaller and 60% faster than BERT-base while retaining ~97% of its language understanding capability.

---

##  Results

| Metric | Score |
|---|---|
| Validation Accuracy | **95%** |
| Test Accuracy | **83%** |

The gap between validation and test accuracy reflects the challenge of generalizing across diverse Reddit writing styles and the inherent class imbalance in mental health datasets.

---

##  Getting Started

### Prerequisites

```bash
Python >= 3.8
torch >= 1.12
transformers >= 4.20
```

### Installation

```bash
git clone https://github.com/mohithnovoct/Mental-Health-Monitoring-on-Social-Media.git
cd Mental-Health-Monitoring-on-Social-Media
pip install -r requirements.txt
```

### Training

```bash
python train.py \
  --model distilbert-base-uncased \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5
```

### Inference

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./saved_model",
    tokenizer="distilbert-base-uncased"
)

result = classifier("I've been feeling really low lately and can't get out of bed.")
print(result)
# [{'label': 'Depression', 'score': 0.91}]
```

---

##  Project Structure

```
Mental-Health-Monitoring-on-Social-Media/
│
├── data/
│   ├── raw/                    # Raw Reddit dataset
│   └── processed/              # Tokenized & split data
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis
│
├── src/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   ├── dataset.py              # PyTorch Dataset class
│   └── model.py                # Model definition
│
├── saved_model/                # Fine-tuned model weights
├── requirements.txt
└── README.md
```

---

##  Tech Stack

- **Model:** HuggingFace Transformers (`distilbert-base-uncased`)
- **Training:** PyTorch
- **Evaluation:** scikit-learn (classification report, confusion matrix)
- **Demo:** Gradio
- **Deployment:** Hugging Face Spaces

---

##  Ethical Considerations

This project handles sensitive mental health data and must be used responsibly:

- **Not a clinical tool.** Predictions should never replace professional mental health assessment.
- **Privacy:** All data used is anonymized/publicly sourced. No personal identification is performed.
- **Bias:** The model may reflect biases present in Reddit's user demographics. Performance may vary across communities.
- **Misuse risk:** Avoid deploying this system in ways that could stigmatize or surveil individuals without consent.

---

##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> *This project is for research and educational purposes only. If you or someone you know is struggling with mental health, please reach out to a qualified professional or contact a crisis helpline.*
