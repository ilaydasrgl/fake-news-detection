# Fake News Detection: A Cross-Domain Generalization Study

Do fake news detectors actually learn to spot fake news, or just the style of the sources they were trained on?

This project compares a classical NLP baseline (**TF-IDF + Logistic Regression**) with a fine-tuned transformer (**DistilBERT**). Both are trained on the ISOT dataset, then tested on WELFake, a dataset from completely different sources, to see how well they generalise.

Built for the AI Lab course (Computer Vision, Signal Processing & NLP) at Sapienza University of Rome, 2025/2026.

## Key finding

Both models score very high on their own dataset, but drop sharply on new data:

| Model | ISOT (in-domain) F1 | WELFake (cross-domain) F1 | Drop |
| --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 0.953 | 0.719 | -23.4 pp |
| DistilBERT | 0.987 | 0.812 | -17.5 pp |

The cause is **shortcut learning**: every real article in ISOT comes from Reuters and starts with a tag like "WASHINGTON (Reuters)". The models learn that tag as a signal of "real" instead of learning anything about the content. On data without that tag, performance falls.

Takeaway: single-dataset accuracy (often reported at 98–99% in the literature) overstates how well these systems would work in the real world.

## Method

- **Preprocessing:** title + body concatenated, lowercased, URLs/HTML/special characters removed.
- **Baseline:** TF-IDF (50k features, unigrams + bigrams) with Logistic Regression.
- **Transformer:** `distilbert-base-uncased` fine-tuned for 3 epochs (AdamW, lr 2e-5, batch 16, linear warmup).
- **Evaluation:** stratified 80/10/10 split on ISOT, then the full WELFake dataset as an unseen cross-domain test set. Weighted F1 is the main metric.
- **Attention analysis:** extracts DistilBERT's self-attention from the [CLS] token, averaged over all 6 layers and 12 heads, and plots it as a heatmap over each article's tokens. It also measures how much more attention the token "reuters" gets than an average token, to test the shortcut hypothesis.

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the datasets and place the CSV files in the project folder:
   - **ISOT Fake News Dataset** (University of Victoria): `True.csv` and `Fake.csv`
   - **WELFake Dataset**: `WELFake_Dataset.csv`
3. Run:
   ```
   python fake_news_detector.py
   ```

The script prints metrics, saves confusion matrices, a model comparison chart and attention heatmaps as PNG files, and saves the fine-tuned model to `./model/`.

A GPU is strongly recommended for DistilBERT (we used a free NVIDIA T4 on Google Colab). The baseline runs on CPU in a couple of minutes.

## Limitations

Both datasets are English-language and mostly political news. Results vary slightly across random seeds.

## Authors

Ilayda Esma Asaroglu and Liluz Atan, Sapienza University of Rome
