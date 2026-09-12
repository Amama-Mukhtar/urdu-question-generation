# Urdu Question Generation

Assignment 1, Generative AI. Give it an Urdu sentence with the answer marked as `<ans> ... </ans>`, it writes the question.

From scratch - own tokenizer (SentencePiece), own model (BiLSTM encoder, LSTM decoder, attention). No pretrained stuff.

## Example

```
Input:  ہیوسٹن ، ٹیکساس میں پیدا ہوئی ... اور <ans> 1990 کی دہائی کے آخر میں </ans> R&B گرل گروپ ...
Output: بیونس نے کب مقبولیت حاصل کرنا شروع کی؟
```

## Dataset

[UQA](https://huggingface.co/datasets/uqa/UQA) - Urdu SQuAD 2.0. Also tested on [Wiki-UQA](https://huggingface.co/datasets/uqa/Wiki-UQA), out of domain.

Only the sentence with the answer in it goes in, not the whole paragraph. Too small a model for a full paragraph.

## Model

- Encoder: 2-layer BiLSTM
- Decoder: 2-layer LSTM + Bahdanau attention
- embedding 256, hidden 512, dropout 0.3
- Adam, teacher forcing, cross-entropy (padding ignored)
- greedy decoding and beam search (k=4)

## How to run

```bash
pip install datasets sentencepiece sacrebleu rouge-score torch gradio
```

Open `GenAI_Task1.ipynb` and run all the cells top to bottom. Use a GPU (Colab/Kaggle) - training takes about 1-2 hours. It'll create:
- `data/train.tsv`, `data/valid.tsv`
- `tokenizer/ur_sp.model`, `ur_sp.vocab`
- `best.pt`
- everything in `results/`

Then for the front end:
```bash
python app.py
```

## Files

```
GenAI_Task1.ipynb        the whole notebook, Tasks 1-5
app.py                   Gradio app
best.pt                  best checkpoint (lowest valid loss)
data/train.tsv           75,067 pairs
data/valid.tsv           10,018 pairs
tokenizer/ur_sp.model    SentencePiece, vocab 8000
tokenizer/ur_sp.vocab
results/samples.tsv               200 examples: source, reference, greedy, beam
results/qualitative_10_examples.tsv   5 good + 5 bad, with failure types
results/human_eval_member1.csv    ratings, member 1
results/human_eval_member2.xlsx   ratings, member 2
results/loss_curve.png
results/length_hist.png
results/attention.png
results/frontend.png
```

## Results

### Dataset

|                    | Train   | Valid  | Wiki-UQA |
|--------------------|---------|--------|----------|
| raw rows           | 124,745 | 16,824 | 210      |
| answerable rows    | 83,018  | 11,169 | -        |
| pairs after filter | 75,067  | 10,018 | 177      |
| avg source length  | 32.6 tokens | - | -    |
| avg target length  | 11.9 tokens | - | -    |

### Model

| | |
|---|---|
| encoder/decoder | LSTM |
| layers/emb/hidden | 2 / 256 / 512 |
| vocab | 8,000 |
| params | 33,457,472 |
| optimiser | Adam, lr 1e-3 |
| batch/epochs | 64 / 10 (best at epoch 5) |
| GPU | CUDA |

### Metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | unk% |
|---|---|---|---|---|---|
| UQA valid | greedy | 2.17 | 0.203 | 44.02 | 0.19% |
| UQA valid | beam (k=4) | 4.83 | 0.210 | - | 0.60% |
| Wiki-UQA | greedy | 3.59 | 0.207 | - | 1.03% |
| Wiki-UQA | beam (k=4) | 4.36 | 0.211 | - | 0.87% |

BLEU is lower than the 6-13 the assignment expects. Best checkpoint was epoch 5 of 10, so it's a bit undertrained. Not a bug though - a bug would give near-0.

### Human eval (50 samples, independent ratings)

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 | 8% yes | 84% yes | 20% yes |
| Member 2 | 8% yes | 50% yes | 8% yes |
| Cohen's kappa | 0.18 | 0.32 | 0.35 |

Agreement was low. A lot of these were borderline calls since the checkpoint is weak, so we didn't always land on the same yes/no.

### 5 good / 5 bad examples

| Type | Source (short) | Generated | Problem |
|---|---|---|---|
| good | Normans settled east of Ireland... | نارمنز کس براعظم میں کس علاقے میں آباد ہوئے؟ | - |
| good | computational problem = solvable by computer | کیا طور پر ایک کمپیوٹر کی طرف سے حل کیا جا سکتا ہے؟ | - |
| good | TSP, integer factorization examples | ایک قابل ذکر الگورتھم کیا ہے جو سفر کرنے کے قابل بناتا ہے؟ | - |
| good | complexity theory = what computers can/can't do | کمپیوٹنگ کیا ہے؟ | - |
| good | river Diabolus, castle of Petra | اس دریا کا نام کیا ہے؟ | - |
| bad | Beyoncé rose to fame late 1990s | بیونس نے 1990 میں کون سا گانا حاصل کیا؟ | asked "which song" instead of "when" |
| bad | William II killed Harold II at Hastings | کس بادشاہ بادشاہ نے بادشاہ کو بادشاہ نے شکست دی؟ | repetition |
| bad | Harvey, Norman mercenary, 1050s | نمیبیا میں پہلی بار کب پہنچا؟ | hallucinated "Namibia" |
| bad | Normans joined Turkish forces | کچھوںوں نے کسوں میں کس ادارے میں شمولیت اختیار کی؟ | garbled words |
| bad | Bayeux Tapestry, most famous Norman art | نارمن آرٹ آرٹ اب اب بھی کام کام کرتا ہے؟ | repetition, broken grammar |

Full table in `results/qualitative_10_examples.tsv`.

### Figures

1. `results/loss_curve.png` - train/valid loss per epoch
2. `results/length_hist.png` - source/target length histograms
3. `results/attention.png` - attention heat-map
4. `results/frontend.png` - front end screenshot

## Discussion

**Which question words does it get right most often?**
"کب" (when) mostly comes out fine - the shape "X کب ہوا؟" is usually there even if the date itself is wrong. "کون" (who) is the weak one. Most of the made-up stuff (Namibia, Napoleon, Victoria - none of them in the source) happened on "who" questions. Guessing a name from nothing is harder than copying a pattern, and 75k pairs isn't a lot of names to learn from.

**Where does beam help, where does it hurt?**
BLEU went up with beam on both splits (2.17 to 4.83 on UQA valid, 3.59 to 4.36 on Wiki-UQA). It also fixed the worst repeat loop we found - greedy said "بیونس نے کس بیونس میں بیونس حاصل کی؟", beam said "بیونس نے 1990 میں کون سا گانا حاصل کیا؟", way cleaner. Beam can step out of a loop greedy gets stuck in. But when the model is just confidently wrong, beam doesn't save it - a few beam outputs still repeat a word 3-4 times because every candidate in the beam came from the same bad habit.

**Why is Wiki-UQA worse?**
unk rate is about double on Wiki-UQA (~1% vs ~0.3%) - the tokenizer only saw UQA's training text, so some Wiki-UQA words are just new to it. Both datasets are machine-translated from English but by different translators, so even the same kind of sentence reads differently. A small model trained on one translation picks up some of that translator's habits along with the actual task, so it struggles when the wording shifts even though the task itself hasn't changed.

## Front end

Gradio app (`app.py`). Paste a sentence, type the answer text exactly, get greedy + beam back.

## About the code

Data loading, sentence splitting, writing the tsv files, tokenizer training, and the scoring function are the starter code from the assignment appendix - used as given. Only thing we changed: the field names (`answer`/`answer_start`/`is_impossible`), because the live dataset stores them flat, not as the `answers` dict the manual shows. The model itself (encoder, attention, decoder, training loop, decoding) and the front end are ours.
