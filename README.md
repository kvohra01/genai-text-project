# Family Guy Small Language Model — GUI

A small, from-scratch decoder-only transformer (GPT-style, ~4 layers / 4 heads /
128-dim) trained on Family Guy episode transcripts. This README covers how to run
the interactive GUI that samples dialogue from the trained model.

## Project files

```
.
├── model.py            # GPT / GPTConfig definition (the transformer)
├── app.py              # Gradio GUI — the file you run for the demo
├── tokenizer.json      # trained byte-level BPE tokenizer (32k vocab)
├── ckpt.pt             # trained model checkpoint
├── requirements.txt
└── README.md
```

`tokenizer.json` and `ckpt.pt` are produced by your training notebook. If yours
are named differently or live in another folder, see **Configuration** below.

## Setup

Use a virtual environment so dependencies stay isolated.

```bash
# create + activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the GUI

```bash
python app.py
```

Gradio prints a local URL (e.g. `http://127.0.0.1:7860`). Open it in a browser.
You'll see:

- **Prompt** — the seed text the model continues (e.g. *"Peter walks into the kitchen and says"*).
- **Max new tokens** — how many tokens to generate.
- **Temperature** — randomness. Low (~0.4) = safe/repetitive, high (~1.2) = chaotic.
- **Top-k** — restricts sampling to the k most likely next tokens.

Type a prompt, adjust the sliders, and submit to get a continuation.

