"""
app.py - Standalone launcher for the Family Guy GPT text generator.

Run:
    python app.py                       # local UI at http://localhost:7860
    python app.py --share               # also create a public *.gradio.live link
    python app.py --weights my.pt --tokenizer tok.json

Requires a `model.py` in the same folder that defines your GPT and GPTConfig
classes (move them out of the notebook unchanged). The tokenizer must have been
saved earlier with `tokenizer.save("tokenizer.json")`.
"""

import argparse

import torch
import gradio as gr
from tokenizers import Tokenizer, decoders

from model import GPT, GPTConfig  # <-- your architecture, copied from the notebook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="gpt_model.pt", help="Path to trained .pt weights")
    p.add_argument("--tokenizer", default="my_tokenizer.json", help="Path to saved tokenizer")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def load(weights_path, tokenizer_path):
    """Load tokenizer + model once at startup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.ByteLevel()  # ensure clean decode even if the saved file predates it

    model = GPT(GPTConfig()).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Sanity check: tokenizer vocab must match the model's vocab size.
    print(f"Loaded model on {device} | tokenizer vocab: {tokenizer.get_vocab_size()}")
    return model, tokenizer, device


def make_generate_fn(model, tokenizer, device):
    def generate_text(prompt, max_new_tokens, temperature, top_k):
        if not prompt.strip():
            return "Enter a prompt first."
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
            )
        return tokenizer.decode(out[0].tolist())

    return generate_text


def build_ui(generate_fn):
    return gr.Interface(
        fn=generate_fn,
        inputs=[
            gr.Textbox(label="Starter prompt", value="Once upon a time", lines=2),
            gr.Slider(10, 500, value=200, step=10, label="Max new tokens"),
            gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature"),
            gr.Slider(1, 100, value=50, step=1, label="Top-k"),
        ],
        outputs=gr.Textbox(label="Generated text", lines=12),
        title="Family Guy GPT",
        description="Enter a prompt and tune the sampling controls.",
    )


def main():
    args = parse_args()
    model, tokenizer, device = load(args.weights, args.tokenizer)
    ui = build_ui(make_generate_fn(model, tokenizer, device))
    ui.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()