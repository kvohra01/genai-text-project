import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 2. Train on your corpus
trainer = trainers.BpeTrainer(
    vocab_size=32000,          # typical range: 8k–64k
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    min_frequency=2,
)
tokenizer.train(files=["family_guy_corpus.txt"], trainer=trainer)

# tokenize the entire corpus
all_ids = []
with open("family_guy_corpus.txt", encoding="utf-8") as f:
    for line in f:
        encoded = tokenizer.encode(line.strip())
        all_ids.extend(encoded.ids)

# Save as numpy — compact, fast to load
all_ids = np.array(all_ids, dtype=np.uint16)  # uint16 supports vocab up to 65535
np.save("token_ids.npy", all_ids)

print(f"Total tokens: {len(all_ids):,}")
print(f"File size: {all_ids.nbytes / 1e6:.1f} MB")