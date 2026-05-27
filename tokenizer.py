import numpy as np

# After training your tokenizer, tokenize the entire corpus
all_ids = []
with open("family_guy_corpus.txt") as f:
    for line in f:
        encoded = tokenizer.encode(line.strip())
        all_ids.extend(encoded.ids)

# Save as numpy — compact, fast to load
all_ids = np.array(all_ids, dtype=np.uint16)  # uint16 supports vocab up to 65535
np.save("token_ids.npy", all_ids)

print(f"Total tokens: {len(all_ids):,}")
print(f"File size: {all_ids.nbytes / 1e6:.1f} MB")