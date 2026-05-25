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

# 3. Encode text → token IDs
output = tokenizer.encode("Hello, world!")
print(output.ids)      # [1824, 16, 829, 264]
print(output.tokens)   # ['Hello', ',', 'Ġworld', '!']

# 4. Save for later
tokenizer.save("my_tokenizer.json")