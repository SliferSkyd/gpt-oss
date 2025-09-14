import tiktoken
enc = tiktoken.get_encoding("o200k_harmony")
with open("/nfs/gpu_trainee/getp02/gpt-oss-copy-2/data/input.txt", "r") as f:
    inputs = f.readlines()[1:]
inputs = inputs[:4097]
lens = [len(enc.encode(x)) for x in inputs]

# analyze the distribution of lens
import matplotlib.pyplot as plt

plt.hist(lens, bins=50)
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.title("Distribution of Token Lengths")
plt.show()
plt.savefig("token_length_distribution.png")