with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

print("Number of characters:", len(text))

chars = sorted(list(set(text)))

print("Vocabulary size:", len(chars))
print(chars)