from data_utils import CharacterData, DEFAULT_DATA_PATH


def main() -> None:
    data = CharacterData.from_file(DEFAULT_DATA_PATH)
    vocabulary = data.vocabulary

    print("Number of characters:", data.num_characters)
    print("Vocabulary size:", vocabulary.size)
    print(list(vocabulary.chars))

    example = "hello"
    encoded = vocabulary.encode(example)
    print("Encoded example:", encoded)
    print("Decoded example:", vocabulary.decode(encoded))
    print("Training tokens:", data.train_data.shape)
    print("Validation tokens:", data.val_data.shape)
    print("First 100 token IDs:", data.train_data[:100])


if __name__ == "__main__":
    main()
