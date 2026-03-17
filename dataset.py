import os

VOCAB_SIZE = 128


def tokenize(text):
    return [(ord(char))/255 for char in text]

def detokenize(tokens):
    chars = []
    for token in tokens:
        if len(token) == 1:
            idx = round(token[0] * 255)
        else:
            idx = max(range(len(token)), key=lambda i: token[i])
        idx = max(0, min(255, idx))
        chars.append(chr(idx))
    return chars



def read_dataset(path=None, window=5):
    # Default dataset paths are relative to this module, so the loader works
    # whether the module is imported or run as a script.
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "input_tiny2.txt")

    dataset_final = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            line_array = []
            for i in range(1, len(line)):
                inp_text = line[max(0, i - window):i]
                inp = tokenize(inp_text)
                char_idx = ord(line[i])
                if char_idx >= VOCAB_SIZE:
                    continue
                result = [0.0] * VOCAB_SIZE
                result[char_idx] = 1.0
                line_array.append([inp, result])

            if line_array:
                dataset_final.append(line_array)

    return dataset_final

if __name__ == "__main__":
    read_dataset()
    # print(read_dataset())