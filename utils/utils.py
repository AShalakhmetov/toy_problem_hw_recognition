def text_to_labels(text, letters):
    return list(map(lambda x: letters.index(x) + 1, text))

def labels_to_text(labels, letters):
    return ''.join(letters[int(x)-1] if x != 0 else '-' for x in labels).split('-')

def encode(text, letters):
    return text_to_labels(text=text, letters=letters)

def decode(labels, letters):
    chunks = labels_to_text(labels=labels, letters=letters)
    return ''.join(
        [char for chunk in chunks for idx, char in enumerate(chunk) if char != chunk[idx - 1] or len(chunk) == 1])
