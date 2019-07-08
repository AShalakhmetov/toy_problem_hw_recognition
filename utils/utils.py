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

def decode_target(labels, letters):
    chunks = labels_to_text(labels=labels, letters=letters)
    return ''.join(chunks)

def custom_accuracy_score(output, decoded_target, letters):
    logits = output.softmax(2).argmax(2)
    logits = logits.squeeze(1).numpy()

    text = decode(logits[0], letters)
    return 1 if text == decoded_target else 0