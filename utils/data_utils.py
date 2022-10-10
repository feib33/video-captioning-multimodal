


def tokenize(tokenizer, sen_batch, max_length):
    return tokenizer(sen_batch, padding='max_length', max_length=max_length, return_tensors='pt')