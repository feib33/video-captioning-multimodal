import torch.nn as nn



def tokenize(tokenizer, sen_batch, max_length):
    return tokenizer(sen_batch, padding='max_length', max_length=max_length, return_tensors='pt')


def create_mask(token):
    """
    Since each position could only see previous positions,
    the latter positions would be "-infinite".
    (N, T, E) -> (T, T)
    :param token: Tensor
    :return: ByteTensor
    """
    return nn.Transformer.generate_square_subsequent_mask(token.shape[1])


def create_key_padding_mask(token):
    """
    Turn FloatTensor or ByteTensor to BoolTensor
    :param token: FloatTensor or ByteTensor
    :return: BoolTensor
    """
    return token == 0
