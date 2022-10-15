import torch
import torch.nn as nn
import math
from transformers import BertModel, AutoTokenizer
import numpy

"""
    This is the original transformer from torch.nn.Transformer
    but the embedding layer and positional encoding layer are modified
    which:
        Positional Encoding layer is borrowed from pytorch -> Transformer -> PositionalEncoding class
        
        Embedding layer at encoder side:
            nn.Linear(): feature size -> d_model size ( d_model: dimension used in the Transformer )
        
        Embedding layer at decoder side:
            nn.Embedding(): text (e.g. "see you") -> token (e.g. ["see", "you"]) -> 
                            -> id (e.g. [2349, 3255]) -> tensor (e.g. [torch.LongTensor([1234.000, 3423.343])]) 
                            -> d_model size
            
                where:
                    pretrained Bert from huggingface is used for nn.Embedding.
                    
                    Tokenizer from bert used to tokenize "text" to "tokens";
                    then, "tokens" are converted to "ids" by Tokenizer.convert_tokens_to_ids();
                    subsequently, "ids" are turned to "Tensors" by torch.LongTensor();
                    finally, "Tensors" are fitted into d_model size by nn.Embedding().
"""


def get_bert_embedding_weight():
    bert = BertModel.from_pretrained("bert-base-uncased")
    for k, v in bert.named_parameters():
        if k == "embeddings.word_embeddings.weight":
            return v


"""
    FUNCTION: generate_square_subsequent_mask() and create_mask() 
    is borrowed from official pytorch tutorial and partial modified
    and webpage is https://pytorch.org/tutorials/beginner/translation_transformer.html
"""






class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx, weight_is_pretrained):
        super(TextEmbedding, self).__init__()
        if weight_is_pretrained is False:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        else:
            weight = get_bert_embedding_weight()
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)  # TBD, if the weight need to be updated during the training, set freeze to False

    def forward(self, sen_ids: torch.Tensor):
        embedded = self.embedding(sen_ids)
        return embedded


class FeatureEmbedding(nn.Module):
    def __init__(self, feat_size, d_model):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Linear(feat_size, d_model)

    def forward(self, feat):
        return self.embedding(feat)


class PositionalEncoding(nn.Module):
    """
        This class is borrowed from @wenjtop
        Github: https://github.com/wenjtop/transformer/blob/main/transformer.py
    """
    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pos_table = numpy.array([
            [pos / numpy.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else numpy.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = numpy.sin(pos_table[1:, 0::2])  # when even
        pos_table[1:, 1::2] = numpy.cos(pos_table[1:, 1::2])  # when odd
        self.pos_table = torch.FloatTensor(pos_table).to(device) # enc_inputs: [seq_len, d_model] TBD FloatTensor().to("cuda")

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_inputs):  # enc_inputs (B, S or T, d_model)
        enc_inputs += self.pos_table[:enc_inputs.size(1), :] # broadcasting is used
        return self.dropout(enc_inputs)  # TBD enc_inputs.to('cuda')


class MMT(nn.Module):
    def __init__(self, cfg, mode, device):
        super(MMT, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.MODEL.ENCODER.D_MODEL, nhead=cfg.MODEL.ENCODER.NHEAD,
                                       dropout=cfg.MODEL.ENCODER.DROPOUT, activation=cfg.MODEL.ENCODER.ACTIVATION,
                                       batch_first=cfg.MODEL.ENCODER.BATCH_FIRST),
            num_layers=cfg.MODEL.ENCODER.NUM_LAYERS
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=cfg.MODEL.DECODER.D_MODEL, nhead=cfg.MODEL.DECODER.NHEAD,
                                       dropout=cfg.MODEL.DECODER.DROPOUT, activation=cfg.MODEL.DECODER.ACTIVATION,
                                       batch_first=cfg.MODEL.DECODER.BATCH_FIRST),
            num_layers=cfg.MODEL.DECODER.NUM_LAYERS
        )

        self.mode = mode
        self.device = device
        self.d_model = cfg.MODEL.D_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.TOKENIZER) # TBD the path should be changed
        vocab_size = self.tokenizer.vocab_size
        padding_idx = self.tokenizer.convert_tokens_to_ids("[PAD]")
        # Preprocessing stage before processing into the Transformer
        self.enc_embedding = FeatureEmbedding(cfg.MODEL.FEAT_SIZE, d_model=cfg.MODEL.D_MODEL)
        self.dec_embedding= TextEmbedding(vocab_size, d_model=cfg.MODEL.D_MODEL, padding_idx=padding_idx,
                                          weight_is_pretrained=True)
        self.pos_encoding = PositionalEncoding(d_model=cfg.MODEL.D_MODEL, dropout=cfg.MODEL.DROPOUT, device=self.device)

        self.generator = nn.Linear(cfg.MODEL.D_MODEL, vocab_size)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor,
                tgt: torch.Tensor, tgt_mask: torch.Tensor, tgt_key_padding_masks: torch.Tensor):
        # src: (N, S, feat_size) -----Embedding-----> (N, S, d_model)
        # tgt: (N, T, vocab_size)  -----Embedding-----> (N, T, d_model)
        # src_mask: None
        # tgt_mask: (N*num_heads, T, T)
        # src_key_padding_mask: (N, S)
        # tgt_key_padding_mask: (N, T)

        # Encoder part
        src = self.enc_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        memory = self.encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)

        # Decoder part
        if self.mode in ["train", "validate"]:
            tgt = self.dec_embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)
            logit = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_masks)
            logit = self.generator(logit)
            return logit
        elif self.mode is "test":
            print("Test mode is not constructed yet")
        else:
            raise ValueError("The mode of model must be 'train', 'validation' or 'test'!")

    def change_mode(self, mode):
        self.mode = mode



# TODO greedy algorithm
# TODO Inference code
# TODO The inference part in forward of model is not completed
# TODO early_stop with loss or metrics
# TODO log
# TODO xtensorboard