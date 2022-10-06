import torch
import torch.nn as nn
import math
from transformers import BertModel, AutoTokenizer

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


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx, weight_is_pretrained):
        super(TextEmbedding, self).__init__()
        if weight_is_pretrained is False:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        else:
            weight = get_bert_embedding_weight()
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)  # TBD, if the weight need to be updated during the training, set freeze to False

    def forward(self, word: torch.LongTensor):
        return self.embedding(word)


class FeatureEmbedding(nn.Module):
    def __init__(self, feat_size, d_model):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Linear(feat_size, d_model)

    def forward(self, feat):
        return self.embedding(feat)


class PositionalEncoding(nn.Module):
    """
        This class is borrowed from original pytorch tutorials.
        web address: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
            Args:
                x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MMT(nn.Module):
    def __init__(self, feat_size, phase: str):
        super(MMT, self).__init__()
        self.transformer = nn.Transformer(
            d_model=768,  # input shape
            dim_feedforward=2048,  # dimension of feedforward network model
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            batch_first=True,  # (batch, seq, feature)
            activation="gelu"  # perform best in Transformer and used in Bert and GPT-2
        )

        self.phase = phase
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrained_models/bert_tokenizer/")
        vocab_size = self.tokenizer.vocab_size
        padding_idx = self.tokenizer.convert_tokens_to_ids("[PAD]")

        # Preprocessing stage before processing into the Transformer
        self.enc_embedding = FeatureEmbedding(feat_size, d_model=768)
        self.dec_embedding = TextEmbedding(vocab_size, d_model=768, padding_idx=padding_idx, weight_is_pretrained=True)
        self.pos_encoding = PositionalEncoding(d_model=768, dropout=0.5)

        self.generator = nn.Linear(768, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_pos = self.pos_encoding(self.enc_embedding(src))

        if self.phase == "Prediction":
            tgt_pos = self.pos_encoding(self.dec_embedding(tgt))
            out = self.transformer(src=src_pos, tgt=tgt_pos,
                                   src_mask=src_mask, tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            logit = self.generator(out)
            return logit
        elif self.phase == "Training":
            enc_output = self.transformer.encoder(src, src_mask, src_key_padding_mask=src_padding_mask)
            tgts, tgt_masks, tgt_padding_masks = tgt, tgt_mask, tgt_padding_mask
            logits = []

            for tgt, tgt_mask, tgt_padding_mask in zip([tgts, tgt_masks, tgt_padding_masks]):
                tgt_pos = self.pos_encoding(self.enc_embedding(tgt))
                out = self.transformer.decoder(tgt_pos, enc_output, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_padding_mask,
                                               memory_key_padding_mask=src_padding_mask)

                logit = self.generator(out)
                logits.append(logit)
            return logits

    def encoder(self, src: torch.Tensor):
        src_d = self.enc_embedding(src)
        src_pos = self.pos_encoding(src_d)
        return self.transformer.encoder(src_pos)

    def decoder(self, tgt: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor):
        tgt_d = self.dec_embedding(tgt)
        tgt_pos = self.pos_encoding(tgt_d)
        return self.transformer.decoder(tgt_pos, enc_output, tgt_mask)

