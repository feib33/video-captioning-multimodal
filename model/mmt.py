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


def create_mask(src, tgt, padding_idx):
    src_seq_len = src.shape[0] #(seq, feat)
    tgt_seq_len = len(tgt)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device="cpu").type(torch.bool)

    src_padding_mask = (src == padding_idx)
    tgt_padding_mask = [target == padding_idx for target in tgt]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def tokenize(tokenizer, sen_batch, max_length):
    return tokenizer(sen_batch, padding='max_length', max_length=max_length, return_tensors='pt')


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
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pos_table = numpy.array([
            [pos / numpy.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else numpy.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = numpy.sin(pos_table[1:, 0::2])  # when even
        pos_table[1:, 1::2] = numpy.cos(pos_table[1:, 1::2])  # when odd
        self.pos_table = torch.FloatTensor(pos_table) # enc_inputs: [seq_len, d_model] TBD FloatTensor().to("cuda")

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_inputs):  # enc_inputs (B, S or T, d_model)
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]  # broadcasting is used
        return self.dropout(enc_inputs)  # TBD enc_inputs.to('cuda')


class MMT(nn.Module):
    def __init__(self, feat_size, mode: str, d_model: int):
        super(MMT, self).__init__()
        """
        self.transformer = nn.Transformer(
            d_model=768,  # input shape
            dim_feedforward=2048,  # dimension of feedforward network model
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            activation="gelu"  # perform best in Transformer and used in Bert and GPT-2
        )
        """
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1,
                                       activation="gelu", batch_first=True),
            num_layers=4
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8, dropout=0.1,
                                       activation="gelu", batch_first=True),
            num_layers=4
        )

        self.mode = mode
        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrained_models/bert_tokenizer/")
        vocab_size = self.tokenizer.vocab_size
        padding_idx = self.tokenizer.convert_tokens_to_ids("[PAD]")
        # Preprocessing stage before processing into the Transformer
        self.enc_embedding = FeatureEmbedding(feat_size, d_model=768)
        self.dec_embedding= TextEmbedding(vocab_size, d_model=768, padding_idx=padding_idx,
                                          weight_is_pretrained=True)
        self.pos_encoding = PositionalEncoding(d_model=768, dropout=0.5)

        self.generator = nn.Linear(768, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_key_padding_masks: torch.Tensor):
        # src: (N,  1, S, feat_size) -----Embedding-----> (N, 1, S, d_model)
        # tgt: (N, T)  -----Embedding-----> (N, T, d_model)
        # tgt_mask: (N*num_heads, T, T)
        # src_key_padding_mask: (N, S)
        # tgt_key_padding_mask: (N, T)

        # Encoder part
        #src_key_padding_mask = MMT.create_key_padding_mask(src)
        #print("src_key_padding_mask", src_key_padding_mask)
        src = self.enc_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        memory = self.encoder(src, mask=None)

        # Decoder part
        tgt_mask = MMT.create_mask(tgt)
        print("tgt_mask.shape", tgt_mask.shape)
        print("tgt_padding_mask.shape", tgt_key_padding_masks.shape)

        tgt = self.dec_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        print("tgt.shape", tgt.shape)
        logit = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_masks)
        return logit



        # Since each video contains 20 captions, video and each caption will be proceeded into encoder by looping

        #return logits



    @staticmethod
    def create_mask(token):
        return nn.Transformer.generate_square_subsequent_mask(token.shape[1])  # token.shape (B, T)

    @staticmethod
    def create_key_padding_mask(token):
        return token == 0  # TBD assume that batch size is 1, need to be changed



        """
        src_pos = self.pos_encoding(self.enc_embedding(src))
        if self.mode == "Inference":
            tgt_pos = self.pos_encoding(self.dec_embedding(tgt, self.tokenizer))
            out = self.transformer(src=src_pos, tgt=tgt_pos,
                                   src_mask=src_mask, tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            logit = self.generator(out)
            return logit

        elif self.mode in ["Train", "Validation"]:
            tgt_pos = self.pos_encoding(self.dec_embedding(tgt, self.tokenizer))
            enc_output = self.transformer.encoder(src_pos, src_mask, src_key_padding_mask=src_padding_mask)
            tgts, tgt_masks, tgt_padding_masks = tgt_pos, tgt_mask, tgt_padding_mask
            print(tgts.shape)
            print(tgt_masks.shape)
            print(tgt_padding_masks.shape)

            logits = []
            # since we have 20 captions for each video, each caption will be inputted to decoder (totally 20 rounds)
            for tgt, tgt_mask, tgt_padding_mask in zip([tgts, tgt_masks, tgt_padding_masks]):
                tgt_pos = self.pos_encoding(self.dec_embedding(tgt, self.tokenizer))
                out = self.transformer.decoder(tgt_pos, enc_output, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_padding_mask,
                                               memory_key_padding_mask=src_padding_mask)

                logit = self.generator(out)
                logits.append(logit)
            return logits
        else:
            raise ValueError("The model's mode is not set up")
        """
    """
    def encoder(self, src: torch.Tensor):
        src_d = self.enc_embedding(src)
        src_pos = self.pos_encoding(src_d)
        return self.transformer.encoder(src_pos)

    def decoder(self, tgt: str, enc_output: torch.Tensor, tgt_mask: torch.Tensor):
        tgt_d = self.dec_embedding(tgt, self.tokenizer)
        tgt_pos = self.pos_encoding(tgt_d)
        return self.transformer.decoder(tgt_pos, enc_output, tgt_mask)
    """



model = MMT(feat_size=2048, mode="Train", d_model=768)
# src: [T, C]
src = numpy.load("../data/sample/resnet152_fps3/video0.npy")
src = torch.tensor(src, dtype=torch.float)

# tgt: [S, E]
tgt = "a car is shown"
tokenizer = AutoTokenizer.from_pretrained("../pretrained_models/bert_tokenizer")


# at here, since this sample is unbatched, this should be (T) instead of (1, T)
tgt_dict = tokenize(tokenizer, tgt, max_length=8)


tgt_id = tgt_dict["input_ids"][0]
print(tgt_id)
tgt_key_padding_mask = tgt_dict["attention_mask"][0]
tgt_key_padding_mask = (tgt_key_padding_mask == 0)
print(tgt_key_padding_mask)

src = src.unsqueeze(0) # Delete make a batch size
tgt_id = tgt_id.unsqueeze(0) # Delete make a batch size
tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(0) # Delete make a batch size

logit = model(src, tgt_id, tgt_key_padding_mask)
print("logit:", logit)

# out: [B, S, vocab_size]
#logit = model(src, tgt, None, tgt_mask, src_padding_mask, tgt_padding_mask)
#print(logit)



# TODO Make use of congfig.yaml at mmt.py
# TODO clean up the code, such as move certain functions to '/utils'
# TODO dataloader.py (dataset: (video1.npy, caption1-string))
# TODO train.py
# TODO greedy algorithm
# TODO Inference code
# TODO The inference part in forward of model is not completed
# TODO early_stop with loss or metrics
# TODO log
# TODO xtensorboard