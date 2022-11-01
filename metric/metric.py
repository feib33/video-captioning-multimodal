
__author__ = 'tylin'

from transformers import AutoTokenizer

from metric.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from metric.pycocoevalcap.bleu.bleu import Bleu
from metric.pycocoevalcap.meteor.meteor import Meteor
from metric.pycocoevalcap.rouge.rouge import Rouge
from metric.pycocoevalcap.cider.cider import Cider
from dataloader import Msrvtt



def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    return score

def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    return score

def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return score

def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    return score


def eval(gts, res):

    metric = {"bleu": bleu(gts, res), "meteor": meteor(gts, res),
              "rouge": rouge(gts, res), "cider": cider(gts, res)}
    return metric

"""
gts = {"video6871": ['a guy in glasses is sitting down and talking', "a man with brown shirt and spectacles is explaining something",
                     "a young man shaking his hands and talking about some subject", "there is a suit man talking in front of the laptop",
                     "a men in dark brown suit is sitting and talking", "one man talking about something and explaining also wearing white glass",
                     "a woman sits at a desk in front of a laptop and talks", "a person is explaining to us about the gases and its uses  and he further says this was so crazily disruptive that took another 50 years of meetings among scientists  to decide another standard carob 12",
                     "a man cups his lifted hands together while making a point seated in front of rectangular windows decorated with vases and knickknacks"],
       "video0": ["Let's say this is just a ground-truth"]}

res = {'video6871': ['a man is sitting at a desk and talking to the camera'], 'video0': ["Let's say this is a reference from model"]}  # len is only 1
# gts, res = make_coco(gts, res)

eval(gts, res)




tokenizer = AutoTokenizer.from_pretrained("../pretrained_models/bert_tokenizer")

dataset = Msrvtt(feat_path="../data/MSR-VTT/feats/timesformer/howto100m",
                         cap_path="../data/MSR-VTT/annotations/train_val_videodatainfo.json",
                         split="validate", tokenizer=tokenizer)

print(dataset.vid2cap)
"""