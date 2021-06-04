import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
from torch import cuda
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import model
import os
import pickle
import scipy.stats
import gensim
import argparse


def eval_bias_analogy(emb, w2v):
    sembias_filename = './SemBias/Sembias_Kor' if (emb == 'glove_Kor') else './SemBias/SemBias'
    bias_analogy_f = open(sembias_filename)
    definition_num = 0
    none_num = 0
    stereotype_num = 0
    total_num = 0

    sub_definition_num = 0
    sub_none_num = 0
    sub_stereotype_num = 0
    sub_size = 40
    sub_start = -(sub_size - sum(1 for line in open(sembias_filename)))

    
    gender_v =  w2v['남성'] - w2v['여성'] if (emb == 'glove_Kor') else w2v['he']-w2v['she']
    for sub_idx, l in enumerate(bias_analogy_f):
        l = l.strip().split()
        max_score = -100
        for i, word_pair in enumerate(l):
            word_pair = word_pair.split(':')
            pre_v = w2v[word_pair[0]] - w2v[word_pair[1]]
            score = dot(gender_v, pre_v)/(norm(gender_v)*norm(pre_v))
            if score > max_score:
                max_idx = i
                max_score = score
        if max_idx == 0:
            definition_num += 1
            if sub_idx >= sub_start:
                sub_definition_num += 1
        elif max_idx == 1 or max_idx == 2:
            none_num += 1
            if sub_idx >= sub_start:
                sub_none_num += 1
        elif max_idx == 3:
            stereotype_num += 1
            if sub_idx >= sub_start:
                sub_stereotype_num += 1
        total_num += 1
    print('definition: {}'.format(definition_num / total_num))
    print('stereotype: {}'.format(stereotype_num / total_num))
    print('none: {}'.format(none_num / total_num))

    if sub_definition_num == 0:
        print('sub definition: 0')
    else:
        print('sub definition: {}'.format(sub_definition_num / sub_size))
    if sub_stereotype_num == 0:
        print('sub stereotype: 0')
    else:
        print('sub stereotype: {}'.format(sub_stereotype_num / sub_size))
    if sub_none_num == 0:
        print('sub none: 0')
    else:
        print('sub none: {}'.format(sub_none_num / sub_size))

def de_biassing_emb(emb_name, hp, generator):
    generator.eval()
    debias_emb_txt = 'debiased_{}/gender_debiased.txt'.format(emb_name)
    debias_emb_bin = 'debiased_{}/gender_debiased.bin'.format(emb_name)
    w2v = \
        KeyedVectors.load_word2vec_format(hp.word_embedding,
                                          binary=hp.emb_binary)

    vector_size = 100 if (emb_name == 'glove_Kor') else 300
    emb = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=vector_size)

    # print('Start generating')
    inputs = torch.split(torch.stack([torch.FloatTensor(w2v[word]) for word in w2v.vocab.keys()]), 1024)
    debias_embs = []
    for input in inputs:
        if hp.gpu >= 0:
            input = input.cuda()
        with torch.no_grad():
            debias_embs += [generator(input).data.cpu().numpy()]
    debias_embs = np.concatenate(debias_embs)
    emb.add([word for word in w2v.vocab.keys()], debias_embs)

    return emb


def main(args):
    torch.manual_seed(0)
    cuda.manual_seed_all(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emb", type=str, default="glove_Kor", required=False)
    args = parser.parse_args(args)

    sys.path.append('./hyperparams/')
    if args.emb == 'glove':
        from hyperparams_glove import Hyperparams as hp
    elif args.emb == 'gn':
        from hyperparams_gn_glove import Hyperparams as hp
    elif args.emb == 'glove_Kor':
        from hyperparams_glove_Kor import Hyperparams as hp

    if hp.gpu:
        cuda.set_device(hp.gpu)
    torch.manual_seed(hp.seed)

    # print('Generating emb...')
    checkpoint = torch.load(hp.eval_model, map_location=lambda storage, loc: storage.cuda(hp.gpu))

    encoder = model.Encoder(hp.emb_size, hp.hidden_size, hp.dropout_rate)
    if hp.gpu >= 0:
        encoder.cuda()
    encoder.load_state_dict(checkpoint['encoder'])
    w2v = de_biassing_emb(args.emb, hp, encoder)
    eval_bias_analogy(args.emb, w2v)

    # print('Saving emb...')
    debias_emb_txt = 'src/debiased_{}/gender_debiased.txt'.format(args.emb)
    debias_emb_bin = 'src/debiased_{}/gender_debiased.bin'.format(args.emb)
    w2v.save_word2vec_format(debias_emb_bin, binary=True)
    w2v.save_word2vec_format(debias_emb_txt, binary=False)

if __name__ == "__main__":
    main(sys.argv[1:])
