import argparse
import os
import numpy as np
import dataloader
import criteria
import random

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig

def syntax_tree_replacement(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[
                    0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


