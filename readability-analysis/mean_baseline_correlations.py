import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc
import sys
import sklearn.metrics
from wordfreq import word_frequency
import readability_features
import scipy



def calculate_mean_baseline(feat, analysis, dataset):

    SEEDS = ["79", "549", "237", "886", "12"]

    baselines = {}
    true_data = pd.DataFrame()
    for s in SEEDS:
        try:
            testset = pd.read_csv("fine-tuned/" + dataset + "/" + s + "/test_dataset.csv", index_col=0)

            scaled_true_feats = pd.read_csv("fine-tuned/"+ dataset + "/" + s+ "/scaled-test-"+ dataset +".csv", index_col=0)
            scaled_true_feats.columns = [str(col) + '_scaled' for col in scaled_true_feats.columns]
            scaled_true_feats = testset.join(scaled_true_feats)
            true_data = pd.concat([true_data, scaled_true_feats])
        except FileNotFoundError:
            print("no file for: ", s, DATASET)
            continue

    scaled = true_data[feat]
    scaled_mean = [np.mean(scaled)]*len(scaled)
    true_data["mean"] = np.mean(scaled)

    scaled = true_data[feat]
    preds = true_data["mean"]
    words = true_data['word']

    word_stats = {}
    if analysis == "word_length" or analysis == "word_freq":
        for word, true, pred in zip(words, scaled, preds):

            if analysis == "word_length":
                ws = readability_features.get_word_length(word)
            elif analysis == "word_freq":
                ws = readability_features.get_word_freq(word)

            if ws not in word_stats:
                word_stats[ws] = [pred]
            else:
                word_stats[ws].append(pred)
        mean_preds = [np.mean(l) for l in list(word_stats.values())]
        spearman = scipy.stats.spearmanr(list(word_stats.keys()), mean_preds)

    if analysis == "sent_length":
        sent_lengths, sent_predictions = readability_features.get_sent_lengths(true_data, feat)
        spearman = scipy.stats.spearmanr(sent_predictions, sent_lengths)
        for l, p in zip(sent_lengths, sent_predictions):
            word_stats[l] = [p]
    elif analysis == "flesch":
        sent_fleschs, sent_predictions = readability_features.get_flesch_scores(true_data, dataset, feat)
        spearman = scipy.stats.spearmanr(sent_predictions, sent_fleschs)
        for l, p in zip(sent_fleschs, sent_predictions):
            if round(l) not in word_stats:
                word_stats[round(l)] = [p]
            else:
                word_stats[round(l)].append(p)

    #print(analysis, feat, "mean-baseline", spearman[0], spearman[1])
    return spearman[0], spearman[1], word_stats
