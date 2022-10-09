from wordfreq import word_frequency, zipf_frequency
from readability_score.calculators.nl.fleschdouma import *
from readability_score.calculators.flesch import *
from readability_score.calculators.de.flesch_german import *
from readability_score.calculators.ru.flesch_russian import *
import textstat
import numpy as np

def get_word_length(word):
    """calculate word length"""
    return len(str(word))

def get_sent_lengths(merged,feat):
    """calculate sentence length and group sentence predictions"""
    sentence_data = merged.groupby('sentence_num')
    sent_lengths = []
    sent_predictions = []
    for s, data in sentence_data:
        #print(data, len(data))
        sent_lengths.append(len(data))
        #sent_predictions.append(np.mean(data[feat]))
        sent_predictions.append(np.sum(data[feat]))
    return sent_lengths, sent_predictions

def get_word_freq(word):
    """calculate word length"""
    return zipf_frequency(str(word), 'en')

def reading_ease(text, dataset):
    if dataset == "geco-nl":
        fd = FleschDouma(text, locale='nl_NL')
        flesch = fd.readingindex
    if dataset == "potsdam":
        fg = FleschGerman(text, locale='de_DE')
        flesch = fg.readingindex
    if dataset == "rsc":
        fr = FleschRussian(text, locale='ru_RU')
        flesch = fr.readingindex
    if dataset == "all-en" or dataset == "dundee" or dataset == "geco" or dataset == "zuco" or dataset == "all-langs":
        fl = Flesch(text, locale='en_GB')
        flesch = textstat.flesch_reading_ease(text)
        #print(text)
        #print(fl.reading_ease, flesch)
    return flesch

def get_flesch_scores(merged, dataset, feat):
    """calculate sentence flesch score and group sentence predictions"""
    sentence_data = merged.groupby('sentence_num')
    sent_fleschs = []
    sent_predictions = []

    for s, data in sentence_data:
        flesch = reading_ease(" ".join(map(str, data['word'].tolist())), dataset)
        sent_fleschs.append(flesch)
        # todo: this should also be the sum??
        sent_predictions.append(np.sum(data[feat]))
        #sent_predictions.append(np.mean(data[feat]))

    return sent_fleschs, sent_predictions
