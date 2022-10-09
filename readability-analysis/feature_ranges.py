import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.tokens import Doc
import sys
import sklearn.metrics
import collections
import scipy
from wordfreq import word_frequency


DATASET = "all-langs" #"geco-nl", "all-langs", "potsdam", "rsc", "all-en"
print(DATASET)
ds_pretty = "ALL-langs"
MODELS = ["true_labels","bert-base-multilingual-cased", "xlm-mlm-100-1280"]
SEEDS = ["12", "237", "79", "886", "549"]
features = ["mean_fix_dur", "fix_prob", "n_refix", "n_fix", "first_fix_dur", "first_pass_dur", "total_fix_dur", "reread_prob"]
model_names = {'bert-base-uncased': 'BERT-en', 'bert-base-dutch-cased' : 'BERT-nl', 'bert-base-german-cased' : 'BERT-de', 'rubert-base-cased' : 'BERT-ru', 'bert-base-cased' : 'BERT-en-cased', 'bert-base-multilingual-cased': 'BERT-multi', "bert-multi-notune": 'BERT-multi', 'xlm-mlm-en-2048': 'XLM-en', 'xlm-mlm-ende-1024' : 'XLM-ende', 'xlm-mlm-17-1280':'XLM-17', 'xlm-mlm-100-1280':'XLM-100', "true_labels":"Gaze Data"}
dataset_names = {'rsc': 'RU', 'potsdam' : 'DE', 'all-en' : 'EN', 'all-langs' : 'ALL-LANGS', 'geco-nl' : 'NL'}
cmaplist =  ["#D53B80", "#D5763B", "#33BA7A", '#26B5AD', "#33BA7A", "#26B5AD"]


for feat in features:
    merged = pd.DataFrame()
    fig, ax = plt.subplots(1,len(MODELS),sharey=True)

    for midx, model in enumerate(MODELS):

        print(model)
        merged_seed = pd.DataFrame()

        for s in SEEDS:

            if model == "true_labels":
                testset = pd.read_csv("fine-tuned/" +DATASET + "/" + s + "/test_dataset.csv")
                testset.columns = [str(col) + '_true' for col in testset.columns]
                scaled_true_feats = pd.read_csv("fine-tuned/" +DATASET  + "/" + s+ "/scaled-test-"+DATASET+".csv")
                scaled_true_feats.columns = [str(col) + '_true_scaled' for col in scaled_true_feats.columns]
                merged_seed = pd.concat([merged_seed, scaled_true_feats], axis=0, ignore_index=True)
            elif "notune" in model:
                preds_random = pd.read_csv("random-baseline/" + DATASET + "/" + s + "/preds-"+s+"-bert-base-multilingual-cased.csv")
                preds_random.columns = [str(col) + '_random-'+model for col in preds_random.columns]
                merged_seed = pd.concat([merged_seed, preds_random], axis=0, ignore_index=True)

            else:
                preds = pd.read_csv("fine-tuned/" + DATASET + "/" + s+ "/preds-"+s+"-"+model+".csv", index_col=0)
                preds.columns = [str(col) + '_finetuned-'+model for col in preds.columns]
                merged_seed = pd.concat([merged_seed, preds], axis=0, ignore_index=True)

            merged_seed = merged_seed.dropna()

        if model == "true_labels":
            sns.violinplot(ax=ax[midx], y=merged_seed[feat+"_true_scaled"], inner="quartile", color=cmaplist[midx])
            ax[midx].set_xlabel("true", fontsize=14)
            ax[midx].set_ylabel("")
        elif "notune" in model:
            sns.violinplot(ax=ax[midx], y=merged_seed[feat+"_random-"+model], inner="quartile", color=cmaplist[midx])
            ax[midx].set_xlabel("random", fontsize=14)
            ax[midx].set_ylabel("")
        else:
            sns.violinplot(ax=ax[midx], y=merged_seed[feat+"_finetuned-"+model], inner="quartile", color=cmaplist[midx])
            ax[midx].set_xlabel(model_names[model], fontsize=14)
            ax[midx].set_ylabel("")

    plt.suptitle(dataset_names[DATASET] + " " + feat, fontsize=16)
    plt.ylim(-10,110)
    plt.savefig("feature_ranges-"+DATASET+"-"+feat+"-camready.pdf")
    plt.show()
