import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc
import sys
import sklearn.metrics
import seaborn as sns
import collections
import scipy
import readability_features
from mean_baseline_correlations import calculate_mean_baseline


DATASET = "all-en" #"geco-nl", "all-langs", "potsdam", "rsc", "all-en"
print(DATASET)
ds_pretty = "ALL-EN"
MODELS = ["true_labels", "bert-multi-notune", "bert-multi-random-init", "xlm100-random-init", "bert-base-multilingual-cased", "xlm-mlm-100-1280"]
SEEDS = ["12", "237", "79", "886", "549"]
features = ["fix_prob", "mean_fix_dur"] #"n_fix", "n_refix", "first_fix_dur", "first_pass_dur", "total_fix_dur", "reread_prob"
analysis_set = ["flesch", "word_freq", "word_length", "sent_length"]#, "flesch", "sent_length"]


def plot_predictions(model_dicts, analysis, feat):
    """plot predictions of a give feature and the real eye-tracking feature values"""

    model_names = {'bert-base-uncased': 'BERT-en', 'bert-base-dutch-cased' : 'BERT-nl', 'bert-base-german-cased' : 'BERT-de', 'rubert-base-cased' : 'BERT-ru', 'bert-base-cased' : 'BERT-en-cased', 'bert-base-multilingual-cased': 'Fine-tuned BERT-multi', "bert-multi-notune": 'BERT-multi', 'xlm-mlm-en-2048': 'XLM-en', 'xlm-mlm-ende-1024' : 'XLM-ende', 'xlm-mlm-17-1280':'XLM-17', 'xlm-mlm-100-1280':'Fine-tuned XLM-100', "true_labels":"Gaze Data"}
    model_list = ['bert-base-uncased', 'bert-base-dutch-cased', 'bert-base-german-cased', 'rubert-base-cased', 'bert-base-cased', 'bert-base-multilingual-cased', "bert-multi-notune", 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-17-1280', 'xlm-mlm-100-1280']
    cmaplist =  ["#C00000", "#C00000","#C00000","#C00000", "#D53B80", "#D53B80", "#D5763B", "#000000", "#33BA7A", '#26B5AD', "#33BA7A", "#26B5AD"]

    for model, word_stats in model_dicts.items():
        model_means = {}
        for k,v in word_stats.items():
            model_means[k] = (np.mean(v), np.std(v))
            #print(k, np.mean(v), np.std(v), len(v))
        ordered = collections.OrderedDict(sorted(model_means.items()))
        #print(analysis, feat, model, len(ordered))
        if model == "true_labels":
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], label=model_names[model], color="#A30071", linestyle="-.", linewidth=2)
            err_min = [d[0] - d[1] for d in ordered.values()]
            err_plus = [d[0] + d[1] for d in ordered.values()]
        elif model == "bert-multi-notune" or model == "xlm100-notune":
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], color='grey', linestyle="--", linewidth=2, label="Random baseline")
        elif model == "bert-multi-random-init":
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], color='red', linestyle="--", linewidth=2, label="Random init. BERT")
            err_min = [d[0] - d[1] for d in ordered.values()]
            err_plus = [d[0] + d[1] for d in ordered.values()]
            plt.fill_between(list(ordered.keys()), err_min, err_plus, color='red', alpha=0.2)
        elif model == "xlm100-random-init":
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], color='blue', linestyle="--", linewidth=2, label="Random init. XLM")
            err_min = [d[0] - d[1] for d in ordered.values()]
            err_plus = [d[0] + d[1] for d in ordered.values()]
            plt.fill_between(list(ordered.keys()), err_min, err_plus, color='blue', alpha=0.2)
        elif model == "mean":
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], color='#ccc4c4', linestyle=":", linewidth=2, label="Mean baseline")
        else:
            plt.plot(list(ordered.keys()), [n[0] for n in ordered.values()], label=model_names[model], color=cmaplist[model_list.index(model)], linewidth=2)
            err_min = [d[0] - d[1] for d in ordered.values()]
            err_plus = [d[0] + d[1] for d in ordered.values()]
            plt.fill_between(list(ordered.keys()), err_min, err_plus, color=cmaplist[model_list.index(model)], alpha=0.2)

    plt.xlim(0,100)  # flesch
    #plt.ylim(20,100)
    plt.xlabel(analysis, fontsize=16)
    plt.ylabel("(predicted) "+feat, fontsize=16)
    plt.title(ds_pretty, fontsize=18)
    plt.legend(loc='lower right', fontsize=13)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.tight_layout()
    #plt.show()
    plt.savefig(analysis+"-"+feat+"-"+DATASET+"-_new.pdf")
    plt.close()

def bar_plot_correlations(corr_results):
    print(corr_results)
    model_names = {'bert-base-uncased': 'BERT-en', 'bert-base-dutch-cased' : 'BERT-nl', 'bert-base-german-cased' : 'BERT-de', 'rubert-base-cased' : 'BERT-ru', 'bert-base-cased' : 'BERT-en-cased', 'bert-base-multilingual-cased': 'Fine-tuned BERT-multi', "bert-multi-notune": 'BERT-multi', 'xlm-mlm-en-2048': 'XLM-en', 'xlm-mlm-ende-1024' : 'XLM-ende', 'xlm-mlm-17-1280':'XLM-17', 'xlm-mlm-100-1280':'Fine-tuned XLM-100', "true_labels":"Gaze Data"}

    # todo: which hue? both groupings are good?
    #ax = sns.barplot(x="model", y="corr", data=corr_results, hue="analysis", palette="viridis", linewidth=2.5, edgecolor=".2")
    cmaplist =["#A30071", "grey", "red", "blue", "#D53B80", "#26B5AD", "#ccc4c4"]
    ax = sns.barplot(x="analysis", y="corr", data=corr_results, hue="model", palette=cmaplist, linewidth=1, edgecolor="black", alpha=0.8)
    #plt.legend(labels=["Gaze Data", "Random", "Random init. BERT", "Random init. XLM", "Fine-tuned BERT", "Fine-tuned XLM", "Mean"], loc='center left')

    #plt.show()
    #plt.savefig("...".pdf")
    plt.close()


for feat in features:
    corr_results = pd.DataFrame()
    corr_results_num = pd.DataFrame()
    i = 0
    for analysis in analysis_set:

        all_models = {}
        for midx, model in enumerate(MODELS):
            merged = pd.DataFrame()
            word_stats = {}

            for s in SEEDS:

                try:
                    testset = pd.read_csv("fine-tuned/" +DATASET + "/" + s + "/test_dataset.csv", index_col=0)

                    scaled_true_feats = pd.read_csv("fine-tuned/" +DATASET  + "/" + s+ "/scaled-test-"+DATASET+".csv", index_col=0)
                    scaled_true_feats.columns = [str(col) + '_scaled' for col in scaled_true_feats.columns]
                    if model == "true_labels":
                        # these are just fillers
                        preds = pd.read_csv("fine-tuned/" +DATASET + "/" + s+ "/preds-"+s+"-xlm-mlm-100-1280.csv", index_col=0)
                    elif model == "bert-multi-notune":
                        preds = pd.read_csv("random-baseline/" + DATASET + "/" + s + "/preds-"+s+"-bert-base-multilingual-cased.csv", index_col=0)
                    elif model == "xlm100-notune":
                        preds = pd.read_csv("random-baseline/" + DATASET + "/" + s + "/preds-"+s+"-xlm-mlm-100-1280.csv", index_col=0)
                    elif model == "xlm100-random-init":
                        preds = pd.read_csv("random-init-D/" + DATASET + "/" + s + "/preds-"+s+"-xlm-mlm-100-1280-RandomTrue-FinetuneTrue.csv", index_col=0)
                    elif model == "bert-multi-random-init":
                        preds = pd.read_csv("random-init-D/" + DATASET + "/" + s + "/preds-"+s+"-bert-base-multilingual-cased-RandomTrue-FinetuneTrue.csv", index_col=0)
                    else:
                        preds = pd.read_csv("fine-tuned/" + DATASET + "/" + s+ "/preds-"+s+"-"+model+".csv", index_col=0)
                    preds.columns = [str(col) + '_pred' for col in preds.columns]

                    merged_seed = testset.join(scaled_true_feats)
                    merged_seed = merged_seed.join(preds)
                    merged_seed = merged_seed.dropna()
                    merged_seed['sentence_num'] = str(s) + "-" + merged_seed['sentence_num'].astype(str)

                    merged = pd.concat([merged, merged_seed])

                except FileNotFoundError:
                    print("no model for seed:", model, s)
                    continue

            scaled = merged[feat+"_scaled"]
            preds = merged[feat+"_pred"]
            words = merged['word']

            if analysis == "word_length" or analysis == "word_freq":
                for word, true, pred in zip(words, scaled, preds):

                    if analysis == "word_length":
                        ws = readability_features.get_word_length(word)
                    elif analysis == "word_freq":
                        ws = readability_features.get_word_freq(word)

                    if model == "true_labels":
                        if round(ws) not in word_stats:
                            word_stats[round(ws)] = [true]
                        else:
                            word_stats[round(ws)].append(true)
                    else:
                        if round(ws) not in word_stats:
                            word_stats[round(ws)] = [pred]
                        else:
                            word_stats[round(ws)].append(pred)
                mean_preds = [np.mean(l) for l in list(word_stats.values())]
                spearman = scipy.stats.spearmanr(list(word_stats.keys()), mean_preds)

            if analysis == "sent_length":
                if model == "true_labels":
                    sent_lengths, sent_predictions = readability_features.get_sent_lengths(merged, feat+'_scaled')
                else:
                    sent_lengths, sent_predictions = readability_features.get_sent_lengths(merged, feat+'_pred')
                spearman = scipy.stats.spearmanr(sent_predictions, sent_lengths)
                for l, p in zip(sent_lengths, sent_predictions):
                    if l not in word_stats:
                        word_stats[l] = [p]
                    else:
                        word_stats[l].append(p)
                    #word_stats[l] = [p]

            elif analysis == "flesch":
                if model == "true_labels":
                    sent_fleschs, sent_predictions = readability_features.get_flesch_scores(merged, DATASET, feat+'_scaled')
                else:
                    sent_fleschs, sent_predictions = readability_features.get_flesch_scores(merged, DATASET, feat+'_pred')
                #sent_fleschs, sent_predictions = readability_features.get_flesch_scores(merged, DATASET, feat)
                spearman = scipy.stats.spearmanr(sent_predictions, sent_fleschs)
                for l, p in zip(sent_fleschs, sent_predictions):
                    if round(l) not in word_stats:
                        word_stats[round(l)] = [p]
                    else:
                        word_stats[round(l)].append(p)

            #print(analysis, feat, model, spearman[0], spearman[1])

            significance = "*" if spearman[1] < 0.01 else ""

            corr_results.at[model, analysis] = "{:.2f}".format(spearman[0]) + significance
            corr_results_num.at[i, "corr"] = spearman[0]
            corr_results_num.at[i, "analysis"] = analysis
            corr_results_num.at[i, "model"] = model

            i += 1

            #print(word_stats)
            all_models[model] = word_stats

        mean_bs_corr, sign, word_stats_mean_bs = calculate_mean_baseline(feat+'_scaled', analysis, DATASET)
        significance_mean = "*" if sign < 0.01 else ""

        corr_results.at["mean", analysis] = "{:.2f}".format(mean_bs_corr) + significance_mean

        corr_results_num.at[i, "corr"] = mean_bs_corr
        corr_results_num.at[i, "analysis"] = analysis
        corr_results_num.at[i, "model"] = "mean"
        all_models["mean"] = word_stats_mean_bs
        i += 1

        plot_predictions(all_models, analysis, feat)
    print(feat)
    print(corr_results)

    print("\n\n")
    #print(feat)
    #print(corr_results.to_latex())
    #print("\n\n")

    #bar_plot_correlations(corr_results_num)
