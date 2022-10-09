import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc
import sys
import sklearn.metrics
from nltk.corpus import wordnet
from itertools import chain
import seaborn as sns
import scipy



DATASET = "dundee"
MODELS = ["xlm-mlm-17-1280", "xlm-mlm-100-1280","bert-base-multilingual-cased", "bert-base-uncased"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["12"]

DATASET = "geco"
MODELS = ["xlm-mlm-17-1280", "xlm-mlm-100-1280","bert-base-multilingual-cased", "bert-base-uncased"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["549", "237", "12", "886", "79"]

DATASET = "geco-nl"
ds_pretty = "GECO (nl)"
MODELS = ["xlm-mlm-100-1280"]#, "bert-base-multilingual-cased", "xlm-mlm-17-1280", "xlm-mlm-100-1280"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["549", "886", "237", "79", "12"]





DATASET = "potsdam"#"geco-nl"#"dundee"#"all-langs"#"all-en" #"geco-nl"
ds_pretty = "PoTeC"
MODELS = [ "xlm-mlm-100-1280"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["549", "237", "12", "79", "886"]

DATASET = "rsc"
ds_pretty = "RSC"
MODELS = ["xlm-mlm-100-1280","bert-base-multilingual-cased", "rubert-base-cased"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["549", "237", "12", "79", "886"]






DATASET = "all-langs"
ds_pretty = "ALL langs"
MODELS = ["xlm-mlm-100-1280", "bert-base-multilingual-cased"]# "bert-base-uncased", "bert-base-multilingual-cased"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["12", "237", "79", "549", "886"]

DATASET = "all-en"
ds_pretty = "ALL (en)"
MODELS = ["xlm-mlm-100-1280",  "bert-base-multilingual-cased"]#, "bert-base-multilingual-cased", "xlm-mlm-17-1280", "xlm-mlm-100-1280"]#, "xlm-mlm-17-1280", "bert-base-uncased", "bert-base-multilingual-cased"] #"bert-base-dutch-cased", "xlm-mlm-17-1280"
SEEDS = ["549", "237", "886", "79"]



model_list = ['xlm-mlm-17-1280', 'xlm-mlm-100-1280', 'bert-base-multilingual-cased', 'bert-base-uncased', 'bert-base-dutch-cased', 'bert-base-german-cased', 'rubert-base-cased', 'bert-base-cased', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024']
model_names = {'bert-base-uncased': 'BERT-en', 'bert-base-dutch-cased' : 'BERT-nl', 'bert-base-german-cased' : 'BERT-de', 'rubert-base-cased' : 'BERT-ru', 'bert-base-cased' : 'BERT-en-cased', 'bert-base-multilingual-cased': 'BERT-multi', 'xlm-mlm-en-2048': 'XLM-en', 'xlm-mlm-ende-1024' : 'XLM-ende', 'xlm-mlm-17-1280':'XLM-17', 'xlm-mlm-100-1280':'XLM-100'}
#cmaplist =  ["#44A2C4", "#44A2C4","#44A2C4","#44A2C4", "#337F9A", "#92D050", "#D5E600", "#FFEB00", "#FFB14C", '#DC7810', "#A30071", "#A072C4"]
cmaplist = ["#44A2C4", "#337F9A", "#10997D", "#66BB97", "#92D050", "#D5E600", "#FFEB00", "#FFB14C", "#DC7810", "#C00000", "#A30071", "#A072C4", "#642D8F", "#203864", "#44A2C4", "#337F9A", "#44A2C4", "#337F9A"]
#cmaplist =  ["#44A2C4", "#92D050", "#FFB14C", '#DC7810', "#A30071", "#A072C4"] #"#337F9A", "#D5E600", "#FFEB00"


def load_spacy_lang():
    if DATASET == "potsdam":
        nlp = spacy.load("de_core_news_sm")
    elif DATASET == "geco-nl":
        nlp = spacy.load("nl_core_news_sm")
    elif DATASET == "dundee" or DATASET == "geco" or DATASET == "zuco" or DATASET == "all-en" or DATASET == "all-langs":
        nlp = spacy.load("en_core_web_sm")
    elif DATASET == "rsc":
        import ru2
        nlp = spacy.load('ru2')
        nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    else:
        sys.exit("ERROR! Dataset does not exist: "+DATASET)

    return nlp

def custom_tokenizer(wordlist):
    """replace spacy tokenizer with already tokenized list of words"""
    return Doc(nlp.vocab, words=wordlist, spaces=None)


def pos_tag(dataset):
    #print(dataset.head())
    words = [str(w).replace("<eos>", ".") for w in dataset['word']]
    #print(words)
    #nlp.tokenizer = custom_tokenizer()
    tagged_sent = nlp(custom_tokenizer(words))
    dataset['pos_tags'] = [token.pos_ for token in tagged_sent]

    return dataset

def feature_analysis():
    """analyze predictions per feature"""
    feats = [c for c in merged.columns if "_scaled" in c]
    print(feats)
    for f in feats:
        scaled = merged[f]
        pred = merged[f.replace("_scaled","_pred")]

        fig, ax = plt.subplots(1, 1, sharey=True)
        bp = ax.violinplot([scaled,pred], showmeans=True )
        #bp = ax2.violinplot(pos_data['n_fix_pred'])
        ax.set_xticks([1,2])
        ax.set_xticklabels(["true", "predicted"])
        plt.title(f)
        plt.show()

def get_synonyms(dataset):
    words = [str(w).replace("<eos>", ".") for w in dataset['word']]
    synonyms_number = []
    for word in words:
        synonyms = wordnet.synsets(word)
        #print(word, synonyms)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        synonyms_number.append(len(lemmas))
    #print(word, lemmas)

    dataset['synonyms'] = [s for s in synonyms_number]

    return dataset



model_res = {}
model_res_syn = {}
nlp = load_spacy_lang()
print("Spacy model loaded.")
nlp.tokenizer = custom_tokenizer

tags = ['NOUN', 'PRON', 'DET', 'ADP', 'ADJ', 'VERB', 'ADV', 'CCONJ', 'SCONJ'] # 'PROPN', 'AUX', 'NUM', "SYM", 'PUNCT', 'INTJ', 'X' 'PART'

#tags = ['ADJ','NOUN','VERB','ADV']
for midx, model in enumerate(MODELS):
    #print(model)
    if "true" not in model_res:
        model_res["true"] = {}
    if model not in model_res:
        model_res[model] = {}
    if "true" not in model_res_syn:
        model_res_syn["true"] = {}
    if model not in model_res_syn:
        model_res_syn[model] = {}

    for s in SEEDS:
        try:
            testset = pd.read_csv("fine-tuned/"+DATASET + "/" + s + "/test_dataset.csv", index_col=0)

            scaled_true_feats = pd.read_csv("fine-tuned/"+DATASET  + "/" + s+ "/scaled-test-"+DATASET+".csv", index_col=0)
            scaled_true_feats.columns = [str(col) + '_scaled' for col in scaled_true_feats.columns]

            preds = pd.read_csv("fine-tuned/"+DATASET + "/" + s+ "/preds-"+s+"-"+model+".csv", index_col=0)
            preds.columns = [str(col) + '_pred' for col in preds.columns]

            merged = testset.join(scaled_true_feats)
            merged = merged.join(preds)
            merged = merged.dropna()


            # PoS tag
            tagged_data = pos_tag(merged)
            #tagged_data = get_synonyms(tagged_data)
            #print(tagged_data.head())
            #print(tagged_data['synonyms'].unique())
            #print(tagged_data[tagged_data.pos_tags == "PART"])
            #print(tagged_data[tagged_data.pos_tags == "PROPN"])
            tagged_data.loc[tagged_data['pos_tags'] == 'SCONJ', 'pos_tags'] = 'CONJ'
            tagged_data.loc[tagged_data['pos_tags'] == 'CCONJ', 'pos_tags'] = 'CONJ'
            #print(tagged_data[tagged_data.pos_tags == "ADP"])
            tagged_data.loc[tagged_data['pos_tags'] == 'ADP', 'pos_tags'] = 'PREP'
            #print("---")
            # analyze absolute difference
            for index, row in tagged_data.iterrows():
                diff = abs(row['fix_prob_scaled']-row['fix_prob_pred'])
                #if diff > 20.0:
                    #print(row['word'], row['pos_tags'], diff, row['mean_fix_dur_scaled'], row['mean_fix_dur_pred'])
                #if diff < 0.005:
                #    print(row['word'], row['pos_tags'], diff, row['mean_fix_dur_scaled'], row['mean_fix_dur_pred'])
                #    print(row['word'], diff)
            #print(tagged_data['pos_tags'].unique())
            #tags.extend([t for t in list(tagged_data['pos_tags'].unique()) if t in all_tags and t not in tags])
            # analyze predictions per PoS tag
            tags = ['NOUN', 'PRON', 'DET', 'ADP', 'ADJ', 'VERB', 'ADV', 'CONJ', 'PREP']
            for idx,pos in enumerate(tags):
                isPOS = tagged_data['pos_tags'] == pos
                pos_data = tagged_data[isPOS]

                #print(list(pos_data['mean_fix_dur_scaled']))
                #mae = sklearn.metrics.mean_absolute_error(pos_data['mean_fix_dur_scaled'],pos_data['mean_fix_dur_pred'])
                #print(pos, mae)
                if midx == 0:
                    if pos not in model_res["true"]:
                        model_res["true"][pos] = list(pos_data['fix_prob_scaled'])
                    else:
                        model_res["true"][pos].extend(list(pos_data['fix_prob_scaled']))

                if pos not in model_res[model]:
                    model_res[model][pos] = list(pos_data['fix_prob_pred'])
                else:
                    model_res[model][pos].extend(list(pos_data['fix_prob_pred']))

            """
            for idx,pos in enumerate(tagged_data['synonyms'].unique()):
                isPOS = tagged_data['synonyms'] == pos
                syn_data = tagged_data[isPOS]

                #print(list(pos_data['mean_fix_dur_scaled']))
                #mae = sklearn.metrics.mean_absolute_error(pos_data['mean_fix_dur_scaled'],pos_data['mean_fix_dur_pred'])
                #print(pos, mae)
                if midx == 0:
                    if pos not in model_res_syn["true"]:
                        model_res_syn["true"][pos] = list(syn_data['mean_fix_dur_scaled'])
                    else:
                        model_res_syn["true"][pos].extend(list(syn_data['mean_fix_dur_scaled']))

                if pos not in model_res_syn[model]:
                    model_res_syn[model][pos] = list(syn_data['mean_fix_dur_pred'])
                else:
                    model_res_syn[model][pos].extend(list(syn_data['mean_fix_dur_pred']))
            """
        except FileNotFoundError:
            print("no model for seed:", model, s)
            continue

#print(model_res.keys())
#print(model_res_syn)


for m in MODELS:
    fig, ax = plt.subplots(1, 1, figsize=(7,3))#, sharey=True)
    tags = ['DET', 'PREP', 'CONJ', 'NOUN', 'ADJ', 'VERB', 'ADV'] #'PRON'
    for pidx, pos in enumerate(tags):
        print(pos, np.mean(model_res[m][pos]), np.std(model_res[m][pos]))
        mae = sklearn.metrics.mean_absolute_error(model_res["true"][pos],model_res[m][pos])
        true_mean = [np.mean(model_res["true"][pos])] * len(model_res["true"][pos])
        mae_mean = sklearn.metrics.mean_absolute_error(model_res["true"][pos],true_mean)
        diff_mean = mae - mae_mean
        #print(pos, m, mae, diff_mean, 100-mae)
        #ax.bar(pidx, 100-mae, label=pos, color=cmaplist[pidx])#, alpha=1-(0.1*midx))
        pal = sns.color_palette("husl", 8)
        ax.bar(pidx, np.std(model_res[m][pos]), label=pos, color=pal[pidx])
            #tags = [t for t in tags if t != pos]
    #ax[pidx].bar(midx, diff_mean, label=model_names[m], color=cmaplist[model_list.index(m)])
    ax.set_xticks(list(range(len(tags))))
    ax.set_xticklabels(tags, rotation=90, fontsize=12)
    ax.title.set_text(m)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylim([90,100])
    #ax[pidx].set_ylim([89,93])


    plt.title(ds_pretty + " " + m)
    #plt.legend()
    ax.set_ylabel("Standard dev.", fontsize=12)
    plt.tight_layout()
    plt.savefig(DATASET+"-"+m+"-fixProp-PoS_StDev.pdf")
    plt.show()

"""
for m in MODELS:
    fig, ax = plt.subplots(1, 1, figsize=(7,3))#, sharey=True)
    #tagged_data = tagged_data.sort_values(by=['synonyms'])
    for pidx, pos in enumerate(tagged_data['synonyms'].unique()):
        mae = sklearn.metrics.mean_absolute_error(model_res_syn["true"][pos],model_res_syn[m][pos])
        true_mean = [np.mean(model_res_syn["true"][pos])] * len(model_res_syn["true"][pos])
        mae_mean = sklearn.metrics.mean_absolute_error(model_res_syn["true"][pos],true_mean)
        diff_mean = mae - mae_mean
        #print(pos, m, mae, diff_mean, 100-mae)
        ax.line(pos, 100-mae, label=pos)
        tagged_data.loc[pidx, 'mae'] = 100-mae

    #ax[pidx].bar(midx, diff_mean, label=model_names[m], color=cmaplist[model_list.index(m)])
    ax.set_xticks(list((tagged_data['synonyms'].unique())))
    ax.set_xticklabels(tagged_data['synonyms'].unique(), rotation=90, fontsize=12)
    ax.title.set_text(m)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([90,100])
    #ax[pidx].set_ylim([89,93])

    plt.title(ds_pretty + " " + m)
    #plt.legend()
    ax.set_ylabel("Accuracy", fontsize=12)
    plt.tight_layout()
    plt.savefig(DATASET+"-"+m+"-MFD-Syn_new.pdf")
    plt.show()
    plt.close()

    #print(tagged_data[['synonyms', 'mae']])
    tagged_data2 = tagged_data.dropna()
    spearman = scipy.stats.spearmanr(tagged_data2['synonyms'], tagged_data2['mae'])
    print("spearman synonyms", m, spearman[0], spearman[1])
"""
