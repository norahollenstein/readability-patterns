import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc
import sys
import sklearn.metrics

DATASETS = ["geco-nl", "potsdam", "rsc", "all-en", "all-langs", 'zuco', 'dundee', 'geco']#"potsdam"#"zuco"#"rsc"#"dundee"
SEEDS = ["79", "549", "237", "886", "12"]

def get_mean_baseline(DATASETS):
    final = {}
    for d in DATASETS:
        baselines = {}
        true_data = pd.DataFrame()
        # take only seeds not in training data?
        for s in SEEDS:
            try:
                scaled_true_feats = pd.read_csv("fine-tuned/"+d  + "/" + s+ "/scaled-test-"+d+".csv", index_col=0)
                scaled_true_feats.columns = [str(col) + '_scaled' for col in scaled_true_feats.columns]
                true_data = pd.concat([true_data, scaled_true_feats])
            except FileNotFoundError:
                print("no file for: ", s, d)
                continue

        feats = [c for c in true_data.columns if "_scaled" in c]

        for f in feats:
            scaled = true_data[f]
            scaled_mean = [np.mean(scaled)]*len(scaled)
            #print(scaled_mean)

            mse = sklearn.metrics.mean_squared_error(np.array(scaled),np.array(scaled_mean))
            mae = sklearn.metrics.mean_absolute_error(np.array(scaled),np.array(scaled_mean))
            f = f.replace("_scaled", "")
            if not f in baselines:
                baselines[f] = [mae]
            else:
                baselines[f].append(mae)

        for f, maes in baselines.items():
            baselines[f] = np.mean(maes)
        baselines["mean"] = np.mean(list(baselines.values()))

        final[d] = baselines
        #print(final)


    for k,v in final.items():
        #print(v)
        print(k, "{:.2f}".format(100-v["mean"]))

    return baselines

get_mean_baseline(DATASETS)
