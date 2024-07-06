# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# %%
evaluation_path = "evaluation_results"
models = ["glip_large_model", "glip_tiny_model_o365_goldg_cc_sbu", "glip_tiny_model_o365_goldg", "mdetr", "fiber"]
model_labels = ["GLIP Large", "GLIP Tiny (O365+GoldG+CC+SBU)", "GLIP Tiny (O365+GoldG)", "MDETR", "Fiber"]
manipulations = ['test', 'test_filtered', 'point33', 'point5', 'point66','minus1']
manipulations_label = ["Unfiltered", "Filtered", "Scrambled\n33% Depth", "Scrambled\n50% Depth", "Scrambled\n66% Depth", "Scrambled\nWord-level"]
data = {}
for manipulation in manipulations:
    df = {}
    for model in models:
        df[model] = pd.read_csv(os.path.join(evaluation_path, manipulation, model + ".csv"), index_col=0)
    data[manipulation] = df

# %%
sns.set_theme()
nord_colors = ["#5e81ac", "#81a1c1", "#88c0d0", "#bf616a", "#ebcb8b"]
sns.set_palette(nord_colors)
metric = "Recall@1_all"
for model in models:
    performances = []
    for manipulation in manipulations:
        performances.append(data[manipulation][model].loc[metric][0])
    sns.scatterplot(x=manipulations_label, y=performances, s=50)
plt.legend(model_labels)
plt.xlabel("Manipulation applied to flickr30k", fontdict={"size": 13})
plt.ylabel("Clean accuracy (Recall@1 all)", fontdict={"size": 13})
plt.tight_layout()
plt.savefig("figures/manipulations_recall.pdf")
plt.show()
# %%
# sns.set_style("whitegrid", {'axes.grid' : False})
manipulation = "minus1"
metrics = data['test'][models[0]][data['test'][models[0]].index.str.startswith("Recall@1_")]
plt.figure(figsize=(9,6))
for i, model in enumerate(models):
    diff = data[manipulation][model].loc[metrics.index].values - data['test'][model].loc[metrics.index].values
    diff = diff.flatten()
    x = [metric[9:].capitalize() for metric in metrics.index]
    graph = sns.scatterplot(x=x[1:], y=diff[1:], label=model_labels[i], s=50)
plt.xlabel("Category", fontdict={"size": 13})
plt.ylabel("Robustness (higher is better)", fontdict={"size": 13})
plt.legend(bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig("figures/phrase_category_recall_diff.pdf")
plt.show()
# %%
filtered_perfs = []
minus1_perfs = []
for model in models:
    filtered_perf = data['test_filtered'][model].loc[metric][0]
    minus1_perf = data['minus1'][model].loc[metric][0]
    filtered_perfs.append(filtered_perf)
    minus1_perfs.append(minus1_perf)
    sns.scatterplot(x=[minus1_perf],y=[filtered_perf])
sns.regplot(x=minus1_perfs, y=filtered_perfs, scatter=False, line_kws={"color":"black", "lw":1.5, "ls":"--"}, ci=None)
plt.xlabel("Robustness (Recall@1 with scrambling at word level)")
plt.ylabel("Clean accuracy (Recall@1 all)")
plt.legend(model_labels)
plt.tight_layout()
plt.savefig("figures/absolute_robustness_vs_clean.pdf")
plt.show()
# %%
filtered_perfs = []
robustness = []
for model in models:
    filtered_perf = data['test_filtered'][model].loc[metric][0]
    minus1_perf = data['minus1'][model].loc[metric][0]
    filtered_perfs.append(filtered_perf)
    robustness.append(minus1_perf-filtered_perf)
    sns.scatterplot(x=[filtered_perf],y=[minus1_perf-filtered_perf], s=50)
sns.regplot(x=filtered_perfs, y=robustness, scatter=False, line_kws={"color":"black", "lw":1.5, "ls":"--"}, ci=None)
plt.xlabel("Clean accuracy (Recall@1 all)")
plt.ylabel("Robustness (higher is better)")
plt.legend(model_labels)
plt.tight_layout()
plt.savefig("figures/relative_robustness_vs_clean.pdf")
plt.show()
# %%
