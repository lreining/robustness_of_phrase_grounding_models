# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    sns.scatterplot(x=manipulations_label, y=performances)
plt.legend(model_labels)
plt.xlabel("Manipulation applied to flicker30k", fontdict={"size": 13})
plt.ylabel(metric, fontdict={"size": 13})
plt.title("Influence of manipulations on Recall", fontdict={"size": 15})
plt.tight_layout()
# %%
# sns.set_style("whitegrid", {'axes.grid' : False})
manipulation = "minus1"
metrics = data['test'][models[0]][data['test'][models[0]].index.str.startswith("Recall@1_")]
plt.figure(figsize=(9,6))
for i, model in enumerate(models):
    diff = data['test'][model].loc[metrics.index].values - data[manipulation][model].loc[metrics.index].values
    diff = diff.flatten()
    x = [metric[9:].capitalize() for metric in metrics.index]
    graph = sns.scatterplot(x=x[1:], y=diff[1:], label=model)
    graph.axhline(diff[0], color=nord_colors[i], lw=1)
plt.title("Influence of phrase category on recall difference", fontdict={"size": 15})
plt.xlabel("Category", fontdict={"size": 13})
plt.ylabel("Difference in Recall@1 between \n unscrambled and scrambled at word-level ", fontdict={"size": 13})
plt.tight_layout()
# %%
