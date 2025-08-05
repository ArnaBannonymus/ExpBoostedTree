import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def bar(importances, feature_names, top_n=10, title="Importances", **kwargs):
    idx = np.argsort(importances)[::-1][:top_n]
    vals = importances[idx]
    names = np.array(feature_names)[idx]
    fig, ax = plt.subplots(**kwargs)
    ax.barh(range(len(vals)), vals[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1])
    ax.set_title(title)
    plt.tight_layout()
    return fig

def waterfall(contrib, feature_names, base_value=0, title="Waterfall", **kwargs):
    cum = np.concatenate(([base_value], base_value + np.cumsum(contrib)))
    fig, ax = plt.subplots(**kwargs)
    for i, v in enumerate(contrib):
        ax.barh(0, v, left=cum[i], color="tab:red" if v>=0 else "tab:blue")
    ax.axvline(base_value, color="k", linestyle="--")
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    return fig

def waterfall_plotly(contrib, feature_names, base_value=0, title="Interactive Waterfall"):
    cum = np.concatenate(([base_value], base_value + np.cumsum(contrib)))
    fig = go.Figure()
    for i, v in enumerate(contrib):
        fig.add_trace(go.Bar(
            x=[v], y=[feature_names[i]], base=[cum[i]],
            orientation="h", marker_color="red" if v>=0 else "blue",
            name=feature_names[i]
        ))
    fig.add_vline(x=base_value, line_dash="dash")
    fig.update_layout(title=title, xaxis_title="Contribution")
    return fig
