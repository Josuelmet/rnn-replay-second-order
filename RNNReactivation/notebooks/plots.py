import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def plot_pos(ax, pos, title):
    ax.plot(pos[...,0].T, pos[...,1].T, c='C0', alpha=0.5)
    ax.set(title=title)
    
def scatter_pos(ax, pos, title):
    ax.scatter(pos[...,0].T, pos[...,1].T, c='C0', alpha=0.01)
    ax.set(title=title)


colors = ['blue', 'purple', 'red', 'orange']
markers = ['o', 's', 'D', '^']

def stemplot(scalar_dict, xlabels):
    plt.figure(figsize=(12,2))
    plt.xticks(range(len(xlabels)), xlabels)
    ls = list(scalar_dict.values())
    for i in range(len(xlabels)):
        plt.plot([i]*2, [min(ls), ls[i]], c=colors[i%4])
        plt.plot(i, ls[i], marker=markers[i//4], c=colors[i%4])
        

def violinplot(array_dict, xlabels):
    plt.figure(figsize=(12,4))
    plt.xticks(1+np.arange(len(xlabels)), xlabels)
    parts = plt.violinplot(list(array_dict.values()), showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set(facecolor=colors[i%4], alpha=1)
    parts['cmedians'].set_color('yellow')
    
    
    
def add_border_and_ticks(ax, xlim, ylim, xticks=[0,1], yticks=[0,1],
                         linewidth=6, tickpercent=0.05, extra_padding=(0,0)):
    ranges = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    ticklengths = ranges * tickpercent
    # extra_padding is useful when saving as png
    ax.set_xlim((xlim[0] - ticklengths[0], xlim[1] + extra_padding[0]))
    ax.set_ylim((ylim[0] - ticklengths[1], ylim[1] + extra_padding[1]))
    border = Rectangle((xlim[0], ylim[0]), ranges[0], ranges[1],
                       fill=True, facecolor='gray', edgecolor='black',
                       linewidth=linewidth, clip_on=False)
    ax.add_artist(border)
    for xt in xticks:
        ax.vlines(xt, ylim[0]-ticklengths[0], ylim[0], color='k', linewidth=linewidth)
    for yt in yticks:
        ax.hlines(yt, xlim[0]-ticklengths[1], xlim[0], color='k', linewidth=linewidth)