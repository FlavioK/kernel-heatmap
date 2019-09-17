#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import minmax_scale
import matplotlib.colors as colors
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def splitStartEndTimes(blockTimes):
    start_times = []
    stop_times = []
    for i in range(len(blockTimes)):
        if (i%2) == 0:
            start_times.append(blockTimes[i])
        else:
            stop_times.append(blockTimes[i])
    return start_times, stop_times

def getMinStartMaxEndTimeOfRepetition(times):
    start_times, end_times = splitStartEndTimes(times)
    minStart = min(start_times)
    maxEnd = max(end_times)
    return minStart, maxEnd

def getTimesOfRepetition(nofKernel, nofBlocks, nofRepetitions, thisRepetition, blockTimes):
    durationRep = {}
    for kernel in range(0,nofKernel):
        # Get time subset of this blocks
        startIndex = 2 * nofBlocks*kernel*nofRepetitions
        repIndex = 2 * nofBlocks * thisRepetition
        # Take only the first repetitions of blocktimes
        times = blockTimes[startIndex + repIndex: startIndex + repIndex + 2 *nofBlocks]
        minStart, maxEnd = getMinStartMaxEndTimeOfRepetition(times)
        durationRep[kernel] = maxEnd-minStart
    return durationRep

def getKernelDurations(nofKernel, nofBlocks, nofRepetitions, blockTimes):
    durations = []
    for rep in range(0, nofRepetitions):
        dur_rep = getTimesOfRepetition(1, nofBlocks, nofRepetitions, rep, blockTimes)
        durations.append(dur_rep)

    return durations

def getJitter(kernelDurations, nofRepetitions, kernel=0):
    durations = []
    for rep in range(0, nofRepetitions):
        durations.append(kernelDurations[rep][kernel])
    durations.sort()
    minv = min(durations[:int(len(durations)*0.9)])
    maxv = max(durations[:int(len(durations)*0.9)])
    meanv = np.mean(durations[:int(len(durations)*0.9)])
    jitter = (maxv-minv)/meanv
    return jitter, meanv, maxv, minv

def heatmap(data, row_labels, col_labels, ax=None, log=False, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    if log:
        im = ax.imshow(data, norm=colors.LogNorm(vmin=data.min(), vmax=data.max()), **kwargs)
    else:
        im = ax.imshow(data, **kwargs)

    # Create colorbar
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    colorData = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(colorData.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mp.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(colorData[i, j]) > threshold])
            if j%4 == 0:
                kw.update(weight='bold')
            else:
                kw.update(weight='normal')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def showHeatMap(heatData, kernels, norm=False, filename = 'dummy.pdf', xlabel='', label = '', maxparallel = 3, name='', log=False):
    # Generate x/y labels
    xlabels = []
    for kernel in kernels:
        for i in range(maxparallel):
                xlabels.append("{:d} x {:s}".format(i, kernel))

    fig, ax = plt.subplots(figsize=[16,5])

    fig.suptitle(name)

    colorData = heatData.copy()

    if norm:
        colorData = minmax_scale(colorData, axis=1)
        for y in range(heatData.shape[0]): # y axis
            for x in range(heatData.shape[1]): # x axis
                colorData[y, x] = heatData[y,x]/heatData[y,0]

    im, cbar = heatmap(colorData, kernels, xlabels, ax=ax,
		       cmap="magma_r", cbarlabel=label, log=log)

    kw = dict(fontsize='small')
    texts = annotate_heatmap(im, data=heatData, valfmt="{x:.1f}", **kw)

    ax.set_xlabel(xlabel)
    fig.savefig(filename, format='pdf')


def loadData(kernel1, kernel2, nof_kernel2, filepath):
    filename = filepath+"{:s}-{:s}-numberinterfering-{:d}-1000rep.json".format(kernel1, kernel2, nof_kernel2)
    with open(filename) as f1:
        data = json.load(f1)

    nofBlocks  = int(data['nofBlocks'])
    nofRep     = int(data['nof_repetitions'])
    dataSize   = int(data['data_size'])
    blockTimes = [float(i) for i in data['blocktimes']]
    durations = [float(i) for i in data['kernelDurations']]
    return nofBlocks, nofRep, blockTimes, durations

def processFiles(filepath = 'out/', name=''):
    kernels_tex = ['sqrt\_norm', 'conj', 'mult', 'gauss']
    kernels = ['sqr_norm', 'conj', 'mult', 'gauss']
    heatData = np.zeros((len(kernels), 3*len(kernels)), dtype=float)
    meanData = np.zeros((len(kernels), 3*len(kernels)), dtype=float)
    WCETData = np.zeros((len(kernels), 3*len(kernels)), dtype=float)
    BCETData = np.zeros((len(kernels), 3*len(kernels)), dtype=float)

    for i, kernel1 in enumerate(kernels):
        for j, kernel2 in enumerate(kernels):
            for nof_kernel2 in range(3):
                # Load data
                nofBlocks, nofRep, blockTimes, durations2 = loadData(kernel1, kernel2, nof_kernel2, filepath)
                durations = getKernelDurations(1, nofBlocks, nofRep, blockTimes)
                print(durations2)
                print(durations)
                jitter, meanv, maxv, minv= getJitter(durations, nofRep, 0)
                jitter = jitter * 100.0
                meanv = meanv/1000.0
                maxv = maxv/1000.0
                minv = minv/1000.0
                heatData[i, j*3+nof_kernel2] = jitter
                meanData[i, j*3+nof_kernel2] = meanv
                WCETData[i, j*3+nof_kernel2] = maxv
                BCETData[i, j*3+nof_kernel2] = minv
                print("{:s} {:s} kernel2:{:d} jitter: {:.2f}%, avg: {:.2f}us, WCET: {:.2f}, BCET: {:.2f}".format(kernel1, kernel2, nof_kernel2, jitter, meanv, maxv, minv))
    showHeatMap(heatData, kernels_tex, filename='traditional-jitter.pdf', xlabel='Jitter compared to avg. execution time [\%]', label="Jitter ratio\n Corresponds to algorithms listed on y-axis", name=name + ' - Jitter compared to avg. execution time', norm=True, log=False)
    showHeatMap(meanData, kernels_tex, filename='traditional-avg.pdf', xlabel='Avg. execution time in [us]', label="Avg. execution time ratio\n Corresponds to algorithms listed on y-axis", name=name + ' - Avg. execution time' , norm=True)
    #showHeatMap(WCETData, kernels, label="WCET [us]\n Corresponds to algorithms listed on y-axis", name=name, norm=True)
    #showHeatMap(BCETData, kernels, label="BCET [us]\n Corresponds to algorithms listed on y-axis", name=name, norm=True)

if __name__ == "__main__":
    processFiles('out/', 'Traditional')
    #processFiles('out-1000/', 'Traditional memory model')
    #plt.tight_layout()
    plt.show()
