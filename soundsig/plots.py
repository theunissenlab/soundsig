from __future__ import division, print_function

import copy
import operator
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .stats import compute_R2


def multi_plot(the_data_list, plot_func, title=None, nrows=4, ncols=5, figsize=None, output_pattern=None,
               transpose=False, facecolor='gray', hspace=0.20, wspace=0.20, bottom=0.02, top=0.95, right=0.97, left=0.03):

    nsp = 0
    fig = None
    fig_num = 0
    plots_per_page = nrows*ncols

    data_list = the_data_list
    overflow_index = 0
    if transpose:
        data_list = [None]*len(data_list)
        for k in range(len(the_data_list)):
            page_offset = int(float(k) / plots_per_page)*plots_per_page
            if len(the_data_list) - page_offset < plots_per_page:
                new_index = page_offset + overflow_index
                overflow_index += 1
            else:
                sp = k % plots_per_page
                row = sp % nrows
                col = int(float(sp) / nrows)
                new_index = page_offset + row*ncols + col
            print('nsp=%d, k=%d, sp=%d, page_offset=%d, row=%d, col=%d, new_index=%d' %
                  (len(the_data_list), k, sp, page_offset, row, col, new_index))
            data_list[new_index] = the_data_list[k]

    for pdata in data_list:
        if nsp % plots_per_page == 0:
            if output_pattern is not None and fig is not None:
                #save the current figure
                ofile = output_pattern % fig_num
                plt.savefig(ofile, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')
            fig = plt.figure(figsize=figsize, facecolor=facecolor)
            fig_num += 1
            fig.subplots_adjust(top=top, bottom=bottom, right=right, left=left, hspace=hspace, wspace=wspace)
            if title is not None:
                plt.suptitle(title + (" (%d)" % fig_num))

        nsp += 1
        sp = nsp % plots_per_page
        ax = fig.add_subplot(nrows, ncols, sp)
        plot_func(pdata, ax)

    #save last figure
    if fig is not None and output_pattern is not None:
        ofile = output_pattern % fig_num
        plt.savefig(ofile, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')


def plot_pairwise_analysis(data_mat, feature_columns, dependent_column, column_names):
    """
        Does a basic pairwise correlation analysis between features and a dependent variable,
        meaning it plots a scatter plot with a linear curve fit through it, with the R^2.
        Then it plots a correlation matrix for all features and the dependent variable.

        data_mat: an NxM matrix, where there are N samples, M-1 features, and 1 dependent variable.

        feature_columns: the column indices of the features in data_mat that are being examined

        dependent_column: the column index of the dependent variable in data_mat

        column_names: a list of len(feature_columns)+1 feature/variable names. The last element is
                      the name of the dependent variable.
    """

    plot_data = list()
    for k,fname in enumerate(column_names[:-1]):
        fi = feature_columns[k]

        pdata = dict()
        pdata['x'] = data_mat[:, fi]
        pdata['y'] = data_mat[:, dependent_column]
        pdata['xlabel'] = column_names[fi]
        pdata['ylabel'] = column_names[-1]
        pdata['R2'] = compute_R2(pdata['x'], pdata['y'])
        plot_data.append(pdata)

    #sort by R^2
    plot_data.sort(key=operator.itemgetter('R2'), reverse=True)
    multi_plot(plot_data, plot_pairwise_scatter, title=None, nrows=3, ncols=3)

    all_columns = copy.copy(feature_columns)
    all_columns.append(dependent_column)

    C = np.corrcoef(data_mat[:, all_columns].transpose())

    Cy = C[:, -1]
    corr_list = [(column_names[k], np.abs(Cy[k]), Cy[k]) for k in range(len(column_names)-1)]
    corr_list.sort(key=operator.itemgetter(1), reverse=True)

    print('Correlations  with %s' % column_names[-1])
    for cname,abscorr,corr in corr_list:
        print('\t%s: %0.6f' % (cname, corr))

    fig = plt.figure()
    plt.subplots_adjust(top=0.99, bottom=0.15, left=0.15)
    ax = fig.add_subplot(1, 1, 1)
    fig.autofmt_xdate(rotation=45)
    im = ax.imshow(C, interpolation='nearest', aspect='auto', vmin=-1.0, vmax=1.0, origin='lower')
    plt.colorbar(im)
    ax.set_yticks(range(len(column_names)))
    ax.set_yticklabels(column_names)
    ax.set_xticks(range(len(column_names)))
    ax.set_xticklabels(column_names)


def plot_pairwise_scatter(plot_data, ax):

    x = plot_data['x']
    y = plot_data['y']
    if 'R2' not in plot_data:
        R2 = compute_R2(x, y)
    else:
        R2 = plot_data['R2']
    slope,bias = np.polyfit(x, y, 1)
    sp = (x.max() - x.min()) / 25.0
    xrng = np.arange(x.min(), x.max(), sp)

    clr = '#aaaaaa'
    if 'color' in plot_data:
        clr = plot_data['color']
    ax.plot(x, y, 'o', mfc=clr)
    ax.plot(xrng, slope*xrng + bias, 'k-')
    ax.set_title('%s: R2=%0.2f' % (plot_data['xlabel'], R2))
    if 'ylabel' in plot_data:
        ax.set_ylabel(plot_data['ylabel'])
    ax.set_ylim(y.min(), y.max())


def plot_histogram_categorical(x, xname='x', y=None, yname='y', color='g'):
    """
        Makes a histogram of the variable x, which is an array of categorical variables in their native representation
        (string or intger) . If a dependent continuous variable y is specified, it will make another plot which
        is a bar graph showing the mean and standard deviation of the continuous variable y.
    """

    ux = np.unique(x)
    xfracs = np.array([(x == xval).sum() for xval in ux]) / float(len(x))

    nsp = 1 + (y is not None)
    ind = range(len(ux))

    fig = plt.figure()
    ax = fig.add_subplot(nsp, 1, 1)
    ax.bar(ind, xfracs, facecolor=color, align='center', ecolor='black')
    ax.set_xticks(ind)
    ax.set_xticklabels(ux)
    ax.set_xlabel(xname)
    ax.set_ylabel('Fraction of Samples')

    if y is not None:
        y_per_x = dict()
        for xval in ux:
            indx = x == xval
            y_per_x[xval] = y[indx]

        ystats = [ (xval, y_per_x[xval].mean(), y_per_x[xval].std()) for xval in ux]
        ystats.sort(key=operator.itemgetter(0), reverse=True)

        xvals = [x[0] for x in ystats]
        ymeans = np.array([x[1] for x in ystats])
        ystds = np.array([x[2] for x in ystats])

        ax = fig.add_subplot(nsp, 1, 2)
        ax.bar(ind, ymeans, yerr=ystds, facecolor=color, align='center', ecolor='black')
        ax.set_xticks(ind)
        ax.set_xticklabels(xvals)
        #fig.autofmt_xdate()
        ax.set_ylabel('Mean %s' % yname)
        ax.set_xlabel(xname)
        ax.set_ylim(0, (ymeans+ystds).max())


def whist(x, **kwds):
    return plt.hist(x, weights=np.ones([len(x)]) / float(len(x)), **kwds)


def plot_confusion_matrix_single(pdata, ax):
    plt.imshow(pdata['cmat'], interpolation='nearest', aspect='auto', origin='upper', vmin=0, vmax=1)
    plt.title('p=%0.3f' % pdata['p'])


def make_phase_image(amp, phase, normalize=True, saturate=True, threshold=True):
    """
        Turns a phase matrix into an image to be plotted with imshow.
    """

    import husl

    nelectrodes,d = amp.shape
    alpha = copy.deepcopy(amp)
    if normalize:
        max_amp = np.percentile(amp, 98)
        alpha = alpha / max_amp

    img = np.zeros([nelectrodes, d, 4], dtype='float32')

    #set the alpha and color for the bins
    if saturate:
        alpha[alpha > 1.0] = 1.0 #saturate
    if threshold:
        alpha[alpha < 0.05] = 0.0 #nonlinear threshold

    cnorm = ((180.0 / np.pi) * phase).astype('int')
    for j in range(nelectrodes):
        for ti in range(d):
            #img[j, ti, :3] = husl.husl_to_rgb(cnorm[j, ti], 99.0, 50.0) #use HUSL color space: https://github.com/boronine/pyhusl/tree/v2.1.0
            img[j, ti, :3] = husl.husl_to_rgb(cnorm[j, ti], 99.0, 61.0) #use HUSL color space: https://github.com/boronine/pyhusl/tree/v2.1.0

    img[:, :, 3] = alpha

    return img


def draw_husl_circle():
    """ Draw an awesome circle whose angle is colored using the HUSL color space. The HUSL color space is circular, so
        it's useful for plotting phase. This figure could serve as a "color circle" as opposed to a color bar.
    """

    import husl

    #generate a bunch of points on the circle
    theta = np.arange(0.0, 2*np.pi, 1e-3)

    plt.figure()

    radii = np.arange(0.75, 1.0, 1e-2)
    for t in theta:
        x = radii*np.cos(t)
        y = radii*np.sin(t)

        a = (180.0/np.pi)*t
        c = husl.husl_to_rgb(a, 99.0, 50.0)
        plt.plot(x, y, c=c)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)

    plt.show()


def register_husl_colormap(nsegs=90):

    import husl

    cdict = {'red':list(), 'green':list(), 'blue':list()}

    r0,g0,b0 = husl.husl_to_rgb(0, 99.0, 50)
    cdict['red'].append((0., r0, r0))
    cdict['blue'].append((0., b0, b0))
    cdict['green'].append((0., g0, g0))

    inc = 360. / nsegs
    for k in range(nsegs):
        d = (k+1)*inc
        x = (k+1) / float(nsegs)
        r,g,b = husl.husl_to_rgb(d, 99.0, 50.0)
        cdict['red'].append((x, r, r))
        cdict['green'].append((x, g, g))
        cdict['blue'].append((x, b, b))

    cm = LinearSegmentedColormap('HUSL', cdict)
    plt.register_cmap(cmap=cm)


def custom_legend(colors, labels, linestyles=None):
    """ Creates a list of matplotlib Patch objects that can be passed to the legend(...) function to create a custom
        legend.

    :param colors: A list of colors, one for each entry in the legend. You can also include a linestyle, for example: 'k--'
    :param labels:  A list of labels, one for each entry in the legend.
    """

    if linestyles is not None:
        assert len(linestyles) == len(colors), "Length of linestyles must match length of colors."

    h = list()
    for k,(c,l) in enumerate(zip(colors, labels)):
        clr = c
        ls = 'solid'
        if linestyles is not None:
            ls = linestyles[k]
        patch = patches.Patch(color=clr, label=l, linestyle=ls)
        h.append(patch)
    return h


def grouped_boxplot(data, group_names=None, subgroup_names=None, ax=None, subgroup_colors=None,
                    box_width=0.6, box_spacing=1.0, legend_loc='upper right'):
    """ Draws a grouped boxplot. The data should be organized in a hierarchy, where there are multiple
        subgroups for each main group.

    :param data: A dictionary of length equal to the number of the groups. The key should be the
                group name, the value should be a list of arrays. The length of the list should be
                equal to the number of subgroups.
    :param group_names: (Optional) The group names, should be the same as data.keys(), but can be ordered.
    :param subgroup_names: (Optional) Names of the subgroups.
    :param subgroup_colors: A list specifying the plot color for each subgroup.
    :param ax: (Optional) The axis to plot on.
    """

    if group_names is None:
        group_names = data.keys()

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    nsubgroups = np.array([len(v) for v in data.values()])
    assert len(np.unique(nsubgroups)) == 1, "Number of subgroups for each property differ!"
    nsubgroups = nsubgroups[0]

    if subgroup_colors is None:
        subgroup_colors = list()
        for k in range(nsubgroups):
            subgroup_colors.append(np.random.rand(3))
    else:
        assert len(subgroup_colors) == nsubgroups, "subgroup_colors length must match number of subgroups (%d)" % nsubgroups

    def _decorate_box(_bp, _d):
        plt.setp(_bp['boxes'], lw=0, color='k')
        plt.setp(_bp['whiskers'], lw=3.0, color='k')

        # fill in each box with a color
        assert len(_bp['boxes']) == nsubgroups
        for _k,_box in enumerate(_bp['boxes']):
            _boxX = list()
            _boxY = list()
            for _j in range(5):
                _boxX.append(_box.get_xdata()[_j])
                _boxY.append(_box.get_ydata()[_j])
            _boxCoords = zip(_boxX, _boxY)
            _boxPolygon = plt.Polygon(_boxCoords, facecolor=subgroup_colors[_k])
            ax.add_patch(_boxPolygon)

        # draw a black line for the median
        for _k,_med in enumerate(_bp['medians']):
            _medianX = list()
            _medianY = list()
            for _j in range(2):
                _medianX.append(_med.get_xdata()[_j])
                _medianY.append(_med.get_ydata()[_j])
                plt.plot(_medianX, _medianY, 'k', linewidth=3.0)

            # draw a black asterisk for the mean
            plt.plot([np.mean(_med.get_xdata())], [np.mean(_d[_k])], color='w', marker='*',
                      markeredgecolor='k', markersize=12)

    cpos = 1
    label_pos = list()
    for k in group_names:
        d = data[k]
        nsubgroups = len(d)
        pos = np.arange(nsubgroups) + cpos
        label_pos.append(pos.mean())
        bp = plt.boxplot(d, positions=pos, widths=box_width)
        _decorate_box(bp, d)
        cpos += nsubgroups + box_spacing

    plt.xlim(0, cpos-1)
    plt.xticks(label_pos, group_names)

    if subgroup_names is not None:
        leg = custom_legend(subgroup_colors, subgroup_names)
        plt.legend(handles=leg, loc=legend_loc)


def boxplot_with_colors(data, group_names=None,ax=None, group_colors=None, box_width=0.6, box_spacing=1.0, box_alpha=1.):

    assert isinstance(data, dict)

    if group_names is None:
        group_names = data.keys()
    ngroups = len(data)

    if group_colors is None:
        group_colors = ['w']*ngroups

    if ax is None:
        ax = plt.gca()

    def _decorate_box(_bp, _d):

        plt.setp(_bp['boxes'], lw=0, color='k')
        plt.setp(_bp['whiskers'], lw=3.0, color='k')

        # fill in each box with a color
        assert len(_bp['boxes']) == ngroups
        for _k, _box in enumerate(_bp['boxes']):
            _boxX = list()
            _boxY = list()
            for _j in range(5):
                _boxX.append(_box.get_xdata()[_j])
                _boxY.append(_box.get_ydata()[_j])
            _boxCoords = zip(_boxX, _boxY)
            _boxPolygon = plt.Polygon(_boxCoords, facecolor=group_colors[_k], alpha=box_alpha)
            ax.add_patch(_boxPolygon)

        # draw a black line for the median
        for _k, _med in enumerate(_bp['medians']):
            _medianX = list()
            _medianY = list()
            for _j in range(2):
                _medianX.append(_med.get_xdata()[_j])
                _medianY.append(_med.get_ydata()[_j])
                plt.plot(_medianX, _medianY, 'k', linewidth=3.0)

            # draw a black asterisk for the mean
            plt.plot([np.mean(_med.get_xdata())], [np.mean(_d[_k])], color='w', marker='*',
                     markeredgecolor='k', markersize=12)

    pos = np.arange(ngroups) + 1
    d = [data[gname] for gname in group_names]
    bp = plt.boxplot(d, positions=pos, widths=box_width)
    _decorate_box(bp, d)

    plt.xlim(0.5, pos.max() + 0.5)
    plt.xticks(pos, group_names)


def compute_mean_from_scatter(x, y, bins=20, num_smooth_points=0, bin_by_quantile=False):
    assert len(x) == len(y)

    xcenter = np.zeros([bins])
    ymean = np.zeros([bins])
    yerr = np.zeros([bins])

    # bin the data, compute the mean and standard error of y for each bin
    if bin_by_quantile:
        p = 100. / (bins + 1)
        hist_edges = list([np.percentile(x, int((k + 1) * p)) for k in range(bins-1)])
        if len(np.unique(hist_edges)) != len(hist_edges):
            '[compute_mean_from_scatter] number of unique quantiles is less than the number of bins, defaulting to use hist'
            hist, hist_edges = np.histogram(x, bins=bins)
        else:
            hist_edges.insert(0, x.min())
            hist_edges.append(x.max())
            hist = None # unused
    else:
        hist, hist_edges = np.histogram(x, bins=bins)

    for k in range(bins):
        start = hist_edges[k]
        end = hist_edges[k + 1]
        i = (x >= start) & (x < end)

        xcenter[k] = ((end - start) / 2.) + start
        # print 'k=%d, start=%f, end=%f, xcenter=%f' % (k, start, end, xcenter[k])
        ymean[k] = y[i].mean()
        yerr[k] = y[i].std(ddof=1) / np.sqrt(i.sum())

    ymean_cs = None
    if num_smooth_points > 0:
        # interpolate the mean and sd with a cubic spline and resample
        x_rs = np.linspace(xcenter.min(), xcenter.max(), num_smooth_points)
        ymean_cs = interp1d(xcenter, ymean, kind='cubic')
        yerr_cs = interp1d(xcenter, yerr, kind='cubic')

        xcenter = x_rs
        ymean = ymean_cs(x_rs)
        yerr = yerr_cs(x_rs)

    return xcenter, ymean, yerr, ymean_cs


def plot_mean_from_scatter(x, y, bins=20, num_smooth_points=0,
                           color='k', ecolor='#D8D8D8', linewidth=4., elinewidth=3.0, alpha=0.5,
                           bin_by_quantile=False):
    """ For scatterplot data x,y, bin x, and plot the mean and standard error for each bin with respect to y. """

    xcenter,ymean,yerr,ymean_cs = compute_mean_from_scatter(x, y, bins, num_smooth_points, bin_by_quantile=bin_by_quantile)
    plt.errorbar(xcenter, ymean, yerr=yerr, c=color, linewidth=linewidth, elinewidth=elinewidth,
                 ecolor=ecolor, alpha=alpha, capthick=0.)


def plot_x_samps(x, y=0.):
    """ Plot ticks along the x axis at the bottom of a plot, one tick per sample. """

    plt.plot()


if __name__ == '__main__':

    # draw_husl_circle()

    data = { 'A':[np.random.randn(100), np.random.randn(100) + 5],
             'B':[np.random.randn(100)+1, np.random.randn(100) + 9],
             'C':[np.random.randn(100)-3, np.random.randn(100) -5]
           }

    grouped_boxplot(data, group_names=['A', 'B', 'C'], subgroup_names=['Apples', 'Oranges'], subgroup_colors=['#D02D2E', '#D67700'])
    plt.show()


def plot_binary_matrix(t, B, ax=None):
    """ Plot a binary matrix using patches, to make sure nothing disappears due to interpolation...

    :param t: The timestamp for each column of B, of length num_time_points
    :param B: The binary matrix of shape (ncols, num_time_points)
    :param ax: The axis to plot on (default=current axis)
    :return:
    """

    if ax is not None:
        plt.sca(ax)

    # first plot grid



    # now plot rectangles for each event





