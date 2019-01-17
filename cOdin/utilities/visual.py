# data visualization functionalities, such as scatter plot, line plot, correlation circle, heatmap, treemap, dendrogram etc. (1 points);

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import scipy.cluster.hierarchy as hiclu

def correlogram2(t, title=None, valmin=-1, valmax=1):
    f = plt.figure(title, figsize=(8, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    sb.heatmap(np.round(t,2), cmap='bwr', vmin=valmin, vmax=valmax, annot=True)

def correlogram(df, isCorrMatrix = True, title = "Correlogram", dropDuplicates = True):

    # Your dataset is already a correlation matrix.
    # If you have a dataset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    if not isCorrMatrix:
        df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sb.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Set up title
    plt.title(title)

    # Add diverging colormap from red to blue
    cmap = sb.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sb.heatmap(df, mask=mask, cmap=cmap,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sb.heatmap(df, cmap=cmap,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)

# Function to plot eigenvalues
def eighenValues(alpha):
    plt.figure("Eigenvalues - Variance of the Components", figsize=(11, 7))

    # one line, one column, index=1
    # f1 = f.add_subplot(1, 1, 1, title="Eigenvalues - Variance of the Components",
    #                    xlabel="Components", ylabel="Eigenvalues")

    plt.title("Eigenvalues")
    plt.xlabel("Components")
    plt.ylabel("Eigenvalues")
    plt.plot(alpha, 'bo-')
    plt.xticks([k for k in range(len(alpha))])
    plt.axhline(1, color='c')
    plt.show()

# Function to plot the Correlation Circles
def corrCircles(R, k1, k2, title="The Correlation Circles"):
    plt.figure(title, figsize=(6, 6))
    plt.title(title, fontsize=16, color='b', verticalalignment='bottom')
    T =[t for t in np.arange(0, np.math.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(0, color='y')
    plt.axvline(0, color='y')
    plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='b')
    plt.xlabel(R.columns[k1], fontsize=12, color='b', verticalalignment='top')
    plt.ylabel(R.columns[k2], fontsize=12, color='b', verticalalignment='bottom')
    for i in range(len(R)):
        plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    plt.show()

# Function to plot a scatter diagram of the discriminants
def scatter_discriminant(z1, z2, g, labels, zg1, zg2, labels_g):
    f = plt.figure(figsize=(10, 7))
    assert isinstance(f, plt.Figure)
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Instances and centers in z1 and z2 axes ",fontsize=14,color='b')
    sb.scatterplot(z1, z2, g)
    sb.scatterplot(zg1, zg2, labels_g, s=200,legend=False)
    for i in range(len(labels)):
        ax.text(z1[i], z2[i], labels[i])
    for i in range(len(labels_g)):
        ax.text(zg1[i], zg2[i], labels_g[i], fontsize=26)
    plt.show()

# Function to plot the distribution of values
def distribution(z, y, g, axa):
    f = plt.figure(figsize=(10, 7))
    assert isinstance(f, plt.Figure)
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Group distribution. Axis " + str(axa + 1), fontsize=14, color='b')
    for v in g:
        sb.kdeplot(data=z[y == v], shade=True, ax=ax, label=v)
    plt.show()

# Function to plot the variance of values
def plot_variance(alpha, title='Variance plot'):
    n = len(alpha)
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xticks(np.arange(1, n + 1))
    f1.set_xlabel('Component', fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel('Variance', fontsize=12, color='r', verticalalignment='bottom')
    f1.plot(np.arange(1, n + 1), alpha, 'ro-')
    f1.axhline(1, c='g')
    j_Kaiser = np.where(alpha < 1)[0][0]
    eps = alpha[:n - 1] - alpha[1:]
    d = eps[:n - 2] - eps[1:]
    j_Cattel = np.where(d < 0)[0][0]
    f1.axhline(alpha[j_Cattel + 1], c='m')
    return j_Cattel + 2, j_Kaiser

def t_scatter(x, y, label=None, tx="", ty="", title='Scatterplot'):
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
    f1.scatter(x=x, y=y, c='r')
    if label is not None:
        n = len(label)
        for i in range(n):
            f1.text(x[i], y[i], label[i])

def t_scatter_s(x, y, x1, y1, label=None, label1=None, tx="", ty="", title='Scatterplot'):
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
    f1.scatter(x=x, y=y, c='r')
    f1.scatter(x=x1, y=y1, c='b')
    if label is not None:
        n = len(label);
        p = len(label1)
        for i in range(n):
            f1.text(x[i], y[i], label[i], color='k')
        for i in range(p):
            f1.text(x1[i], y1[i], label1[i], color='k')

def scatter(x, y, label=None, tx="", ty="", title='Scatterplot'):
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
    f1.scatter(x=x, y=y, c='r')
    if label is not None:
        n = len(label)
        for i in range(n):
            f1.text(x[i], y[i], label[i])

def dendrogram(h, labels, title='Hierarchical classification', threshold=None):
    f = plt.figure(figsize=(12, 7))
    axis = f.add_subplot(1, 1, 1)
    axis.set_title(title, fontsize=16, color='b')
    hiclu.dendrogram(h, labels=labels, leaf_rotation=30, ax=axis, color_threshold=threshold)
    plt.show()
