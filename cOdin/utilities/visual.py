import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def correlogram1(data, title = "Correlogram"):
    plt.figure(title, figsize=(12, 6))
    sb.heatmap(data=np.round(data, 2), vmin=-1, vmax=1, cmap='bwr', annot=True)
    plt.title(title)
    plt.show()


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

# copiat
def eighenValues(alpha):
    plt.figure("Eigenvalues - Variance of the Components", figsize=(11, 7))

    # one line, one column, index=1
    # f1 = f.add_subplot(1, 1, 1, title="Eigenvalues - Variance of the Components",
    #                    xlabel="Components", ylabel="Eigenvalues")

    plt.title("Eigenvalues - Variance of the Components")
    plt.xlabel("Components")
    plt.ylabel("Eigenvalues")
    plt.plot(alpha, 'bo-')
    plt.xticks([k for k in range(len(alpha))])
    plt.axhline(1, color='r')
    plt.show()

# copiat
def corrCircles(R, k1, k2, title="The Correlation Circles"):
    plt.figure(title, figsize=(6, 6))
    plt.title(title, fontsize=16, color='b', verticalalignment='bottom')
    T =[t for t in np.arange(0, np.math.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(0, color='g')
    plt.axvline(0, color='g')
    plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r')
    plt.xlabel(R.columns[k1], fontsize=12, color='r', verticalalignment='top')
    plt.ylabel(R.columns[k2], fontsize=12, color='r', verticalalignment='bottom')
    for i in range(len(R)):
        plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    plt.show()