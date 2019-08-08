
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # for kmeans



def todo_equalization(imrgb1,imrgb2):

    imgray1 = imrgb1[:,:,0]
    imgray2 = imrgb2[:,:,0]
    [nrow,ncol] = imgray1.shape
        
    imhisto1,bins= np.histogram(imgray1, range=(0,1), bins = 256)
    imhistocum1 = np.cumsum(imhisto1) 
    imhisto2,bins= np.histogram(imgray2, range=(0,1), bins = 256)
    imhistocum2 = np.cumsum(imhisto2) 

    imeq1 = imhistocum1[np.uint8(imgray1*255)]
    imeq1=imeq1.reshape(nrow,ncol)
    imeq2 = imhistocum2[np.uint8(imgray2*255)]
    imeq2=imeq2.reshape(nrow,ncol)

    #Display images
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 5))
    axes[0].set_title('Bright image')
    axes[0].imshow(imrgb1)
    axes[1].set_title('Bright image equalized')
    axes[1].imshow(imeq1,cmap = 'gray')
    axes[2].set_title('Dark image')
    axes[2].imshow(imrgb2)
    axes[3].set_title('Dark image equalized')
    axes[3].imshow(imeq2,cmap = 'gray')
    fig.tight_layout()
    plt.show()
 
def todo_noise_histograms(imgray):

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
    for k in range(4):
        sigma = k*20/255
        imnoise = imgray+sigma*np.random.randn(imgray.shape[0],imgray.shape[1])
        imhisto, bins = np.histogram(imnoise, range=(0,1), bins = 256) 
        axes[k,0].imshow(imnoise,cmap='gray')
        axes[k,1].bar(np.arange(0,256),imhisto)
        

def todo_lloyd_max(imgray):
    K        = 10 # number of classes
    X        = imgray.reshape((imgray.shape[0]*imgray.shape[1],1))
    clusters = KMeans(n_clusters=K).fit_predict(X)
    mu       = np.zeros((K,1))
    for k in range(K):
        mu[k] = np.mean((X[clusters==k]))

    plt.figure(figsize=(7, 7))
    plt.title('Lloyd-Max quantization with 10 levels')
    plt.imshow(mu[clusters].reshape((imgray.shape[0],imgray.shape[1])),cmap='gray')


def todo_dithering(imgray):
    imgray = imgray+5/255*np.random.randn(imgray.shape[0],imgray.shape[1])   
    K        = 10 # number of classes
    X        = imgray.reshape((imgray.shape[0]*imgray.shape[1],1))
    clusters = KMeans(n_clusters=K).fit_predict(X)
    mu       = np.zeros((K,1))
    for k in range(K):
        mu[k] = np.mean((X[clusters==k]))

    plt.figure(figsize=(7, 7))
    plt.title('Lloyd-Max quantization with 10 levels with dithering (Gaussian noise of std 5/255)')
    plt.imshow(mu[clusters].reshape((imgray.shape[0],imgray.shape[1])),cmap='gray')
