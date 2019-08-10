import numpy as np
import scipy.signal as scs

def todo_specification_separate_channels(u,v):
    nrowu,ncolu,nchu = u.shape
    w = np.zeros(u.shape)
    for i in range(3):
        uch = u[:,:,i]
        vch = v[:,:,i]
        u_sort,index_u=np.sort(uch,axis=None),np.argsort(uch,axis=None)
        v_sort,index_v=np.sort(vch,axis=None),np.argsort(vch,axis=None)
        uspecifv= np.zeros(nrowu*ncolu)
        uspecifv[index_u] = v_sort
        uspecifv = uspecifv.reshape(nrowu,ncolu)   
        w[:,:,i] = uspecifv.reshape(nrowu,ncolu)
    return w


def transport1D(X,Y):
    sx = np.argsort(X) #argsort retourne les indices des valeurs s'ils étaient ordonnés par ordre croissant   
    sy = np.argsort(Y)
    return((sx,sy)) 

def todo_transport3D(X,Y,N,e): #X,y,Z are nx3 matrices
    Z=np.copy(X) # output
    for k in range(N):
        u=np.random.randn(3,3)
        q=np.linalg.qr(u)[0] #orthonormal basis with uniform distibution on the sphere 
        for i in range(3):
            # projection on the basis 
            Yt=np.dot(Y,q[:,i])
            Zt=np.dot(Z,q[:,i])
            #Permutations
            [sZ,sY]=transport1D(Zt,Yt)
            for j in range(X.shape[0]):
                Z[sZ[j],:]=Z[sZ[j],:]+e*(Yt[sY[j]]-Zt[sZ[j]])*(q[:,i]) #transport 3D
        
    return Z,sZ,sY


def todo_guided_filter(u,guide,r,eps):
    phi         = np.ones((2*r+1,2*r+1))/(2*r+1)**2
    C           = scs.convolve2d(np.ones(u.shape), phi, mode='same')   # to avoid image edges pb 
    mean_u      = scs.convolve2d(u, phi, mode='same')/C
    mean_guide  = scs.convolve2d(guide, phi, mode='same')/C
    corr_guide  = scs.convolve2d(guide*guide, phi, mode='same')/C
    corr_uguide = scs.convolve2d(u*guide, phi, mode='same')/C
    var_guide   = corr_guide - mean_guide * mean_guide
    cov_uguide  = corr_uguide - mean_u * mean_guide

    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide

    mean_alph = scs.convolve2d(alph, phi, mode='same')/C
    mean_beta = scs.convolve2d(beta, phi, mode='same')/C

    q = mean_alph * guide + mean_beta
    return q
