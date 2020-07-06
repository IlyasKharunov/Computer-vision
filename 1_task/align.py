import numpy as np
from skimage.transform import resize

def align(imagg, g_coord):
    imag = imagg[0 : np.shape(imagg)[0] - (np.shape(imagg)[0] % 3)]
    imag1 = imag[0 : len(imag)//3]
    imag2 = imag[len(imag)//3 : (len(imag)//3)*2]
    imag3 = imag[(len(imag)//3)*2 : len(imag)]
    imag1 = imag1[int(0.05 * np.shape(imag1)[0]) : int(0.95 * np.shape(imag1)[0])] 
    imag1 = imag1[:,int(0.05 * np.shape(imag1)[1]) : int(0.95 * np.shape(imag1)[1])]
    imag2 = imag2[int(0.05 * np.shape(imag2)[0]) : int(0.95 * np.shape(imag2)[0])] 
    imag2 = imag2[:,int(0.05 * np.shape(imag2)[1]) : int(0.95 * np.shape(imag2)[1])]
    imag3 = imag3[int(0.05 * np.shape(imag3)[0]) : int(0.95 * np.shape(imag3)[0])] 
    imag3 = imag3[:,int(0.05 * np.shape(imag3)[1]) : int(0.95 * np.shape(imag3)[1])]
    
    def offs(imag1, imag2, k, m):
        min = 260
        for i in range(-k,k+1):
            for j in range(-m,m+1):
                if (i>=0):
                    if(j>=0):
                        tmpim = imag1[0: len(imag1)-i, 0: np.shape(imag1)[1]-j] - imag2[i:, j:]
                        if ((np.sum(tmpim**2) / (np.shape(tmpim)[0] * np.shape(tmpim)[1])) < min):
                            min = np.sum(tmpim**2) / (np.shape(tmpim)[0] * np.shape(tmpim)[1])
                            offset = [i, j]
                    else:
                        tmpim = imag1[0: len(imag1)-i, abs(j):] - imag2[i:, 0: np.shape(imag2)[1]-abs(j)]
                        if ((np.sum(tmpim**2)/(np.shape(tmpim)[0] * np.shape(tmpim)[1])) < min):
                            min = np.sum(tmpim**2)/(np.shape(tmpim)[0] * np.shape(tmpim)[1])
                            offset = [i, j]
                else:
                    if(j>=0):
                        tmpim = imag1[abs(i):, 0: np.shape(imag1)[1]-j]-imag2[0: len(imag2)-abs(i), j:]
                        if ((np.sum(tmpim**2)/(np.shape(tmpim)[0] * np.shape(tmpim)[1])) < min):
                            min = np.sum(tmpim**2)/(np.shape(tmpim)[0]*np.shape(tmpim)[1])
                            offset = [i, j]
                    else:
                        tmpim = imag1[abs(i) :, abs(j) :]-imag2[0: len(imag2)-abs(i), 0: np.shape(imag2)[1] - abs(j)]
                        if ((np.sum(tmpim**2)/(np.shape(tmpim)[0] * np.shape(tmpim)[1])) < min):
                            min = np.sum(tmpim**2)/(np.shape(tmpim)[0] * np.shape(tmpim)[1])
                            offset = [i, j]
        return offset
    
    def shift(imag1, offset):
        if offset[0] >= 0 :
            if offset[1] >=0:
                tmm = imag1[0:len(imag1) - offset[0], 0: np.shape(imag1)[1] - offset[1]]
            else:
                tmm = imag1[0: len(imag1) - offset[0], abs(offset[1]):]
        else:
            if offset[1] >= 0:
                tmm = imag1[abs(offset[0]):, 0: np.shape(imag1)[1] - offset[1]]
            else:
                tmm = imag1[abs(offset[0]):, abs(offset[1]):]
        return tmm
    
    def recimgoffs(imag1, imag2):
        if (np.shape(imag1)[0]+np.shape(imag1)[1])//2 > 500 :
            off = recimgoffs(resize(imag1, (imag1.shape[0] // 2, imag1.shape[1] // 2), anti_aliasing = False), resize(imag2, (imag2.shape[0] // 2, imag2.shape[1] // 2), anti_aliasing = False))
            off[0] = off[0]*2
            off[1] = off[1]*2
            tmm1 = shift(imag1, off)
            tmm2 = shift(imag2, [-off[0],-off[1]])
            off1 = offs(tmm1, tmm2, 1, 1)
            off = [off[0]+off1[0],off[1]+off1[1]]
            return off
            
        else:
            return offs(imag1, imag2, 15, 15)
        
    if (np.shape(imag1)[0]+np.shape(imag1)[1])//2 > 500 :
        offset1 = recimgoffs(imag3, imag2)
        offset2 = recimgoffs(imag1, imag2)
    else:   
        offset1 = offs(imag3, imag2, 15, 15)
        offset2 = offs(imag1, imag2, 15, 15)
    
    red = (g_coord[0] + len(imag)//3 - offset1[0], g_coord[1] - offset1[1] )
    blue = (g_coord[0] - len(imag)//3 - offset2[0], g_coord[1] - offset2[1] )
    k = shift(shift(imag3, offset1), [-offset2[0], -offset2[1]])
    m = shift(shift(imag1, offset2), [-offset1[0], -offset1[1]])
    l = shift(shift(imag2, [-offset1[0], -offset1[1]]), [-offset2[0], -offset2[1]])
    k = k[:, :, np.newaxis ]
    l = l[:, :, np.newaxis]
    m = m[:, :, np.newaxis]
    return np.concatenate((k,l,m), axis = -1), blue, red