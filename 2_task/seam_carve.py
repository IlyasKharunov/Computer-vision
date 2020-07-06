import numpy as np
from math import sqrt
def seam_carve(imag, mode, mask=None):
    img = np.copy(imag)
    lenn = np.shape(img)[0]
    wid = np.shape(img)[1]
    retmask = np.ndarray([])
    if mask is None:
        f = 1
    else:
        f = 0
        maskn = np.zeros((lenn,wid)) + mask
    bright = 0.299*img[:, :, 0] + 0.587*img[:, :,1] + 0.114*img[:, :, 2]
    energy = np.zeros((lenn, wid)) 
    energy[1 : lenn - 1, 1 : wid - 1] = np.sqrt(
    (bright[2:, 1:wid - 1] - bright[0:lenn - 2, 1:wid - 1])**2 +
    (bright[1 : lenn -1, 2 :] - bright[1: lenn - 1, 0 : wid - 2])**2)
    energy[0,0] = sqrt((bright[0,1] - bright[0,0])**2 +
          (bright[1,0] - bright[0,0])**2)
    energy[lenn-1,0] = sqrt((bright[lenn-1,1] - bright[lenn-1,0])**2 + (bright[lenn-1,0] - bright[lenn-2,0])**2)
    energy[0,wid - 1] = sqrt((bright[0, wid - 1] - bright[0, wid - 2])**2 + (bright[1, wid - 1] - bright[0, wid - 1])**2)
    energy[lenn - 1, wid - 1] = sqrt((bright[lenn - 1, wid - 1] - bright[lenn - 1, wid - 2])**2 + (bright[lenn - 1, wid - 1] - bright[lenn - 2, wid - 1])**2)
    energy[0,1:wid-1] = np.sqrt((bright[1,1:wid-1]-bright[0, 1:wid-1])**2+(bright[0,2:] - bright[0, 0: wid-2])**2)
    energy[1:lenn-1,wid-1] = np.sqrt((bright[2:,wid-1]-bright[0: lenn-2,wid-1])**2+(bright[1:lenn-1,wid-1] - bright[1:lenn-1, wid-2])**2)
    energy[lenn-1,1:wid-1] = np.sqrt((bright[lenn-1,1:wid-1]-bright[lenn-2, 1:wid-1])**2+(bright[lenn-1,2:] - bright[lenn-1, 0: wid-2])**2)
    energy[1:lenn-1,0] = np.sqrt((bright[2:,0]-bright[0: lenn-2,0])**2+(bright[1:lenn-1,1] - bright[1:lenn-1, 0])**2)
    
    if f == 0 :
        energy = energy + (maskn*lenn*wid*256) 
    seams = np.zeros((lenn, wid)) 
    m = mode.split()
    
    if m[0] == 'horizontal':
        seams[0,:] = energy[0,:]
        for i in range(1, lenn):
            for j in range(wid):
                if ((j+1)< wid):
                    if ((j-1)>=0 ):    
                        minn = min(seams[i-1, j-1], seams[i-1,j], seams[i-1,j+1])
                        if minn == seams[i-1, j+1]:
                            seams[i,j] = energy[i,j] + seams[i-1,j+1]
                        if minn == seams[i-1,j]:
                            seams [i,j] = energy[i,j] + seams[i-1,j]
                        if minn ==  seams[i-1, j-1] :
                            seams[i,j] = energy[i,j] + seams[i-1,j-1]
                    else:
                        minn = min(seams[i-1,j], seams[i-1,j+1])
                        if minn == seams[i-1,j+1]:
                            seams[i,j] = energy[i,j] + seams[i-1,j+1]
                        if minn == seams[i-1,j]:
                            seams [i,j] = energy[i,j] + seams[i-1,j]
                else:
                        minn = min(seams[i-1, j-1], seams[i-1,j])
                        if minn == seams[i-1,j]:
                            seams [i,j] = energy[i,j] + seams[i-1,j]
                        if minn ==  seams[i-1, j-1]:
                            seams[i,j] = energy[i,j] + seams[i-1,j-1]
        
    else:
        seams[:,0] = energy[:,0]
        
        for i in range(1, wid):
            
            for j in range(lenn):
                
                if ((j-1)>=0 ):
                    if ((j+1)< lenn):
            
                        minn = min(seams[j-1, i-1], seams[j,i-1], seams[j+1,i-1])
                        
                        if minn == seams[j+1, i-1]:
                            seams[j,i] = energy[j,i] + seams[j+1,i-1]
                            
                        if minn == seams[j,i-1]:
                            seams [j,i] = energy[j,i] + seams[j,i-1]
                            
                        if minn ==  seams[j-1, i-1] :
                            seams[j,i] = energy[j,i] + seams[j-1,i-1]
                            
                    else:
                        minn = min(seams[j-1, i-1], seams[j,i-1])
                        
                        if minn == seams[j,i-1]:
                            seams [j,i] = energy[j,i] + seams[j,i-1]
                            
                        if minn ==  seams[j-1, i-1]:
                            seams[j,i] = energy[j,i] + seams[j-1,i-1]
                            
                else:
                    minn = min(seams[j,i-1], seams[j+1,i-1])
                    
                    if minn == seams[j+1,i-1]:
                        seams[j,i] = energy[j,i] + seams[j+1,i-1]
                        
                    if minn == seams[j,i-1]:
                        seams[j,i] = energy[j,i] + seams[j,i-1]
                        
    seamcoord = [np.array([0]),np.array([0])]
    
    if m[1] == 'shrink':
        
        if m[0] == 'horizontal':
            seamcoord[1][0] = np.argmin(seams[lenn-1, :])
            seamcoord[0][0] = lenn-1
            
            for i in range((lenn-2),-1,-1):
                pil = seamcoord[1][lenn-2-i]
                seamcoord[0] = np.append(seamcoord[0],i)
                if ((pil+1) < wid) :
                    if ((pil-1) >=0):
                        minn = min(seams[i,pil+1],seams[i,pil],seams[i,pil-1])
                        if seams[i,pil +1] == minn:
                            pil1 = pil+1
                        if seams[i,pil] == minn:
                            pil1 = pil
                        if seams[i,pil-1] == minn:
                            pil1 = pil-1
                    else:
                        minn = min(seams[i,pil+1],seams[i,pil])
                        if seams[i,pil+1] == minn:
                            pil1 = pil+1
                        if seams[i,pil] == minn:
                            pil1 = pil
                else:
                    minn = min(seams[i,pil],seams[i,pil-1])
                    if seams[i,pil] == minn:
                            pil1 = pil
                    if seams[i,pil-1] == minn:
                            pil1 = pil-1
                if (seams[i,pil1]+energy[i+1,pil] == seams[i+1,pil]):
                    seamcoord[1] = np.append(seamcoord[1], pil1)
                
            retseamask = np.zeros((lenn,wid), bool)
            retseamask[seamcoord[0],seamcoord[1]] = True
            seamask = np.logical_not(retseamask)
            retim = img[seamask].reshape((lenn,wid-1,3))            
            if f == 0:
                retmask = maskn[seamask].reshape((lenn,wid-1))
                
        else:
            seamcoord[0][0] = np.argmin(seams[:,wid - 1])
            seamcoord[1][0] = wid-1
            
            for j in range((wid-2),-1,-1):
                row = seamcoord[0][wid-2-j]
                seamcoord[1] = np.append(seamcoord[1], j)
                
                if row+1 < lenn :
                    if row-1 >=0:
                        if seams[row+1,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row+1
                        if seams[row,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row
                        if seams[row-1,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row-1
                    else:
                        if seams[row+1,j] == min(seams[row+1,j],seams[row,j]):
                            row1 = row+1
                        if seams[row,j] == min(seams[row+1,j],seams[row,j]):
                            row1 = row
                else:
                    if seams[row,j] == min(seams[row,j],seams[row-1,j]):
                            row1 = row
                    if seams[row-1,j] == min(seams[row,j],seams[row-1,j]):
                            row1 = row-1
                            
                seamcoord[0] = np.append(seamcoord[0], row1)
            retseamask = np.zeros((lenn,wid), bool)
            retseamask[seamcoord[0],seamcoord[1]] = True
            seamask = np.logical_not(retseamask)
            retim = img[seamask].reshape((lenn-1,wid,3))
            
            if f == 0:
                retmask = maskn[seamask].reshape((lenn-1,wid))
                
    else:
        
        if m[0] == 'horizontal':
            seamcoord[1][0] = np.argmin(seams[lenn-1, :])
            seamcoord[0][0] = lenn-1
            
            for i in range((lenn-2),-1,-1):
                pil = seamcoord[1][lenn-2-i]
                seamcoord[0] = np.append(seamcoord[0], i)
                
                if pil+1 <= wid - 1 :
                    if pil-1 >=0:
                        if seams[i,pil +1] == min(seams[i,pil+1],seams[i,pil],seams[i,pil-1]):
                            pil1 = pil+1
                        if seams[i,pil] == min(seams[i,pil+1],seams[i,pil],seams[i,pil-1]):
                            pil1 = pil
                        if seams[i,pil-1] == min(seams[i,pil+1],seams[i,pil],seams[i,pil-1]):
                            pil1 = pil-1
                    else:
                        if seams[i,pil+1] == min(seams[i,pil+1],seams[i,pil]):
                            pil1 = pil+1
                        if seams[i,pil] == min(seams[i,pil+1],seams[i,pil]):
                            pil1 = pil
                else:
                    if seams[i,pil] == min(seams[i,pil],seams[i,pil-1]):
                            pil1 = pil
                    if seams[i,pil-1] == min(seams[i,pil],seams[i,pil-1]):
                            pil1 = pil-1
                seamcoord[1] = np.append(seamcoord[1], pil1)
            retseamask = (energy != energy)
            retseamask[seamcoord[0], seamcoord[1]] = True
            
            if np.max(seamcoord[1])+1 < wid :
                mid1 = img[seamcoord[0], seamcoord[1]]/2
                mid2 = img[seamcoord[0], (seamcoord[1]+1)]/2
                mid = mid1 + mid2
            else:
                mid1 = img[seamcoord[0], seamcoord[1]]/2
                mid2 = img[seamcoord[0], (seamcoord[1]-1)]/2
                mid = mid1 + mid2
            img = img.reshape((lenn*wid,3))
            if np.max(seamcoord[1])+1 < wid :
                img = np.insert(img,(wid*seamcoord[0] + (seamcoord[1]+1)), mid, axis = 0)
            else:
                img = np.insert(img,(wid*seamcoord[0] + (seamcoord[1]-1)), mid, axis = 0)
            retim = img.reshape((lenn,wid+1,3))
            
            if f == 0:
                if np.max(seamcoord[1])+1 < wid :
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = np.insert(maskn,(wid*seamcoord[0] + (seamcoord[1]+1)), 0)
                else:
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = np.insert(maskn,(wid*seamcoord[0] + (seamcoord[1]-1)), 0)
                retmask = maskn.reshape((lenn,wid+1))
            else:
                maskn = np.zeros((lenn,wid))
                if np.max(seamcoord[1])+1 < wid :
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = np.insert(maskn,(wid*seamcoord[0] + (seamcoord[1]+1)), 0)
                else:
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = np.insert(maskn,(wid*seamcoord[0] + (seamcoord[1]-1)), 0)
                retmask = maskn.reshape((lenn,wid+1))
                
        else:
            seamcoord[0][0] = np.argmin(seams[:,wid - 1])
            seamcoord[1][0] = wid-1
            
            for j in range((wid-2),-1,-1):
                row = seamcoord[0][wid-2-i]
                seamcoord[1] = np.append(seamcoord[1], j)
                
                if row+1 < lenn :
                    if row-1 >=0:
                        if seams[row+1,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row+1
                        if seams[row,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row
                        if seams[row-1,j] == min(seams[row+1,j],seams[row,j],seams[row-1,j]):
                            row1 = row-1
                    else:
                        if seams[row+1,j] == min(seams[row+1,j],seams[row,j]):
                            row1 = row+1
                        if seams[row,j] == min(seams[row+1,j],seams[row,j]):
                            row1 = row
                else:
                    if seams[row,j] == min(seams[row,j],seams[row-1,j]):
                            row1 = row
                    if seams[row-1,j] == min(seams[row,j],seams[row-1,j]):
                            row1 = row-1
                seamcoord[0] = np.append(seamcoord[0], row1)
                
            retseamask = (energy != energy)
            retseamask[seamcoord[0],seamcoord[1]] = True
            
            if np.max(seamcoord[0])+1 < lenn :
                mid1 = img[seamcoord[0], seamcoord[1]]/2
                mid2 = img[seamcoord[0]+1, seamcoord[1]]/2
                mid = mid1 + mid2
            else:
                mid1 = img[seamcoord[0], seamcoord[1]]/2
                mid2 = img[seamcoord[0]-1, seamcoord[1]]/2
                mid = mid1 + mid2
            img = img.transpose((1,0,2))
            img = img.reshape((lenn*wid,3))
            if np.max(seamcoord[0])+1 < lenn :
                img = np.insert(img,(lenn*seamcoord[1] + seamcoord[0]+1), mid, axis = 0)
            else:
                img = np.insert(img,(lenn*seamcoord[1] + seamcoord[0]-1), mid, axis = 0)
            
            retim = img.reshape((wid,lenn+1,3))
            retim = retim.transpose((1,0,2))
            
            if f == 0:
                if np.max(seamcoord[0])+1 < lenn :
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = maskn.transpose()
                    maskn = np.insert(maskn,(lenn*seamcoord[1] + seamcoord[0]+1), 0)
                else:
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn.transpose()
                    maskn = np.insert(maskn,(lenn*seamcoord[1] + seamcoord[0]-1), 0)
                retmask = maskn.reshape((wid,lenn+1))
                retmask = retmask.transpose()
            else:
                maskn = np.zeros((lenn,wid))
                if np.max(seamcoord[0])+1 < lenn :
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn = maskn.transpose()
                    maskn = np.insert(maskn,(lenn*seamcoord[1] + seamcoord[0]+1), 0)
                else:
                    maskn[seamcoord[0],seamcoord[1]] = 1
                    maskn.transpose()
                    maskn = np.insert(maskn,(lenn*seamcoord[1] + seamcoord[0]-1), 0)
                retmask = maskn.reshape((wid,lenn+1))
                retmask = retmask.transpose()
    return retim, retmask, retseamask