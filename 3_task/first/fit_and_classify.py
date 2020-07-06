def extract_hog(img):
    #from time import time 
    import numpy as np 
    from math import sqrt
    from skimage.transform import resize
    cell = 11
    blockwid = 3
    blocklen = 3
    bin = 9
    pixnum = 6
    step = 2
    angle = 360/bin #угол на корзину
    #lenn = np.shape(img)[0]
    #wid = np.shape(img)[1]
    #lenm = lenn % cell
    #widm = wid % cell
    newsize = pixnum*cell# новый размер картинки для удобства работы
    img = resize(img,((newsize),(newsize)),anti_aliasing = False)
    #img = img[lenm//2:lenn - lenm+ lenm//2, widm//2:wid - widm + widm//2]
    lenn = np.shape(img)[0]
    wid = np.shape(img)[1]
    cellwid = wid//cell
    celllen = lenn//cell
    bright = 0.299*img[:, :, 0] + 0.587*img[:, :,1] + 0.114*img[:, :, 2]#считаем яркость 
    ix = np.empty((lenn, wid)) # считаем градиент 
    ix[:,1:wid - 1] = bright[:, 2 :] - bright[:, 0 : wid - 2]
    ix[:,0] = bright[:,1] - bright[:, 0]
    ix[:,wid-1] = bright[:,wid-1] - bright[:, wid-2]
    iy = np.empty((lenn,wid))
    iy[1:lenn-1,:] = bright[2:,:] - bright[0 : lenn - 2,:]
    iy[0,:] = bright[1,:] - bright[0, :]
    iy[lenn - 1,:] = bright[lenn - 1,:] - bright[lenn - 2, :]
    phi = np.arctan2(iy,ix)*180/np.pi + 180 # считаем угол и приводим его к диапазону от 0 до 360
    gradlen = np.sqrt((iy**2 + ix**2))# считаем длину 
    gradlen3d = np.empty((lenn,wid,bin))# нужен для линейной интерполяции
    for i in range(bin):# составляем массив длин 3хмерный 
        gradlen3d[:,:,i] = gradlen
    phimod = phi % angle# массив отклонений от корзинных углов
    m = np.zeros((lenn,wid,bin))# нужен для весов при углах для корзины 
    k = np.empty((lenn,cell,bin))# нужен для схлопки по гор
    hog = np.empty((cell,cell,bin))# схлопка по верти
    nbin = phimod/angle
    curbin = 1 - nbin
    mmask = phi//angle
    for j in range(bin):
        m[:,:,j] += np.where(mmask == j,curbin,0)
        m[:,:,((j+1) % bin)] += np.where(mmask == j,nbin,0)
    for i in range(cell):# схлопка по горизонтали 
        k[:,i,:] = np.sum(gradlen3d[:,cellwid*i:cellwid*(i+1),:]*m[:,cellwid*i:cellwid*(i+1),:],1)
    for h in range(cell):# составление хога
        hog[h,:,:] = np.sum(k[celllen*h:celllen*(h+1),:,:],0)
    vlen = bin*blocklen*blockwid
    bnuml = cell - blocklen +1
    bnumw = cell - blockwid +1
    feature = np.empty((vlen*(bnuml//step + (bnuml%step != 0))*(bnumw//step + (bnumw%step != 0))))
    #feature = np.array([])
    cc = 0
    for i in range (0,cell - blocklen+1,step):# составление признака с помощью нормализованных блоков 
        for j in range(0,cell - blockwid+1,step):
            normalized_block = np.ravel(hog[i:i + blocklen , j:j + blockwid,:])
            feature[cc*vlen:(cc+1)*vlen] = normalized_block/(sqrt((np.sum(normalized_block**2) + 1e-9)))
            #feature = np.append(feature,normalized_block/(sqrt((np.sum(normalized_block**2) + 1e-9))))
            cc+=1
    return feature
def fit_and_classify(train_features, train_labels, test_features):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf',degree = 3, gamma=3e-2,C = 7.0, tol=1e-3)#3e-2
    #svm.fit(train_features, train_labels) 
    scores = cross_val_score(svm, train_features, train_labels, cv=5)
    print(scores)
    return svm.predict(test_features)