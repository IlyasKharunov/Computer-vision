def extract_hog(img):
    import numpy as np 
    from math import sqrt
    cell = 10
    blockwid = 3
    blocklen = 3
    angle = 9
    step = 2
    binan = 360/angle #корзинный угол
    lenn = np.shape(img)[0]
    wid = np.shape(img)[1]
    lenm = lenn % cell
    widm = wid % cell
    img = img[lenm//2:lenn - lenm+ lenm//2, widm//2:wid - widm + widm//2]
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
    phi = np.arctan2(iy,ix)*180/np.pi + 180 # считаем угол и приводим его к 0 360
    lln = np.sqrt((iy**2 + ix**2))# считаем длину 
    lln3d = np.empty((lenn,wid,angle))# нужен для линейной интерполяции
    for i in range(angle):# составляем массив длин 3хмерный 
        lln3d[:,:,i] = lln
    phimod = phi % binan# массив отклонений от корзинных углов
    m = np.zeros((lenn,wid,angle))# нужен для весов при углах для корзины 
    k = np.empty((lenn,cell,angle))# нужен для схлопки по гор
    hog = np.empty((cell,cell,angle))# схлопка по верти
    nbin = phimod/binan
    curbin = 1 - nbin
    mmask = phi//binan
    for j in range(angle):
        m[:,:,j] += np.where(mmask == j,curbin,0)
        m[:,:,((j+1) % angle)] += np.where(mmask == j,nbin,0)
    for i in range(cell):# схлопка по горизонтали 
        k[:,i,:] = np.sum(lln3d[:,cellwid*i:cellwid*(i+1),:]*m[:,cellwid*i:cellwid*(i+1),:],1)
    for h in range(cell):# составление хога
        hog[h,:,:] = np.sum(k[celllen*h:celllen*(h+1),:,:],0)
    vlen = angle*blocklen*blockwid
    bnuml = cell - blocklen +1
    bnumw = cell - blockwid +1
    feature = np.empty((vlen*(bnuml//step + (bnuml%step != 0))*(bnumw//step + (bnumw%step != 0))))
    cc = 0
    for i in range (0,cell - blocklen+1,step):# составление признака с помощью нормализованных блоков 
        for j in range(0,cell - blockwid+1,step):
            normalized_block = np.ravel(hog[i:i + blocklen , j:j + blockwid,:])
            feature[cc*vlen:(cc+1)*vlen] = normalized_block/(sqrt((np.sum(normalized_block**2) + 1e-9)))
            cc+=1
    return feature
def fit_and_classify(train_features, train_labels, test_features):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf',degree = 3, gamma=3e-2,C = 7.0, tol=1e-3)
    svm.fit(train_features, train_labels) 
    return svm.predict(test_features)