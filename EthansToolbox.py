#wanted to make my own toolbox for simplicity

import time
import numpy as np
import numpy.ma as ma
import naming


#this one is for fitting polynomials I think
def lspolyfit(myTime, data, order, mode, thresholdMask=None,
              firstNan=None, fitColumnIndex=None,
              fitColumnMask=None,
              full=False, alpha=0.95,
              tryLowerOrders=False):
  
    if thresholdMask is None:
        # well use all data points regardless
        thresholdMask = np.zeros(data.shape, dtype=np.int8)
        firstNan = np.int64(np.ones(data.shape[1])*len(myTime))

    if mode == naming.FAST_CAMERA_MODE:
        # fast mode does not use the first frame for fit
        data = data[1:, :]
        # time vector is shorter but must maintain the values
        orgMyTime = myTime
        myTime = myTime[1:]
    else:
        orgMyTime = myTime

    # masked arrays can be multi dimensional
    coef = np.zeros((order+1, data.shape[1]))
    coefMask = np.ones((order+1, data.shape[1]))
    if full:
        # diagonal has same mask as coef
        myDiag = np.zeros((order+1, data.shape[1]))
        lengthVec = np.zeros(data.shape[1])

    if not tryLowerOrders:
        # Vandermonde matrices for legth N of time series are
        # just the first N lines of the full matrix
        van = np.vander(myTime, order+1)

        # first list entry contains array with all columns
        # that can fit full order
        # unmask everything that can do full order
        coefMask[:, fitColumnIndex[0]] = 0

        # now we need to loop through all the possible lengths
        # of time vectors
        goodLengths = np.int8(np.unique(firstNan[fitColumnIndex[0]]))

        # print(goodLengths)
        for k in goodLengths:
            # print('looping through all good lengths', k)
            tmpMask = np.where(firstNan == k, np.zeros(firstNan.shape, dtype=np.int8), 1)
            # combine threshold and fitColumnMask
            tmpMask = thresholdMask+tmpMask
            tmpMask[tmpMask == 2] = 1
            # mask everything that cannot do full order
            yMa = ma.array(data, mask=tmpMask)
            cov = np.linalg.inv(np.dot(van[:k].T, van[:k]))
            coef = coef + ma.dot(cov, ma.dot(van[:k].T, yMa[:k])).data
            if full:
                tmpInd = np.where(firstNan == k)[0]
                myDiag[:, tmpInd] = np.repeat(np.diag(cov)[:, None], len(tmpInd), axis=1)
                lengthVec[tmpInd] = k

        # masked arrays can be multi dimensional
        coef = ma.array(coef, mask=coefMask)

        if not full:
            return coef
        myDiag = ma.array(myDiag, mask=coefMask)
        lengthVec = ma.array(lengthVec, mask=fitColumnMask[0])
        dataMask = thresholdMask + fitColumnMask[0]
        dataMask[dataMask == 2] = 1
        yMa = ma.array(data, mask=dataMask)

        # covariance matrix for parameters is (X.T X)-1 * sig2
        # where sig2 is variance of noise
        # use variance of residuals to estimate sig2
        # n number of points, p degree of polynomial + 1
        # sig2 = 1/(n-p)*sum(res_i^2)
        # coeffsig_i = c*sqrt(sig2*diag((X.T X)-1)_ii)
        # estimating variance for fit parameters
        fitX = np.tile(myTime, (yMa.shape[1], 1))
        fit = np.polyval(coef, fitX.T)
        res = (yMa - fit)
        if mode == naming.FAST_CAMERA_MODE:
            # for fast mode redo the fit so it matches input data shape
            # and add zeros to residuals
            fitX = np.tile(orgMyTime, (yMa.shape[1], 1))
            fit = np.polyval(coef, fitX.T)
            res = res.insert(res, 0, axis=0)
        yRes = yMa - ma.mean(yMa)
        # lengthVec - order -1 is effective degree of freedom
        effDf = lengthVec -(order+1)
        rSquared = 1 - (ma.sum(res**2, axis=0))/(ma.sum(yRes**2, axis=0))
        # r2Adj = 1 - (1-r2) * (lengthVec -1)/(effDf - 1)
        sig2 = 1/(effDf)*ma.sum(res**2, axis=0)
        tValue = stats.t.ppf(alpha, effDf)
        tValue = ma.array(tValue, mask=fitColumnMask[0])
        coefSig = tValue*ma.sqrt(ma.multiply(myDiag, sig2))
        # tSquared = ma.divide(coef**2, (coefSig/tValue)**2)
        # significanHighestOrder = tSquared[0] < tValue**2
        # print(tValue**2)
        # print(tSquared[0])
        # print(significanHighestOrder)
        return coef, coefSig, fit, res, sig2, rSquared

    # else try for all other orders
    # Vandermonde matrices for legth N of time series are
    # just the first N lines of the full matrix
    # we need different matrices for different lower orders
    van = []
    for i in range(order, 0, -1):
        print(i)
        van.append(np.vander(myTime, i+1))
    print(van)
    return van

#this one gets the threshold mask
def get_threshold_mask(data, threshold):
    """
    get a mask with 1 after first occurance of threshold in column
    get firstNan vector with index of first above threshold
    both are ndarrays
    """
  
    # is threshold single value or array
    if threshold is None:
        return np.zeros(data.shape), np.ones(data.shape[1])*data.shape[0]
    t0 = time.time()
    pixelThreshold = np.size(threshold) == 1
    # initiate mask

    l = data.shape[0]
    # t1 = time.time()
    # print('before where', t1-t0)
    b = ma.masked_where(data >= threshold, data)
    # print(b.shape, b.dtype)
    # t2 = time.time()
    # print('before argmax', t2-t1)
    # bugfix 3 Jul 2019: if condition in ma.makske_where is True nowhere
    # the mask will be a single boolean False rather than an array of False
    if np.size(b.mask) == 1:
        first = np.zeros(data.shape[1])
    else:
        first = np.argmax(b.mask, axis=0)
    
    # t3 = time.time()
    # print('before ind', t3-t2)
    # ind = np.repeat(np.arange(data.shape[0])[:,None], data.shape[1],axis=1)
    ind = np.indices(data.shape)[1]
    # s = np.arange(l)
    # ind = np.tile(s[:,None],data.shape[1])
    # t4 = time.time()
    # print('before mask', t4-t3)
    mask = (ind >= first)
    # treat case where no value is above threshold
    # t5 = time.time()
    # print('before mask false', t5-t4)
    mask[:, first==0]=False
    # t6 = time.time()
    # print('before first 0', t6-t5)
    print(l)
    first[first==0]=l+1
    # t7 = time.time()
    # print('before return', t7-t6)
    return mask, first

#this one checks the mask vs the fit order
def check_mask_vs_fitorder(firstNan, order):
    """
    checks what columns can be fitted with what order of polynomial
    """
    # needs to be list as they could have different lengths
    fitColumnIndex = []
    # 3D mask that contains mask for each highest order possible
    mask = np.ones((order+1, firstNan.shape[0]), dtype=np.int8)
    # make firstNan masked array so we can disregard used ones
    firstNan = ma.array(firstNan, mask=np.zeros(firstNan.shape))
    # are there any series where number of data points is
    # not sufficient for order of polynom?
    for i in range(order, 0, -1):
        # can do full order
        tmp = np.int32(ma.where(firstNan > i)[0])
        fitColumnIndex.append(tmp)
        # mask the used values
        firstNan.mask[tmp] = 1
        # for the overall mask we want 0 for the values we can use
        # for this order
        mask[order-i, tmp] = 0
    # remaining columns have one or less data points
    tmp = np.int32(ma.where(firstNan <= i)[0])
    fitColumnIndex.append(tmp)
    # for the overall mask we want 0 for the values we can use
    # for this order
    mask[-1, tmp] = 0

    return fitColumnIndex, mask