#Built using Andres Code. For getting coefficients for curve fitting.

#%% import everyNDR
import numpy as np
import numpy.ma as ma
import sys
sys.path.append("/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/lib")
import EthansToolbox as ct
from cndatacube import CnDataCube
import naming
import os
import matplotlib.pyplot as plt

#%% Let's make the data cube!
a = CnDataCube("/Users/mrpoopoopants/desktop/TestFitsImages",
               "slow-tripple-SiIX-t0-00020-*.fits",
               "pizzabox")
a.read_ramp(naming.SLOW_CAMERA_MODE, verbose=True)
a.invert_16bit_data()
print("this is the current shape of the data:", a.data.shape)

#what is dead beef?

#%% Now let's try the subarray
a.get_subarray([4,4], a.data.shape[1]-8, a.data.shape[1]-8)
print("this is the current shape of the data:", a.subarray.shape)

#%% Now try reshape subarray data from 3D to 2D so we can do everyNDR with it
a.reshape_data("down")
print("this is the current shape of the data:", a.subarray.shape)

#%% now run a for loop that loops 10 times, each with different slicings
for element in range(1,11):
    #%% Now try slicing the data
    sliced_subarray = a.subarray[::element]
    print("this is the current shape of the data:", sliced_subarray.shape)

    #%% Okay, time to make a threshold to take out bad pixels (I think)
    threshold = 2.**16-1
    order = 2
    nrPixels = sliced_subarray[1]

    thresholdMask = np.zeros(sliced_subarray.shape)
    print("this is the shape of the threshold mask:", thresholdMask.shape)

    fitColumnIndex = []
    if a.nrRamps == 1:
        fitColumnMask = np.zeros((order+1, sliced_subarray.shape[1]))
        firstNan = np.zeros((sliced_subarray.shape[1]))

    howMany = np.zeros((a.nrRamps, order+1)) #what is this do?

    for i in range(a.nrRamps):
        if a.nrRamps ==1:
            thresholdMask, firstNan = ct.get_threshold_mask(sliced_subarray, threshold)
            tmp, fitColumnMask = ct.check_mask_vs_fitorder(firstNan, order)
            print(tmp[0].shape, tmp[0].dtype)
        for j in range(order+1):
            howMany[i,j] = len(tmp[j])
        fitColumnIndex.append(tmp)
    print(howMany, "\n") 

    #%% Now finally, some curve fitting everyNDR (I think)
    dataTime = np.arange(sliced_subarray.shape[0])*.502 + 0.502

    if a.nrRamps == 1:
        coef = ct.lspolyfit(dataTime, sliced_subarray, order, 
                                naming.SLOW_CAMERA_MODE,
                                thresholdMask=thresholdMask,
                                firstNan=firstNan,
                                fitColumnIndex=fitColumnIndex[0],
                                fitColumnMask=fitColumnMask,
                                full=False, alpha=0.95)
        print("the shape of the coef is:", coef.shape)

    #%% And now save the sliced_subarray as an outfile

    dir = "/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/"
    coef.dump(os.path.join(dir,"elisaslice"+str(element)))
    print("file made")

#%% Now Histogram making to see bad pixels
import numpy as np
import matplotlib.pyplot as plt

data_everyNDR = np.load('elisaslice1')
perc_err_global = []

for item in range (1, 11):
    data_NDR = np.load('elisaslice'+str(item))

    linear_coef_allNDR = data_everyNDR[1, :]
    linear_coef_NDR = data_NDR[1,:]

    percent_error = (linear_coef_NDR - (2*linear_coef_allNDR)) / (2*linear_coef_allNDR)

    perc_err_global = [perc_err_global, percent_error]

    plt.hist(percent_error, 50, density=True, range=(0, 0.0025))
    plt.show()
    plt.savefig(os.path.join(dir, 'histogram'+str(item)+'.png'))
















#%%
