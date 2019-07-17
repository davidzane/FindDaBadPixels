#Built using Andres Code. For getting coefficients for curve fitting. And comparing them for data analysis

#%% import stuff
import numpy as np
import numpy.ma as ma
import sys
sys.path.append("/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/lib")
import EthansToolbox as ct
from EthansCnDataCube import CnDataCube
import naming
import os
import matplotlib.pyplot as plt
import math

#%% Let's make the data cube!
a = CnDataCube("/Users/mrpoopoopants/desktop/TestFitsImages",
               "slow-tripple-SiIX-t0-00020-*.fits",
               "pizzabox")
a.read_ramp(naming.SLOW_CAMERA_MODE, verbose=True)
a.invert_16bit_data()
print("this is the current shape of the data:", a.data.shape)

#what is dead beef?

#%% Now let's try the subarray
a.get_subarray([4, 4],2040 ,2040)
print("this is the current shape of the data:", a.subarray.shape)

#%% Now try reshape subarray data from 3D to 2D so we can do stuff with it
a.reshape_data("down")
print("this is the current shape of the data:", a.subarray.shape)

#%% now run a for loop that loops 10 times, each with different slicings. Coef array files will be saved.
for element in range(1,11):
    #%% Now slice the data
    sliced_subarray = a.subarray[::element]
    #print("this is the current shape of the data (after slicing):", sliced_subarray.shape)

    #%% Okay, time to make a threshold to take out bad pixels (I think)
    threshold = 27000
    order = 2 
    nrPixels = sliced_subarray.shape[1]

    thresholdMask = np.zeros(sliced_subarray.shape)
    #print("this is the shape of the threshold mask:", thresholdMask.shape)

    fitColumnIndex = []
    if a.nrRamps == 1:
        fitColumnMask = np.zeros((order+1, sliced_subarray.shape[1]))
        firstNan = np.zeros((sliced_subarray.shape[1]))

    howMany = np.zeros((a.nrRamps, order+1)) #what is this do?

    for i in range(a.nrRamps):
        if a.nrRamps ==1:
            thresholdMask, firstNan = ct.old_get_threshold_mask(sliced_subarray, threshold)
            tmp, fitColumnMask = ct.check_mask_vs_fitorder(firstNan, order)
            #print(tmp[0].shape, tmp[0].dtype)
        for j in range(order+1):
            howMany[i,j] = len(tmp[j])
        fitColumnIndex.append(tmp)
    print("how many?", howMany, "\n") 

    #%% Now finally, some curve fitting (I think)
    dataTime = np.arange(sliced_subarray.shape[0])*.502 + 0.502

    if a.nrRamps == 1:
        coef = ct.masked_polyfit(dataTime, sliced_subarray, order, thresholdMask, firstNan)
        print("the bias", ma.median(coef[-1]))
        #print("the shape of the coef is:", coef.shape)

    #%% And now save the sliced_subarray as an outfile

    dir = "/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/"
    coef.dump(os.path.join(dir,"elisaslice"+str(element)))
    print("file made")

        # # Now get the median percent error
        # data_everyNDR = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/elisaslice1')
        # data_slicedNDR = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/elisaslice'+str(element))
        # linear_coef_allNDR = data_everyNDR[1]
        # linear_coef_sliceoNDR = data_slicedNDR[1]
        # percent_error_multiplier = a.data.shape[0] / sliced_subarray.shape[0]
        # #print("percent error multiplier", percent_error_multiplier)
        # percent_error = 100 * ( percent_error_multiplier * linear_coef_allNDR - linear_coef_sliceoNDR ) / (percent_error_multiplier * linear_coef_allNDR)
        # print("median percent error for", element,"is", ma.median(percent_error))


#%% Now make histograms to see bad pixels!!!!!!!!!!!!!!!!

#Load the coef array that we are comparing against 
coef_allNDR = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/elisaslice1')

#loop through to load each slicing and make a histogram for each
for item in range (2, 11):
    coef_sliceoNDR = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/elisaslice'+str(item))

    #select the coeffeciant for the linear term
    linear_coef_allNDR = coef_allNDR[-2]
    linear_coef_sliceoNDR = coef_sliceoNDR[-2]

    #calculate the percent error
    percent_error = ( 100 * ( linear_coef_sliceoNDR - 
                    (item * linear_coef_allNDR) ) / 
                    (item * linear_coef_allNDR) )
  
    #calculate the median value of the percent error array  
    median_percent_error = ma.median(percent_error)
    
    print("the median percent off of",item,"slice coef from no slice coef is:", median_percent_error)

    plt.hist(percent_error, 1000, density=True, range=(-60, 60))
    plt.title = str(item)
    plt.savefig(os.path.join(dir, 'histogram'+str(item)+'.png'))
    plt.show()

#%%
