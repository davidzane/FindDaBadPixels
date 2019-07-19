#Built using Andres Code. For getting coefficients for curve fitting. And comparing them for data analysis
#Applies threshold before slicing and shows percentage error between no slice and 1 through 10 slice over array. COLORIZED
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
a = CnDataCube("/Users/mrpoopoopants/desktop/jul-16-2019",
               "first-00008-*.fits",
               "pizzabox")
a.read_ramp(naming.SLOW_CAMERA_MODE, verbose=True)
a.invert_16bit_data()
print("this is the current shape of the data:", a.data.shape)

#what is dead beef?

#%% Now let's try the subarray
a.get_subarray([500, 500],100,100)
print("this is the current shape of the data:", a.subarray.shape)

#%% Now try reshape subarray data from 3D to 2D so we can do stuff with it
a.reshape_data("down")
print("this is the current shape of the data:", a.subarray.shape)

#%% Okay, this time apply a threshold before slicing
threshold = 45000
order = 2
nrPixels = a.subarray.shape[1]

thresholdMask = np.zeros(a.subarray.shape)
print("this is the shape of the threshold mask:", thresholdMask.shape)

fitColumnIndex = []
if a.nrRamps == 1:
    fitColumnMask = np.zeros((order+1, a.subarray.shape[1]))
    firstNan = np.zeros((a.subarray.shape[1]))

howMany = np.zeros((a.nrRamps, order+1)) #what is this do?

for i in range(a.nrRamps):
    if a.nrRamps ==1:
        thresholdMask, firstNan = ct.old_get_threshold_mask(a.subarray, threshold)
        print("this is the shape of the firstNan:", firstNan.shape)
        tmp, fitColumnMask = ct.check_mask_vs_fitorder(firstNan, order)
        #print(tmp[0].shape, tmp[0].dtype)
    for j in range(order+1):
        howMany[i,j] = len(tmp[j])
    fitColumnIndex.append(tmp)
print("how many?", howMany, "\n") 

#%% now slice it up!!!!!!! Sliced coef array files will be saved. Also need to slice the masks as well

for element in range(1,11):
    #%% Now slice the data and ThresholdMask and firstNan
    sliced_subarray = a.subarray[::element]
    sliced_thresholdMask = thresholdMask[::element]
    sliced_firstNan = np.ceil(firstNan / element)

    #print("this is the current shape of the data (after slicing):", sliced_subarray.shape)


    #%% Now finally, some curve fitting (I think)
    dataTime = np.arange(sliced_subarray.shape[0])*.502 + 0.502

    if a.nrRamps == 1:
        coef = ct.masked_polyfit(dataTime, sliced_subarray, order, sliced_thresholdMask, sliced_firstNan)
        print("the bias", ma.median(coef[-1]))
        print("the shape of the coef is:", coef.shape)

    #%% And now save the coef array of sliced array as an outfile

    dir = "/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/"
    coef.dump(os.path.join(dir,"elisaslice"+str(element)))
    print("file made")

#%% Now construct the fit curve (currently only for quadratic)
#loops for different slicings
for item in range(1,11):
    number_of_NDRs = math.ceil(a.data.shape[0]/item)
    fit = ma.zeros((number_of_NDRs,100,100),mask=thresholdMask)  #occhio alle dimensioni
    
    coef_array = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/elisaslice'+str(item))

    bias = coef_array[-1].reshape(100,100)
    print('bias',bias[50,50])
    linear_coef = coef_array[-2].reshape(100,100)
    quadratic_coef = coef_array[-3].reshape(100,100)

    dataTime = np.arange(number_of_NDRs)*.502 + 0.502
    
    #loops for every NDR
    for readNumber in range(1,number_of_NDRs):
        x = np.ones((100,100))*dataTime[readNumber]
        
        fit[readNumber,:,:] = (bias) + (linear_coef * x) + (quadratic_coef * (x**2))
    
    #makes a 3D array file of fit of every pixel in every NDR
    fit.dump(os.path.join(dir,"FitCurve"+str(item)))

#%% Now plot fit curves
for plot_number in range(1,5):
    sliced_firstNan = np.ceil(firstNan / plot_number)
    fit_array = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/FitCurve'+str(plot_number))
    actual_pixel_values = a.subarray.reshape(120,100,100)
    sliced_pixel_values = actual_pixel_values[::plot_number]

    #select pixel to view fit of
    single_pixel_fit = fit_array[:,20,20]
    plt.plot(single_pixel_fit)
    single_pixel_actual = sliced_pixel_values[:,20,20]
    plt.plot(single_pixel_actual, "x")


#%%implement residuals
for plot_number in range(1,5):
    sliced_firstNan = np.ceil(firstNan / plot_number)
    fit_array = np.load('/Users/mrpoopoopants/desktop/cryonirsp/cnpipeline-summer/cnpipeline/FitCurve'+str(plot_number))
    actual_pixel_values = a.subarray.reshape(120,100,100)
    sliced_pixel_values = actual_pixel_values[::plot_number]
    residual = abs(fit_array - sliced_pixel_values)

    shaped_up_firstNan = sliced_firstNan.reshape(100,100)

    mean_res = ma.sum(residual/sliced_pixel_values,axis=0)/shaped_up_firstNan
    mean_res = mean_res.reshape(100*100)

    hist = plt.hist(mean_res,100,range=(-5e-2,1.1))
    plt.show()


#%%
