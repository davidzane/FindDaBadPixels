'''
module defining the cn data cube class
'''
import glob
import os
import numpy as np
from astropy.io import fits  
import naming
import cnToolbox as ct

class CnDataCube():
    """
    class for CryoNIRSP data cubes
    """
    xSize = 2048
    ySize = 2048
    refWidth = 4
    channelWidth = 64
    nrChannels = 32

    def __init__(self, path=None, fileIdentifier=None,
                 style=None):
        self.path = path
        self.fileIdentifier = fileIdentifier
        self.style = style
        self.fileType = None
        self.nrRamps = None
        self.ndr = None
        self.data = None
        self.flattened = None
        self.cameraMode = None
        self.dataIsInverted = False

        if self.fileIdentifier is not None:
            assert(self.path is not None), "Missing path!"
            self.sort_file_list(self.get_filelist(self.path,
                                                  self.fileIdentifier,
                                                  style=self.style),
                                self.style)

   
    def get_filelist(self, path, fileIdentifier, style=None):
        """
        get a sorted filelist
        """
        self.path = path
        self.fileIdentifier = fileIdentifier
        self.style = style

        # file identifier could be list of strings
        if type(fileIdentifier) == list:
        # if isinstance(fileIdentifier) == list:
            fileList = []
            for i in fileIdentifier:
                # wild card search for Linux style folder structure
                tmp = glob.glob(os.path.join(path, i))
                fileList.extend(tmp)
            return fileList

        # wild card search for Linux style folder structure
        return glob.glob(os.path.join(path, fileIdentifier))

    def get_subarray(self, origin, width, height):
        '''
        getting a subarray from unflattenend data
        '''
        assert(not self.flattened), 'data cannot be flattenend before getting subarray'
        self.subarrayOrigin = origin
        self.subarrayWidth = width
        self.subarrayHeight = height
        if self.nrRamps == 1:
            self.subarray = self.data[:, origin[1]:origin[1]+height, origin[0]:origin[0]+width]
        else:
            self.subarray = self.data[:, :, origin[1]:origin[1]+height, origin[0]:origin[0]+width]

    def invert_16bit_data(self):
        """
        invert the slow mode data for convenience
        """
        try:
            tmp = np.int64(self.data)
            self.data = np.uint16(np.abs(tmp-(2**16-1)))
            self.dataIsInverted = True
        except AttributeError:
            raise AttributeError("no data available to invert. run read_ramp")

    def sort_file_list(self, fileList, style):
        """
        sort file list
        """
        #TODO: Dealing with line by line mode
        if style == "pizzabox":
            # sorted string list with unique ramp identifiers
            uniqueRampId = (list(set([x.split("-")[-2] for x in fileList])))
            uniqueRampId.sort()
            self.nrRamps = len(uniqueRampId)
            # go through the list and get the right files
            files = []
            self.fileType = []
            # loop through the ramps
            for ramp, num in enumerate(uniqueRampId):
                files.append([])
                for myfile in fileList:
                    if myfile.split("-")[-2] == num:
                        files[ramp].append(myfile)
                    # sorting is clever enough if only NDR number changes
                    files[ramp].sort()
                # get file extension from last file
                _, tmp = files[ramp][0].split('.')
                self.fileType.append(tmp)
        else:
            raise ValueError("Define the right style for the file list")

        # do some checking
        if self.nrRamps > 1:
            # make sure all the ramps have the same length
            lens = np.array([len(seq) for seq in files])
            if np.all(lens == lens[0]):
                self.ndr = int(lens[0])
            else:
                raise ValueError("Not all sequences the same length")
        else:
            self.ndr = len(files[0])
        self.fileList = files
        return files

    def read_ramp(self, mode, sortedFileList=None, dtype=np.uint16, verbose=False):
        """
        reading the files
        standard raw data is uint16
        """
        self.cameraMode = mode
        #TODO: dealing and descrambling of line by line mode data
        #TODO: in the future mode could be determined from fits header
        if sortedFileList is None:
            try:
                sortedFileList = self.fileList
            except AttributeError:
                raise AttributeError("file list attribute missing. run sort_file_list method")

        self.data = np.zeros((self.nrRamps, self.ndr, self.xSize, self.ySize), dtype=dtype)

        # print("in read", self.data.shape)
        # populate the array
        for i in range(self.nrRamps):
            for j in range(self.ndr):
                cur = sortedFileList[i][j]
                if verbose:
                    print('reading file:\n', cur, "\n")
                if self.fileType[i] == naming.FITS_EXTENSION:
                    self.data[i, j] = fits.open(cur)[0].data.astype(dtype)
                elif self.fileType[i] == naming.RAW_EXTENSION:
                    self.data[i, j] = ct.read_binary(cur, self.xSize, self.ySize)
                else:
                    raise ValueError("Unknown file type")
        self.flattened = False
        # squeeze to remove unnecessary dimensions for single ramp
        self.data = np.squeeze(self.data)
        # print("in read", self.data.shape)

   
    def reshape_data(self, whichWay):
        """
        reshape the data cube
        """
        # print("in reshape", self.data.shape)
        if whichWay == "down":
            if len(self.data.shape) == 4:
                # multi ramp
                self.data = self.data.reshape((self.nrRamps, self.ndr, self.xSize*self.ySize))
                self.flattened = True
                if hasattr(self, 'subarrayOrigin'):
                    self.subarray = self.subarray.reshape((self.nrRamps, self.ndr, self.subarrayWidth*self.subarrayHeight))
            elif len(self.data.shape) == 3:
                self.data = self.data.reshape((self.ndr, self.xSize*self.ySize))
                self.flattened = True
                if hasattr(self, 'subarrayOrigin'):
                    self.subarray = self.subarray.reshape((self.ndr, self.subarrayWidth*self.subarrayHeight))
            else:
                raise ValueError("Not sure how to handle this data.")
        elif whichWay == "up":
            if len(self.data.shape) == 3:
                self.data = self.data.reshape((self.nrRamps, self.ndr, self.ySize, self.xSize))
                self.flattened = False
                if hasattr(self, 'subarrayOrigin'):
                    self.subarray = self.subarray.reshape((self.nrRamps, self.ndr, self.subarrayHeight, self.subarrayWidth))
            elif len(self.data.shape) == 2:
                self.data = self.data.reshape((self.ndr, self.ySize, self.xSize))
                self.flattened = False
                if hasattr(self, 'subarrayOrigin'):
                    self.subarray = self.subarray.reshape((self.ndr, self.subarrayHeight, self.subarrayWidth))
            else:
                raise ValueError("Not sure how to handle this data.")
        else:
            raise ValueError("Need to specify whichWay to reshape [up or down]")