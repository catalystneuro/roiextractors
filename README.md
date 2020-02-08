# SegmentationExtractors
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file format. Inspired by SpikeExtractor

# Summary of initial code files: 

* Segmentation Extractor Objects: 

* class **SegmentationExtractor:**
  * An abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to the pre-processed data. It also contains methods to read from and write to various data formats ouput from  the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E. All the methods with @abstract decorator have to be defined by the format specific classes that inherit from this.

* class **NumpySegmentationExtractor:**
  * NumpySegmentationExtractor objects are built to contain all data coming from a file format for which there is currently no support. To construct this, all data must be entered manually as arguments.

* class **CnmfeSegmentationExtractor:**
  * This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'CNMF-E' ROI segmentation method.
  
* class **ExtractSegmentationExtractor:**
  * This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'EXTRACT' ROI segmentation method.
  
* class **SimaSegmentationExtractor:**
  *This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'SIMA' ROI segmentation method.
	
* class **NwbSegmentationExtractor:**
  *Class used to extract data from the NWB data format. Also implements a static method to write any format specific object to NWB.
	
  



