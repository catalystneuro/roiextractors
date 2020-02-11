# SegmentationExtractors
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats. Inspired by [SpikeExtractors](https://github.com/SpikeInterface/spikeextractors). 
![image](https://drive.google.com/uc?export=view&id=1bhRA3kyu3SA3k-xWz5psRxLsuP3BJEBg)


### Segmentation Extractor Objects: 

* class **SegmentationExtractor:**
  * An abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to the pre-processed data. It also contains methods to read from and write to various data formats ouput from  the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E. 

* class **NumpySegmentationExtractor:**
  * NumpySegmentationExtractor objects are built to contain all data coming from a file format for which there is currently no support. To construct this, all data must be entered manually as arguments.

* class **CnmfeSegmentationExtractor:**
  * This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'CNMF-E' ROI segmentation method.
  
* class **ExtractSegmentationExtractor:**
  * This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'EXTRACT' ROI segmentation method.
  
* class **SimaSegmentationExtractor:**
  * This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'SIMA' ROI segmentation method.
	
* class **NwbSegmentationExtractor:**
  * Class used to extract data from the NWB data format. Also implements a static method to write any format specific object to NWB.
  
### **Example Datasets:** 
  * These are the datasets which are used to run tests can be downloaded [here](https://drive.google.com/drive/folders/1CeDfr6yza_bh0vYD2E1HF_3_S8pg2yLW?usp=sharing). Nest the folder as '/tests/testdatasets' before running the tests. 
	
  



