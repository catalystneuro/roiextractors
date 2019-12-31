# Traceextractors
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file format. Inspired by SpikeExtractor

# Summary of initial code files: 

* class **CIExtractor:**
  * The abstract class which is inherited by *TraceExtractor* 
  
* class **TraceExtractor:**
  * Contains all data and meta data from the .mat analysis files. 
  * Has various methods to interface with the .mat files and extract relevent meta data. 
  
* method **nwbwriter:**
  * Implemented from the calcium imaging analysis to write all the data within the *CIDataContainer* object. 
  



