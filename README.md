# Traceextractors
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file format. Inspired by SpikeExtractor

# Summary of initial code files: 

* class **SchnitzerTraceExtractor:** 
The main trace extractor class. 

* class **CIDataContainer:**
  * Contains all data and meta data from the .mat analysis files. 
  * Has various methods to interface with the .mat files and extract relevent meta data. 
  
* method **save_NWB:**
  * Implemented from the calcium imaging analysis to write all the data within the *CIDataContainer* object. 
  
* script: **runner.py:**
  * Test file which mines data from the .mat files, creates the *CIDataContainer* object. 
  * Saves the data into nwb format. 
  
* class **CIExtractor:**
  * The abstract class which is inherited by *SchnitzerTraceExtractor* 


