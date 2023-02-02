# LDA-Data-Processing

A repository for a number of scripts used in filtering LDA data.

The general workflow is done entirely in Python and can be summarized as followed:

1. Convert raw .csv files in npy arrays using data_extraction_csv_auto.py
2. Use a separate script to call various post-processing functions from LDA_Toolbox.py
   Make sure this file is in the root folder and import them using: import LDA_Toolbox as LDA
3. Imaging of the resulting, e.g. via matplotlib
