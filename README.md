# Brain-Tissue-Segmentation-using-Expectation-Maximization-Algorithm

The Expectation Maximization Algorithm is used to segment the brain tissues into CSF, GM and WM. The algorithm uses the 2 MRI modalities T1-Weighted and Flair and uses both modalities to perform the segmentation. Dice Coefficient is used as performance metric. The code is written in MATLAB 2018b.

The code requires the NifTI plugin of mathworks for loading nifti files. Starting from Matlab 2017b, nifti reader comes as inbuilt MATLAB function. The code has to be modified to use those new functions. For using this work as it is, Nifti loader and saver,can be downloaded from [MATLAB File Exchange - Tools for NIFTI and Analyze Image] (https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image). The functions has to be in the root folder with name of folder as 'nifti'. 

