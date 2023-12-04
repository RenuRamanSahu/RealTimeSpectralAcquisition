# Real time spectral acquisition
Take data from Ocean Optics Spectrometer and show the reflectivity, CIE color coordinates and the peak wavelength vs. time. 
Use the following steps to run the program

1) Download all the files to a folder.
2) Create a subfolder and make a file named "PeakWavelengthVsSampleLengthCalibration.txt". It is the file containing two columns, the sample length in mm on the first column. And in the second column the peak (or extremum) wavelength of the spectra. 
3) Run the program "NormalisationBrightDark.py" and acquire the bright and dark spectra.
4) Run the program "SingleWindow_Spectrometer_connection_colorTongue_PeakFinder_timer_v2.py"
Enter the name of the subfolder you created in step 2. ( Make sure that the folder has the file "PeakWavelengthVsSampleLengthCalibration.txt" )
A matplotlib window will open up and data will get acquired every 120 s and will be saved in your subFolder.
