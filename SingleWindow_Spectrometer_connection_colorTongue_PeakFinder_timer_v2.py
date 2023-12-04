import seabreeze
from seabreeze.spectrometers import Spectrometer, list_devices
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import os
import R2cie #### self made library

global startTime
startTime = time.time()


device = list_devices()
print(device)

#######################################
######Obtaining the spectrum###########
#######################################

spec = Spectrometer.from_serial_number('FLMT05853')

#set integration time
spec.integration_time_micros(10000) #time in microsecond



#### Folder For saving data ####
print("Enter the filename for your sample below and hit enter: ")
global fname
fname = input()
print('All the data will be saved in the folder '+ fname)
script_path = os.path.realpath(fname)
if not os.path.exists(script_path):
  os.mkdir(script_path)
################################



####### Functions for finding peaks #############


######################### To Find Local extrema #####################
#Obtaining derivative

def find_peaks_live(x, y):
    #Getting Derivatives
    kernel = [1, 0, -1]

    dy = np.convolve(y, kernel, 'valid') 
    x_ = np.convolve(x, [0, 1, 0], 'valid')

   
            
    
    
    #Checking for sign-flipping
    S = np.sign(dy)
    #plt.plot(x_ , S)
    x__ = np.convolve(x_, [0, 1, 0], 'valid')
    dS = np.convolve(S, kernel, 'valid')
    #plt.plot(x__, -dS)
    #plt.plot(x, y)
    #abs_dS = np.abs(dS)
    for j in range(4): #increasing this number will smooth out the data and increase stability at the cost of runtime
        x__, dS = smoothen_BoxedAvg(x__, dS, int(len(x__)/40))
    sign_dS = np.sign(dS)
    x__, d_sign_dS = firstDerivative(x__, sign_dS) #This will generate the spiked data
   
    #The following code will generate the single wavelength value corresponding to a peak
    
    
    #Each element of the array is_peak_or_dip is an array of three elements, namely index and correspoding wavelength and sign
    #peak_or_dip = []  # This array will have the values of wavelengths at which peak or dip is there
    len_d_sign_dS = len(d_sign_dS)
    spike_index = []
    spike_wvl = []
    spike_sign = []
    for j in range(len_d_sign_dS):
        if(d_sign_dS[j] != 0 ):
            spike_index = np.append(spike_index, [j])
            spike_wvl = np.append(spike_wvl, [x__[j]])
            spike_sign = np.append(spike_sign, [ d_sign_dS[j]])
            
     
    is_loc_extrema = np.convolve(spike_sign, [0.5, -0.5], 'valid') #Non-zero value when 
    loc_extrema_wvl = np.convolve(spike_wvl, [0.5, 0.5], 'valid')
    #print(spike_sign)
    #print(is_loc_extrema) ####Determines the the local extrema, elements can be 1 or 0 or -1
    #print(loc_extrema_wvl)
    len_is_loc_extrema = len(is_loc_extrema)
    
    loc_extrema_wavelengthVal = []
    for j in range(len_is_loc_extrema):
        if(is_loc_extrema[j] != 0):
            loc_extrema_wavelengthVal = np.append(loc_extrema_wavelengthVal, [loc_extrema_wvl[j]])
        
           
    ################
    wavelengthVal = np.append(x__, loc_extrema_wvl)
    IndicatorLocExtrema = np.append(np.zeros(len(x__)), is_loc_extrema)
    ################
    sortedIndex_wavelengthVal = np.argsort(wavelengthVal)
    wavelengthVal = np.take_along_axis(wavelengthVal, sortedIndex_wavelengthVal, axis = 0)
    IndicatorLocExtrema = np.take_along_axis(IndicatorLocExtrema, sortedIndex_wavelengthVal, axis = 0)
    
    
    return wavelengthVal, IndicatorLocExtrema, loc_extrema_wavelengthVal



################################## Functions needed for above function #######################################################

def firstDerivative(x__, y__):
    dy__ = np.convolve(y__, [1, 0, -1], 'valid')
    x__ = np.convolve(x__, [0, 1, 0], 'valid')
    return x__, dy__
    

def smoothen_BoxedAvg(x, y, N):
    smoothen = np.ones(N)/N
    y = np.convolve(y, smoothen, 'valid')
    x = np.convolve(x, smoothen, 'valid')
    return x, y
    
#################################################

def nearest(array, val):
  len_array = len(array)
  valArr = val*np.ones(len_array)
  indx = np.argmin(np.abs(array-valArr))
  return indx
#print(nearest([1, 5.4, 9, 100], 89.9))

  
#######################################################################################
#################### End of functions for finding peaks ###############################


################## Function to save data #####################
def autosave():
  currentTimeInt = int(  time.time() - startTime )
  saveDeciding_Int = (currentTimeInt%120) #saves data in every 120 iterations
  if (saveDeciding_Int == 0):
      fig.savefig(script_path + '/timestamp_'+str(currentTimeInt)+'s_Figure.png', format = 'png')
      fig.savefig(script_path + '/timestamp_'+str(currentTimeInt)+'s_Figure.svg', format = 'svg')
      #Save Reflectivity in text format
      np.savetxt(script_path+'/ReflectivityData'+str(currentTimeInt)+'s.txt', np.transpose(ln.get_data()))
      #Save Peak wavelength vs time in text format
      np.savetxt(script_path+'/PeakWavelengthVsTime.txt', np.transpose(peakVsTime.get_data()))
      print("Reflectivity Data and PeakVsTime Data is saved. ")
  

    
##############################################################


######################## window for spectrum axquisition #######################
fig = plt.figure( figsize=(9,6))
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
gs = fig.add_gridspec(nrows = 2, ncols = 2, width_ratios = [2, 1], height_ratios = [1.5, 1])

ax = fig.add_subplot(gs[0, 0])
axCyclic = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
axSample = fig.add_subplot(gs[1, 1])
axSample.set_ylim(380, 780)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

xdat, ydat = spec.wavelengths(), spec.intensities()
ln = ax.plot(xdat, ydat, 'r-' )[0]
peaks =  ax.plot(xdat, np.zeros(len(xdat)), 'b-')[0]
wavelength = xdat
Reflectivity = ydat

timeDat = np.arange(0, 0.02, 0.01)
wvlPeakDat = np.zeros(len(timeDat))
peakVsTime = axCyclic.plot(timeDat, wvlPeakDat, 'g.' )[0]
axCyclic.set_xlabel('time(s)')

I = plt.imread("CIE_xyY40percent.tiff")
I = I[100:1000, 0:800 ]
ax3.imshow(I, extent = [0, 0.8, 0, 0.9])
ax3.set_xlabel('CIE x')
ax3.set_ylabel('CIE y')
cieLn = ax3.plot([0.3127, 0.3127, 0.3127], [0.3290, 0.3290, 0.3290], '.', color = 'black', alpha = 1)[0]
##mngr2 = plt.get_current_fig_manager()
##mngr2.window.setGeometry(670,40,640, 514)

sampleLen_vs_wavelen = np.loadtxt(script_path+'/PeakWavelengthVsSampleLengthCalibration.txt')

axSample.plot(sampleLen_vs_wavelen[:, 0], sampleLen_vs_wavelen[:, 1], 'b.')[0]

m, c = np.polyfit(sampleLen_vs_wavelen[:, 0], sampleLen_vs_wavelen[:, 1], 1 )
p_inv, m_inv, c_inv = np.polyfit(sampleLen_vs_wavelen[:, 1], sampleLen_vs_wavelen[:, 0], 2 )

##fig_calibrate, axCalibrate = plt.subplots()
##axCalibrate.plot(sampleLen_vs_wavelen[:, 1], sampleLen_vs_wavelen[:, 0], '.')
##axCalibrate.plot(sampleLen_vs_wavelen[:, 1], p_inv*sampleLen_vs_wavelen[:, 1]**2+m_inv*sampleLen_vs_wavelen[:, 1]+c_inv)
##axCalibrate.set_xlabel('wavelength (nm)')
##axCalibrate.set_xlabel('Sample Length (mm)')



global min_sampleLen
max_sampleLen = max(sampleLen_vs_wavelen[:, 0])
min_sampleLen = min(sampleLen_vs_wavelen[:, 0])
axSample.plot(sampleLen_vs_wavelen[:, 0], m*sampleLen_vs_wavelen[:, 0]+c)
axSample.set_xlabel('Sample Length(mm)')
axSample.set_ylim(0, 2.6)
axSample.set_yticks([])
axSample.set_xlim(0, max_sampleLen+10)
axSampleVline = axSample.plot([min_sampleLen, min_sampleLen], [0, 3], 'r--')[0]

SimulatedSampleBar = axSample.barh(1, 20, height = 0.6, color = 'green', align = 'center', alpha = 0.75)[0]

#### Button press event on fig window ####
ref_x = [700, 700]
ref_y  = [0, 1]
ref_ln, = ax.plot(ref_x, ref_y, '-', color = 'lime')

def onclick(event):
##    print(event.x, event.y)
##    print(event.xdata, event.ydata)
##    print(event.inaxes)
    if(event.inaxes == ax):
      ix = event.xdata
      ref_x = [ix, ix]
      ref_y = [0, 1]
      ref_ln.set_data(ref_x, ref_y)
      np.savetxt(script_path+'/ReflectivityData.txt', np.transpose(ln.get_data()))
      #print(True)


      
    
##    return ref_ln,

    if (event.inaxes == axSample):
      ix = event.xdata
      axSampleVline.set_data([ix, ix], [0, 3])
##      print(ix)
##    else:
##      axSampleVline.set_data([0, 0], [0, 3])


    return ref_ln, axSampleVline


   

    
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)

#################################################################################








########## spectra window #####
def init():
    ax.set_xlim(380, 780)
    ax.set_xlabel('wavelength(nm)')
    ax.set_ylabel('Reflectivity')
    
    timeDat = peakVsTime.get_xdata()
    max_timeDat = max(timeDat)
    min_timeDat = min(timeDat)
    axCyclic.set_xlim(min_timeDat, max_timeDat)
    axCyclic.set_ylim( 400, 750)
    axCyclic.set_ylabel(r'Peak $\lambda$ (nm)')




    
    return ln, peaks, peakVsTime, cieLn, SimulatedSampleBar



def acquire(boxavg):
    ydat = spec.intensities()
    xdat = spec.wavelengths()
   

    trimLen = boxavg
    totLen = len(ydat)

    ###Smoothing####
    ydat = np.convolve(ydat, np.ones(trimLen)/trimLen, 'same')

    
    ### trimming  ###
    xdat = xdat[trimLen:totLen-trimLen]
    ydat = ydat[trimLen:totLen-trimLen]

    #print(len(ydat))

    brightDat = np.loadtxt("bright.txt")[:, 1]
    #print(len(ydat), len(brightDat))
    darkDat = np.loadtxt("dark.txt")[:, 1]
    ydat = (ydat-darkDat)/(brightDat-darkDat)

    xdat = xdat[500:2900]
    ydat = ydat[500:2900]
    return xdat, ydat


def acquireNaccumulate(accumulation, boxavg):
     xdat, ydat = acquire(boxavg)
     for i in range(accumulation-1):
        x, y = acquire(boxavg)
        xdat = xdat + x
        ydat = ydat + y

     xdat = xdat/accumulation
     ydat = ydat/accumulation
     return xdat, ydat



def update(frame):
    t = time.time()-startTime
    #print(frame)
    accumulation = 10
    boxavg = 450
    xdat, ydat = acquireNaccumulate(accumulation, boxavg)
    ln.set_data(xdat, ydat)
    ax.set_ylim(min(ydat), max(ydat))


    ##### peaks ####
    x = np.arange(380, 780, 1)    # This code chooses the wavelength ranges
    y = np.interp(x, ln.get_xdata(), ln.get_ydata())
    x_, sign_d2y, loc_extrema_wavelengthVal = find_peaks_live(x, y)
    len_loc_extrema_wavelengthVal = len(loc_extrema_wavelengthVal)



    ref_xdata = ref_ln.get_xdata()
    #print(ref_xdata)
    wavelengthValClicked = ref_xdata[0]
    
    idx = 1
    
    idx = nearest(loc_extrema_wavelengthVal, wavelengthValClicked)
        

    currentWvlDetected = x_[np.nonzero(sign_d2y)][idx]
    ref_xdata = [currentWvlDetected, currentWvlDetected]
    ref_ln.set_xdata(ref_xdata)
    #print(currentWvlDetected)
    
    peaks.set_data(x_, np.abs(sign_d2y))

    
    
    if (len(np.nonzero(sign_d2y)) > 0):
        wavelengthPeak = x_[np.nonzero(sign_d2y)][idx]
        timeDat, wvlPeakDat  = peakVsTime.get_data()
        timeDat = np.append(timeDat, [t])
        wvlPeakDat = np.append(wvlPeakDat, [wavelengthPeak])
        peakVsTime.set_data(timeDat, wvlPeakDat)

    #The following line prints the time and the peak wavelength on the canvas
    ax.set_title("t = "+str(format(t, '.2f'))+"s, Peak = "+str(format(wavelengthPeak,  '.2f'))+" nm")


    #
    timeDat, wvlpeakDat = peakVsTime.get_data()
    #print(timeDat)
    len_timeDat = len(timeDat)
    timeDat = timeDat[1:len_timeDat]
    wvlpeakDat = wvlpeakDat[1:len_timeDat]
    max_timeDat = max(timeDat)
    axCyclic.set_xlim(max_timeDat-100, max_timeDat)


    wavelength, Reflectivity = (ln.get_data())
    minR = min(Reflectivity)
    maxR = max(Reflectivity)
    lenR = len(Reflectivity)
    Reflectivity = (Reflectivity - minR*np.ones(lenR))/(np.ones(lenR)*(maxR - minR))
    x, y, Y = R2cie.Reflectivity2cie(wavelength, Reflectivity)
    ciex, ciey = cieLn.get_data()
    ciex = np.append(ciex[2:2], [x])
    ciey = np.append(ciey[2:2], [y])
    cieLn.set_data(ciex, ciey)
    ax3.set_title("CIEx = "+str(format(x, '.2f'))+", CIEy = "+str(format(y,  '.2f')))

    Y = 0.3
    rgb = R2cie.xyY_to_rgb(x, y, Y)
    SimulatedSampleBar.set_color(rgb)
    len_wvlpeakDat = len(wvlpeakDat)
    wvlPeakVal_ = wvlpeakDat[len_wvlpeakDat-1]
    bar_width = p_inv*wvlPeakVal_*wvlPeakVal_ + m_inv*wvlPeakVal_ + c_inv
##    print(wvlPeakVal_, bar_width)
    SimulatedSampleBar.set_y(1)
    SimulatedSampleBar.set_width(bar_width)
    strainVal = (bar_width-min_sampleLen)*100/min_sampleLen
    axSample.set_title('Strain % = '+str(format(strainVal, '.2f'))+"")





##    print("DPI of the figure is ", fig.dpi)
##    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
##    width, height = bbox.width, bbox.height
##    print("Axis sizes are(in pixels):", width, height)



    autosave()


    
    return ln, peaks, peakVsTime, cieLn, SimulatedSampleBar










    
 
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
frame = np.linspace(0, 2*np.pi, 1000)
ani = FuncAnimation(fig, update, frames=frame, init_func=init, interval = 200, blit=False)


##FFwriter = animation.FFMpegWriter(fps=60)
##ani.save(script_path+'/CyclicTest_video.mp4', writer = FFwriter)

def save(event):
  print('Input the filename: ')
  filename = input()
  fig.savefig(script_path + '/'+filename +'.png', format = 'png')
  fig.savefig(script_path + '/'+filename +'.svg', format = 'svg')
  #Save Reflectivity in text format
  np.savetxt(script_path+'/'+filename +'.txt', np.transpose(ln.get_data()))
  #Save Peak wavelength vs time in text format
  print(" Data  saved. ")






ax_save = plt.axes([0.001, 0.92, 0.075, 0.045])
ButtonSave = Button(ax_save, 'Save')
ButtonSave.on_clicked(save)


plt.show()
plt.close()

f = script_path + "/animation.gif" 
writergif = PillowWriter(fps=30) 
ani.save(f, writer=writergif)


spec.close()


