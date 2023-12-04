import seabreeze
from seabreeze.spectrometers import Spectrometer, list_devices
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import os
import R2cie #### self made library


device = list_devices()
print(device)

#######################################
######Obtaining the spectrum###########
#######################################

spec = Spectrometer.from_serial_number('FLMT05853')

#set integration time
spec.integration_time_micros(10000) #time in microsecond











wavelength, Reflectivity = spec.wavelengths(), spec.intensities()
print(len(wavelength))
######### Plot window for cie ################################

######################## window for spectrum axquisition #######################
fig, ax = plt.subplots()
xdat, ydat = spec.wavelengths(), spec.intensities()
ln, = plt.plot(xdat, ydat, 'r-' )
wavelength = xdat
Reflectivity = ydat
mngr2 = plt.get_current_fig_manager()
mngr2.window.setGeometry(670,40,640, 514)

#################################################################################






########## spectra window #####
def init():
    ax.set_xlim(150, 1000)
    ax.set_xlabel('wavelength(nm)')
    return ln,


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


##    xdat = xdat[580:2900]
##    ydat = ydat[580:2900]
    print(len(ydat))
    return xdat, ydat


def acquireNaccumulate(accumulation, boxavg):
     xdat, ydat = acquire(boxavg)
     print(len(ydat))
     for i in range(accumulation-1):
        x, y = acquire(boxavg)
        xdat = xdat + x
        ydat = ydat + y

     xdat = xdat/accumulation
     ydat = ydat/accumulation
     print(len(ydat))

     return xdat, ydat
       


def update(frame):
    accumulation = 10
    boxavg = 450
    xdat, ydat = acquireNaccumulate(accumulation, boxavg)
   
    ln.set_data(xdat, ydat)
    ax.set_ylim(min(ydat), max(ydat))
    return ln,



#########################################################




####################################################




###### definition of buttons ########
def Light(event):
    x_, y_ = (ln.get_data())
    plt.figure()
    plt.plot(x_, y_)
    np.savetxt('bright.txt', np.transpose([x_, y_]))
    plt.show()

def Dark(event):
    x_, y_ = (ln.get_data())
    plt.figure()
    plt.plot(x_, y_)
    np.savetxt('dark.txt', np.transpose([x_, y_]))
    plt.show()



    
 
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
frame = np.arange(0, 2*np.pi, 0.001)
ani = FuncAnimation(fig, update, frames=frame, init_func=init, interval = 200, blit=False)


ax_bright = plt.axes([0.1, 0.9, 0.09, 0.075])
ButtonBright = Button(ax_bright, 'Bright')
ButtonBright.on_clicked(Light)

ax_dark = plt.axes([0.2, 0.9, 0.09, 0.075])
ButtonDark = Button(ax_dark, 'Dark')
ButtonDark.on_clicked(Dark)





plt.show()

spec.close()



