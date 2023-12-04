
#wavelength = np.linspace(380., 780., 400) #nm
#spectrum = np.interp(wavelength, wavelengthData, spectralData)


def xyY_to_rgb(x, y, Y):    # we shall set Y to 100
    z = 1 - x - y
    X = x*Y/y
    Z = z*Y/y
    #print("X, Y, Z  = ", X, Y, Z)

    # max_XYZ = max([X, Y, Z])
    # X_norm = Z/max_XYZ
    # Y_norm = Y/max_XYZ
    # Z_norm = Z/max_XYZ

    XYZ_norm = [X, Y, Z]

    XYZ_to_RGB = [[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]]



    RGB = np.matmul(XYZ_to_RGB, np.transpose(XYZ_norm))
    r, g, b = RGB
    rgb = [r, g, b]
    for i in range(3):
        if(rgb[i]<=0.0031308):
            rgb[i] = 12.92*rgb[i]
        else:
            rgb[i] = 1.055*((rgb[i])**(1.0/2.4)) - 0.055

    for i in range(3):
        if(rgb[i]<0):
            rgb[i] = 0.0
        if(rgb[i]>1):
            rgb[i] = 1.0



    #print("rgb = ",rgb)
    return rgb





############################################################################
############################################################################
############################################################################
import numpy as np




def Reflectivity2cie(wavelength, R):
##    plt.ylabel("Reflectivity(%)")
##    plt.xlabel("wavelength (nm)")
##    plt.plot(wavelength, spectrum, '-')
##    plt.xlim(380, 750)

    #Standard Illuminant
    D65_data = np.genfromtxt("D65.txt")
    D65 = np.interp(wavelength, D65_data[:, 0], D65_data[:, 1])


    #Color Matching Functions
    CMFs = np.loadtxt("StandardCalorimetricObserver1931.txt")
    xbar = np.interp(wavelength, CMFs[:, 0], CMFs[:, 1])
    ybar = np.interp(wavelength, CMFs[:, 0], CMFs[:, 2])
    zbar = np.interp(wavelength, CMFs[:, 0], CMFs[:, 3])
    A = np.transpose([xbar, ybar, zbar])



    #Applying color functions to D65 spectrum
    ATD65 = np.matmul(np.transpose(A), D65)
    #print(ATD65[0], ATD65[1], ATD65[2] )
    XYZ_D65 = [ ATD65[0]/ATD65[1], ATD65[1]/ATD65[1], ATD65[2]/ATD65[1] ]
    XYZ_D65_sum = (sum(XYZ_D65))
    [x_D65, y_D65, z_D65] = XYZ_D65/XYZ_D65_sum   #white point
#    print("White point = ", x_D65, y_D65)


    #Reflectivity spectrum

    #Reflectivity spectrum
    S = R*D65



    #Handling the reflected spectrum
    ATS = np.matmul(np.transpose(A), S) #A_transpose, Spectrum*D65 pointwise multiplication
    XYZ = ATS/ATD65[1]
    XYZ_sum = sum(XYZ)
    #print(XYZ[0], XYZ[1], XYZ[2])
    #print(XYZ_sum)
    xyz = XYZ/XYZ_sum
#    print("x = "+str(xyz[0])+ ", y = "+ str(xyz[1]) + ", Y = "+ str(XYZ[1]))

    x = xyz[0]
    y = xyz[1]
    Y = XYZ[1]


#    print("Chromaticity x = "+ str(x)+ " y = "+ str(y)+ " Luminiscence Y = "+str(Y))
    return x, y, Y

##    img_r, img_g, img_b = xyY_to_rgb(x, y, Y/100)
##
##
##    img = np.zeros((10, 10, 3))
##
##    for j in range(10):
##        for k in range(10):
##                img[j, k] = [ img_r, img_g, img_b ]
