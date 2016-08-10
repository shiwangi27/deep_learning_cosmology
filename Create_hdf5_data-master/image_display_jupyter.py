from astropy.io import fits
import matplotlib 
from matplotlib import pyplot as plt

hdulist = fits.open("/Users/ssingh79/Downloads/DeepLearningTutorials-master/data/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/fico_0200z_conv_0001.fit")
image = hdulist[0].data

%matplotlib inline
plt.imshow(image,interpolation='None') say 
plt.colorbar()