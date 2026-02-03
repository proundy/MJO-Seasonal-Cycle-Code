import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
from scipy import io as io
from scipy.signal import detrend as detrend
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

'''Script finds the phase speed spectra of near equatorial OLR anomaly data.'''


#Create color maps for figures:

#jet_colors = mpl.colormaps['gist_ncar'].resampled(256)(np.linspace(0, 1, 256))
jet_colors = mpl.colormaps['bwr'].resampled(256)(np.linspace(0, 1, 256))

# 2. Change the first 20 bins (index 0-19) to white [R, G, B, Alpha]
for l in np.arange(20):
     jet_colors[l] = [1, 1, 1, 1]

# 3. Create the new colormap
white_jet = ListedColormap(jet_colors)

mpl.rcParams['contour.negative_linestyle'] = 'dashed'
mpl.rcParams.update({'font.size': 13})

#You will need to provide your own ERA5 data arrays

'''Block of code below finds low frequency background index, not presently used
in the manuscript, but allows for regressing wavelet phase speed against 
background wind for a separate project.'''

uarray=np.load('/roundylab_rit/roundy/era5/uwind200global.npy')
uarray=np.concatenate((uarray[:,180:],uarray[:,:180]),axis=1)
print(uarray.shape)

Y=np.arange(-90,91)
Iy=np.where(np.abs(Y)<=10)[0]
uarray=uarray[:,Iy,:].mean(axis=1)
print(uarray.shape)
print((date(2020,12,31)-date(1979,1,1)).days)
uarray=np.concatenate((uarray[:,180:],uarray[:,:180]),axis=1)
uarraylow=np.zeros_like(uarray)
maxlag=60
for t in np.arange(maxlag,uarray.shape[0]-maxlag-1):
     uarraylow[t,:]=uarray[t-maxlag:t+maxlag+1,:].mean(axis=0)

uarraylow=np.mean(uarraylow[:,40:100],axis=1)

#Loads the RMM index, also not presently used in this application. I kept it 
#here for convenience. 

cutter=(date(1979,1,1)-date(1974,6,1)).days
RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt')[cutter:,:]
Yolr=np.arange(-30,32.5,2.5)
Iy=np.where(np.abs(Yolr)<=10)[0]

olr=np.load('/roundylab_rit/roundy/projections/olrbig.npy')
olr=olr[:,Iy,:].mean(axis=1) #Average OLR data in latitude from 10N to 10S
olrsm=olr.copy()

#Subtract the 121-day centered moving average from the OLR data to remove
# low frequency signals.
for t in np.arange(1,olr.shape[0]-1):
     olr[t,:]=olrsm[t-1:t+1,:].mean(axis=0)-olrsm[t-60:t+61,:].mean(axis=0)



Xolr=np.arange(0,360,2.5) #OLR longitude grid
Idiff=np.arange(1,olr.shape[0]-1)
olrdiff=np.zeros_like(olr)
olrdiff[Idiff,:]=olr[Idiff+1,:]-olr[Idiff-1,:]
#Make data begin January 1, 1979
olr=olr[cutter:,:]
olrdiff=olrdiff[cutter:,:]
print(olr.max())
print(olr.min())
Xera=np.arange(360)

print(olr.shape)

baseindex=RMM[:,4:5]

#uarray and zarray are 200 hPa u and geopotential data from ERA5, 
#averaged to a 1 degree latitude-longitude grid. Data begin Jan 1, 1979.
#Your data arrays should be oriented time (days) x latitude x longitude.
uarray=np.load('/roundylab_rit/roundy/era5/uwind200global.npy')
zarray=np.load('/roundylab_rit/roundy/era5/ght200global.npy') #not presently used

Y=np.arange(-90,91)
Iy=np.where(np.abs(Y)<=5)[0]
print(uarray.shape)
print(zarray.shape)
zcut=(date(1979,1,1)-date(1974,1,1)).days
zarray=zarray[zcut:,:,:]
uarray=uarray[:,Iy,:].mean(axis=1)
zarray=zarray[:,Iy,:].mean(axis=1)
print((date(2020,12,31)-date(1974,1,1)).days)
print('uarray.shape,zarray.shape')
plt.contourf(uarray[5000:6000,:],20)
plt.savefig('/pr11/roundy/public_html/exam1.png')

print(uarray.shape)
print(zarray.shape)

uarray=np.concatenate((uarray[:,180:],uarray[:,:180]),axis=1)
zarray=np.concatenate((zarray[:,180:],zarray[:,:180]),axis=1)
d0=date(1979,1,1)
d1=date(2020,12,31)
dates=np.array([d0+timedelta(days=int(t)) for t in np.arange((d1-d0).days)])
olr=olr[:uarray.shape[0],:]
olrdiff=olrdiff[:uarray.shape[0],:]


'''The spectrtum analysis below was done for both u wind data and OLR data,
but only olr data were used in the paper. I repeated the data 3 times around the world in order to prevent the wavelet envelope from ever exceeding the edge of the world, though a similar result could be achieved with cleaver indexing. If you want to run on the u wind data, you will need to revise the indmake function, replacing each occurrance of 144 (the zonal grid size of the OLR data) with 360, for the 1 degree grid of the wind data). 
'''


#A=np.hstack((uarray,uarray,uarray)) #Repeat the data going around the world 3 times to perform the wavelet transform. 
A=np.hstack((olr,olr,olr)) #Repeat the data going around the world 3 times to perform the wavelet transform. 


Xvals=np.array([80]) # Longitude used as the wavelet center.

plt.figure(10,figsize=(10,7))
plt.figure(11,figsize=(10,5))
def indmake(A,speed,k):
     speedindexes=np.zeros((A.shape[0],Xvals.shape[0]))
     Re=6.371e6
     t=np.arange(-100,101)
     psi=np.zeros((201,144*3))
     psii=np.zeros((201,144*3))
     fbx=7000.  #Sets the zonal envelope width.
     fbt=7000.   #Sets the temporal envelope width.
     fcx=k/144.  #Sets the central zonal wavenumber of the wavelet. 

     fct=(k/144.)*(144./(2*np.pi*Re))*speed*86400   #Setting the phase speed sets the frequency. 
     for xx in np.arange(Xvals.shape[0]):
          x=np.arange(144*3)*2.5-(Xvals[xx]+360)  #Center on 80E (treat that as the zero longitude.
          for xxx in np.arange(144*3):
            #for  tt in np.arange(201):
              psi[t+100,xxx]=((np.pi*fbx)**(-.5))*((np.pi*fbt)**(-.5))*np.cos(2*np.pi*(fcx*x[xxx]-fct*t))*np.exp(-x[xxx]**2/fbx)*np.exp(-(t**2)/fbt)
              psii[t+100,xxx]=-((np.pi*fbx)**(-.5))*((np.pi*fbt)**(-.5))*np.sin(2*np.pi*(fcx*x[xxx]-fct*t))*np.exp(-x[xxx]**2/fbx)*np.exp(-(t**2)/fbt)
          psi=psi/np.sum(np.abs(psi))          
          psii=psii/np.sum(np.abs(psii))          
          for tind in np.arange(101,np.shape(A)[0]-100):
              Aseg=detrend(A[tind-100:tind+101,:],axis=0)
              speedindexes[tind,xx]=np.sum(Aseg*psi)**2+np.sum(Aseg*psii)**2
     return speedindexes.squeeze()

'''I calculated the wavelet filtered base indexes once, then saved the data and commented out the calculation lines and loaded stored copies.'''

speeds=np.arange(1,15,.5)
indexes=np.zeros((A.shape[0],speeds.shape[0]))

speedpower=np.zeros((365,speeds.shape[0]))
speedpowerk=np.zeros((365,speeds.shape[0],3))

F=np.load('monthlyhistall.npz')
months=F['months']
bincenters=F['bincenters']
histout=F['histout']
histout3=F['histout3']
levshist=F['levs']

harmonicnum=4  #Number of harmonics for seasonal cycle extraction. 
def harmbuild(N,harmonicnum):
     t=np.arange(N)
     period=365.25
     j=1
     cycle=np.ones((N,2*harmonicnum+1))
     for i in np.arange(1,harmonicnum+1):
         cycle[:,j]=np.sin(i*2*np.pi*t/period)
         cycle[:,j+1]=np.cos(i*2*np.pi*t/period)
         j=j+2
     return cycle
X=harmbuild(indexes.shape[0],harmonicnum)
letters=np.array(['a. ','b. ','c. '])
xtics=np.arange(0,360,50)
lmonths=np.array((31,28,31,30,31,30,31,31,30,31,30,31))
monthsdoy=np.cumsum(lmonths)
print('monthsdoy.shape')
print(monthsdoy.shape)
for k in np.arange(1,3):
     filepath=Path('/roundylab_rit/roundy/speedreg/olrwaveletspeeds808k'+str(k)+'.npz')
     if not filepath.is_file():
         for i in np.arange(speeds.shape[0]):
               print(i)
               indexes[:,i]=indmake(A,speeds[i],k)

         np.savez('olrwaveletspeeds808k'+str(k)+'.npz',indexes=indexes,speeds=speeds)
     F=np.load('olrwaveletspeeds808k'+str(k)+'.npz')
     indexes=F['indexes']
     indexescopy=indexes.copy()
     speeds=F['speeds']

     C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(indexes))
     cycle=X[:365,:].dot(C)
     plt.figure(10)
     plt.subplot(2,1,k)
     if k==1:
          V=np.arange(5,22,1)
     else: 
          V=np.arange(2,5,.5)

     plt.contourf(np.arange(365),speeds,cycle.T,V,cmap=white_jet)
     plt.grid(True)
     plt.colorbar()
     plt.contour(monthsdoy-15,bincenters,histout,levshist,cmap=white_jet,widths=2)
     plt.contour(monthsdoy-15,bincenters,histout,levshist,colors='k',linewidths=.5)
     plt.title(letters[k-1]+'Seasonal Cycle of Phase Speed Spectrum, k='+str(k))
     #plt.title('Seasonal Cycle of Phase Speed Spectrum, k='+str(k))
     if k==2:
          plt.xlabel('Month')
          plt.xticks(monthsdoy,months)
     else:
          plt.xticks(monthsdoy,'')
     plt.ylabel('Speed (m/s)')
     speedpower=speedpower+cycle
     speedpowerk[:,:,k-1]=cycle

plt.savefig('/pr11/roundy/public_html/exam123.png')

plt.figure(11)
plt.clf()
plt.contourf(np.arange(365),speeds,speedpower.T,V,cmap=white_jet)
plt.colorbar()
plt.title('Seasonal Cycle of Phase Speed Spectrum, Waves 1-2')
plt.xlabel('Day of Year')
plt.ylabel('Speed (m/s)')
plt.savefig('/pr11/roundy/public_html/exam1213.png')
plt.clf()


#Normalizes power at each wavenumber by the sum of power at each wavenumber.
#Not used in the final product for the paper. 
powernorm=np.zeros_like(speedpower)
for d in np.arange(speedpower.shape[0]):
     powernorm[d,:]=speedpower[d,:]/np.sum(speedpower[d,:])



plt.contourf(np.arange(365),speeds,powernorm.T,20,cmap=white_jet)
plt.colorbar()
plt.title('Seasonal Cycle of Phase Speed Spectrum, Waves 1-3')
plt.xlabel('Day of Year')
plt.ylabel('Speed (m/s)')
plt.savefig('/pr11/roundy/public_html/exam1313.png')


