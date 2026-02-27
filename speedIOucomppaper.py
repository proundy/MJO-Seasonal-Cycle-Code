import numpy as np
from numpy import linalg as LA
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date,timedelta
from scipy import io as io
#from mpl_toolkits.basemap import Basemap
from scipy.signal import detrend as detrend
from scipy.stats import ttest_1samp as ttest
from scipy.stats import pearsonr as corr
#mpl.rcParams['contour.negative_linestyle'] = 'solid'
mpl.rcParams.update({'font.size': 13})

'''For January 2026 MDPI Paper. This code calculates results for the 10N to 10S cross 
equatorial average. Phase speed spectra are calculated separately.'''

#Load RMM index and trim to remove data up to 1979. 
RMM=np.loadtxt('/roundylab_rit/roundy/RMMindex.txt')
cutter=(date(1979,1,1)-date(1974,6,1)).days
RMM=RMM[cutter:,:]
Iday=np.where(RMM[:,2]==15)[0]

d0=date(1979,1,1)
d1=date(2016,12,31)
L=(d1-d0).days


OLRarray=np.load('/roundylab_rit/roundy/projections/olrbig.npy') #My stored OLR data, 30N to 30S
OLRarray=OLRarray[cutter:,:,:]
Y=np.arange(-30,32.5,2.5)
Iy=np.where(np.abs(Y)<=10)[0]

A=OLRarray[:,Iy,:].mean(axis=1)

cut=(date(2020,12,31)-date(1979,1,1)).days
A=A[:cut,:]
olr=A.copy()
RMM=RMM[:cut,:]

ght=np.load('/roundylab_rit/roundy/era5/ght200.npy')*10 #My stored ERA5 1 degree geopotential data, scaled to geopotential height
ghtcp=ght.copy()
ght[:,:,:180]=ghtcp[:,:,180:] #Sort the longitude data to begin at 0E instead of 180E 
ght[:,:,180:]=ghtcp[:,:,:180]
Y=np.arange(-15,16) #Latitude grid for this geopotential height data array
I=np.where(np.abs(Y)<=10)[0]
ght=ght[:,I,:].mean(axis=1)

u=np.load('/roundylab_rit/roundy/era5/u200.npy')#My stored ERA5 1 degree 200 hPa u wind data
ucp=u.copy()
u[:,:,:180]=ucp[:,:,180:]
u[:,:,180:]=ucp[:,:,:180]
ucp=u.copy()
Y=np.arange(-15,16)
I=np.where(np.abs(Y)<=10)[0]
u=np.mean(u[:,I,:],axis=1)
ucp=u.copy()
u200std=u/np.std(u)

u850=np.load('/roundylab_rit/roundy/era5/u850.npy') #850 hPa zonal wind array
ucp850=u850.copy()
u850[:,:,:180]=ucp850[:,:,180:]
u850[:,:,180:]=ucp850[:,:,:180]
ucp850=u850.copy()
Y=np.arange(-15,16)
I=np.where(np.abs(Y)<=10)[0]
u850=np.mean(u850[:,I,:],axis=1)
ucp850=u850.copy()



harmonicnum=4  #Number of harmonics for seasonal cycle extraction. 
def harmbuild(N,harmonicnum):
     '''Build the Fourier harmonic matrix for removing seasonal cycle from data.'''
     t=np.arange(N)
     period=365.25
     j=1
     cycle=np.ones((N,2*harmonicnum+1))
     for i in np.arange(1,harmonicnum+1):
         cycle[:,j]=np.sin(i*2*np.pi*t/period)
         cycle[:,j+1]=np.cos(i*2*np.pi*t/period)
         j=j+2
     return cycle
X=harmbuild(ucp.shape[0],harmonicnum)
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(ucp))
uanom=ucp-X.dot(C)
C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(ucp850))
uanom850=ucp850-X.dot(C)


#Create standardized u850 and u200 data for the opposite signed difference test.
u850std=-uanom850/np.std(uanom850) #sign reversed U850, standardized, for Monte Carlo experiment
u200std=uanom/np.std(uanom) #sign reversed U850, standardized, for Monte Carlo experiment
udiff=u200std-u850std

C=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(ght))
ghtanom=ght-X.dot(C)


#Find the Background wind
#u=np.mean(u[:,50:91],axis=1)
def lowpass(u):
     umean=np.mean(u,axis=0)
     for x in np.arange(360):
          u[:,x]=u[:,x]-umean[x]
     fftu=np.fft.fft(u,axis=0)
     periods=u.shape[0]/np.arange(u.shape[0])
     I=np.where(periods>100)[0]
     fftufil=np.zeros_like(fftu)
     fftufil[I,:]=fftu[I,:]
     fftufil[-I,:]=fftu[-I,:]
     u=np.fft.ifft(fftufil,axis=0).real
     for x in np.arange(360):
        u[:,x]=u[:,x]+umean[x]
     return u

u=lowpass(u)
u850=lowpass(u850)



X=np.arange(0,360)
I=np.where(np.logical_and(X>=90,X<=120))[0]


IObackgroundu=np.mean(u[:,50:90],axis=1)
stdIObackgroundu=np.std(IObackgroundu)
IObackgroundu=IObackgroundu/stdIObackgroundu
Y=np.expand_dims(IObackgroundu,1)


xticks=np.arange(0,360,45)


#Find date indexes for RMM 3 events > amp 1 for Background wind in given percentile range:

Iall=np.where(np.logical_and(RMM[:,5]==3,RMM[:,6]>1.))[0]
I90=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,IObackgroundu>np.percentile(IObackgroundu,90))))[0]
I80=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu>np.percentile(IObackgroundu,80),IObackgroundu<np.percentile(IObackgroundu,90)))))[0]
I70=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,80),IObackgroundu>np.percentile(IObackgroundu,70)))))[0]
I60=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,70),IObackgroundu>np.percentile(IObackgroundu,60)))))[0]
I50=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,60),IObackgroundu>np.percentile(IObackgroundu,50)))))[0]
I40=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,50),IObackgroundu>np.percentile(IObackgroundu,40)))))[0]
I30=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,40),IObackgroundu>np.percentile(IObackgroundu,30)))))[0]
I20=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,30),IObackgroundu>np.percentile(IObackgroundu,20)))))[0]
I10=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,np.logical_and(IObackgroundu<np.percentile(IObackgroundu,20),IObackgroundu>np.percentile(IObackgroundu,10)))))[0]

I0=np.where(np.logical_and(RMM[:,5]==3,np.logical_and(RMM[:,6]>1.,IObackgroundu<=np.percentile(IObackgroundu,10))))[0]

ubackground=np.zeros((10,360))
u850background=np.zeros((10,360))
for perc in np.arange(10):
     ubackground[perc,:]=np.mean(u[eval('I'+str(perc*10)),:],axis=0)
     u850background[perc,:]=np.mean(u850[eval('I'+str(perc*10)),:],axis=0)


#Create the histograms and background composites for 10N to 10S. These are repeated in 
#separate code for 1-10N or 1-10S, by changing the latitude range of averaging above.

bins=np.arange(-25,7,2)
histout=np.zeros((bins.shape[0]-1,12))
histout3=np.zeros((bins.shape[0]-1,12))
print('IObackgroundu.max,min')
print(IObackgroundu.max()*stdIObackgroundu)
print(IObackgroundu.min()*stdIObackgroundu)

for month in np.arange(1,13):
     I = np.where(RMM[:,1]==month)[0]
     I3 = np.where(np.logical_and(RMM[:,1]==month,np.logical_and(RMM[:,5]==3,RMM[:,6]>1)))[0]
     a,b,p=plt.hist(IObackgroundu*stdIObackgroundu,bins)
     print('a.shape')
     print(a.shape)
     histout[:,month-1],edges=np.histogram(IObackgroundu[I]*stdIObackgroundu,bins=15,range=(-25,7))
     histout3[:,month-1],edges=np.histogram(IObackgroundu[I3]*stdIObackgroundu,bins=15,range=(-25,7))
print(histout)
months=np.arange(1,13)
plt.figure()
bincenters=np.zeros(bins.shape[0]-1)
for b in np.arange(edges.shape[0]-1):
     bincenters[b]=(edges[b]+edges[b+1])/2
levs=np.linspace(histout.min(),histout.max(),20)
levs3=np.linspace(histout3.min(),histout3.max(),20)
plt.contourf(months,bincenters,histout,levs)
plt.xlabel('Month')
plt.ylabel('Bin Center (m/s)')
plt.title('Seasonal Evolution of Background u')
plt.colorbar()
plt.savefig('/pr11/roundy/public_html/seasonaluhist.png')
plt.clf()
plt.contourf(months,bincenters,histout3,levs3)
plt.xlabel('Month')
plt.ylabel('Bin Center (m/s)')
plt.title('Seasonal Evolution of Background u')
plt.colorbar()
plt.savefig('/pr11/roundy/public_html/seasonaluhist3.png')

np.savez('monthlyhistall.npz',months=months,bincenters=bincenters,histout=histout,histout3=histout3,levs=levs)
plt.figure(figsize=(8,12))
plt.subplot(2,1,1)
V=np.arange(-30,32,2)
X=np.arange(360)
percs=np.arange(0,100,10)
plt.contourf(X,percs,ubackground,V,cmap='bwr')
#plt.xlabel('Longitude (Degrees East)')
plt.ylabel('u Wind Percentile')
plt.title('a. Composite 200 hPa Background Zonal Wind\n by Percentile at 70E')
plt.plot([70,70],[0,90],color='k')
plt.colorbar()
plt.subplot(2,1,2)
V=np.arange(-30,32,2)
X=np.arange(360)
percs=np.arange(0,100,10)
plt.contourf(X,percs,u850background,V,cmap='bwr')
plt.xlabel('Longitude (Degrees East)')
plt.ylabel('u Wind Percentile')
plt.title('b. Composite 850 hPa Background Zonal Wind\n by Percentile at 70E')
plt.plot([70,70],[0,90],color='k')
plt.colorbar()

plt.savefig('/pr11/roundy/public_html/backgroundwind.png')

#Store the 10N to 10S results. These allow me to later load them for plotting together with 
#similar results made for 1-10N or S. 
np.savez('monthlybackgroundall.npz',X=X,percs=percs,ubackground=ubackground,u850background=u850background,V=V)

plt.figure(figsize=(9,12))
print('u.max()')
print(u.max())

'''Make Composites:'''

maxlag=30
compositeall=np.zeros((2*maxlag+1,360))
composite90=np.zeros((2*maxlag+1,360))
composite80=np.zeros((2*maxlag+1,360))
composite70=np.zeros((2*maxlag+1,360))
composite60=np.zeros((2*maxlag+1,360))
composite50=np.zeros((2*maxlag+1,360))
composite40=np.zeros((2*maxlag+1,360))
composite30=np.zeros((2*maxlag+1,360))
composite20=np.zeros((2*maxlag+1,360))
composite10=np.zeros((2*maxlag+1,360))
composite0=np.zeros((2*maxlag+1,360))

compositeall850=np.zeros((2*maxlag+1,360))
composite90850=np.zeros((2*maxlag+1,360))
composite80850=np.zeros((2*maxlag+1,360))
composite70850=np.zeros((2*maxlag+1,360))
composite60850=np.zeros((2*maxlag+1,360))
composite50850=np.zeros((2*maxlag+1,360))
composite40850=np.zeros((2*maxlag+1,360))
composite30850=np.zeros((2*maxlag+1,360))
composite20850=np.zeros((2*maxlag+1,360))
composite10850=np.zeros((2*maxlag+1,360))
composite0850=np.zeros((2*maxlag+1,360))

compositeallolr=np.zeros((2*maxlag+1,144))
composite90olr=np.zeros((2*maxlag+1,144))
composite80olr=np.zeros((2*maxlag+1,144))
composite70olr=np.zeros((2*maxlag+1,144))
composite60olr=np.zeros((2*maxlag+1,144))
composite50olr=np.zeros((2*maxlag+1,144))
composite40olr=np.zeros((2*maxlag+1,144))
composite30olr=np.zeros((2*maxlag+1,144))
composite20olr=np.zeros((2*maxlag+1,144))
composite10olr=np.zeros((2*maxlag+1,144))
composite0olr=np.zeros((2*maxlag+1,144))

lags=np.arange(-maxlag,maxlag+1)
for lag in lags:
     compositeall850[lag+maxlag,:]=np.mean(uanom850[Iall+lag,:],axis=0)
     composite90850[lag+maxlag,:]=np.mean(uanom850[I90+lag,:],axis=0)
     composite80850[lag+maxlag,:]=np.mean(uanom850[I80+lag,:],axis=0)
     composite70850[lag+maxlag,:]=np.mean(uanom850[I70+lag,:],axis=0)
     composite60850[lag+maxlag,:]=np.mean(uanom850[I60+lag,:],axis=0)
     composite50850[lag+maxlag,:]=np.mean(uanom850[I50+lag,:],axis=0)
     composite40850[lag+maxlag,:]=np.mean(uanom850[I40+lag,:],axis=0)
     composite30850[lag+maxlag,:]=np.mean(uanom850[I30+lag,:],axis=0)
     composite20850[lag+maxlag,:]=np.mean(uanom850[I20+lag,:],axis=0)
     composite10850[lag+maxlag,:]=np.mean(uanom850[I10+lag,:],axis=0)
     composite0850[lag+maxlag,:]=np.mean(uanom850[I0+lag,:],axis=0)
     
     compositeall[lag+maxlag,:]=np.mean(uanom[Iall+lag,:],axis=0)
     composite90[lag+maxlag,:]=np.mean(uanom[I90+lag,:],axis=0)
     composite80[lag+maxlag,:]=np.mean(uanom[I80+lag,:],axis=0)
     composite70[lag+maxlag,:]=np.mean(uanom[I70+lag,:],axis=0)
     composite60[lag+maxlag,:]=np.mean(uanom[I60+lag,:],axis=0)
     composite50[lag+maxlag,:]=np.mean(uanom[I50+lag,:],axis=0)
     composite40[lag+maxlag,:]=np.mean(uanom[I40+lag,:],axis=0)
     composite30[lag+maxlag,:]=np.mean(uanom[I30+lag,:],axis=0)
     composite20[lag+maxlag,:]=np.mean(uanom[I20+lag,:],axis=0)
     composite10[lag+maxlag,:]=np.mean(uanom[I10+lag,:],axis=0)
     composite0[lag+maxlag,:]=np.mean(uanom[I0+lag,:],axis=0)
     
     compositeallolr[lag+maxlag,:]=np.mean(olr[Iall+lag,:],axis=0)
     composite90olr[lag+maxlag,:]=np.mean(olr[I90+lag,:],axis=0)
     composite80olr[lag+maxlag,:]=np.mean(olr[I80+lag,:],axis=0)
     composite70olr[lag+maxlag,:]=np.mean(olr[I70+lag,:],axis=0)
     composite60olr[lag+maxlag,:]=np.mean(olr[I60+lag,:],axis=0)
     composite50olr[lag+maxlag,:]=np.mean(olr[I50+lag,:],axis=0)
     composite40olr[lag+maxlag,:]=np.mean(olr[I40+lag,:],axis=0)
     composite30olr[lag+maxlag,:]=np.mean(olr[I30+lag,:],axis=0)
     composite20olr[lag+maxlag,:]=np.mean(olr[I20+lag,:],axis=0)
     composite10olr[lag+maxlag,:]=np.mean(olr[I10+lag,:],axis=0)
     composite0olr[lag+maxlag,:]=np.mean(olr[I0+lag,:],axis=0)
     

def sigtest(Icomp,udiff=udiff):
     events=np.zeros(Icomp.shape[0])
     tt=0
     events[0]=0
     for t in np.arange(1,Icomp.shape[0]):
          if Icomp[t]-Icomp[t-1]>1:
               tt=tt+1
          events[t]=tt
     eventinds=np.unique(events)
     print('The number of events is '+str(eventinds.max())+'\n\n\n')
     ucomp=np.zeros((2*maxlag+1,u200std.shape[1],1000))
     composite=np.zeros((2*maxlag+1,360))
     for lag in np.arange(-maxlag,maxlag+1):
          composite[lag+maxlag,:]=np.mean(udiff[Icomp+lag,:],axis=0)
     for i in np.arange(1000):
          #Irand=np.random.choice(Icomp,Icomp.shape[0])
          Irand=np.random.choice(eventinds,eventinds.shape[0])
          base=[]
          eventsdates=np.array([Icomp[j] for i in Irand for j in np.where(events==i)[0]])#eventdates are the date indexes of individual events
          for lag in np.arange(-maxlag,maxlag+1):
               #ucomp[lag+maxlag,:,i]=udiff[Irand+lag,:].mean(axis=0)
               ucomp[lag+maxlag,:,i]=udiff[eventsdates+lag,:].mean(axis=0)
     ucomp=np.sort(ucomp,axis=2)
     sigtestout=np.logical_or(0<ucomp[:,:,25],0>ucomp[:,:,975])
     return sigtestout

#Run Statistical Significance Testing for standardized difference between 850 and 200 hPa wind
sig0=sigtest(I0)
sig10=sigtest(I10)
sig20=sigtest(I20)
sig30=sigtest(I30)
sig40=sigtest(I40)
sig50=sigtest(I50)
sig60=sigtest(I60)
sig70=sigtest(I70)
sig80=sigtest(I80)
sig90=sigtest(I90)


'''The following figure was not included in the final manuscript, but gives the average
over all RMM phase 3 events of 850 and 200 hPa zonal wind.'''

plt.figure()
V=np.arange(-12.5,13.5,1)
V850=np.arange(-5,5.1,.5)

X=np.arange(360)
Vneg=np.arange(-10,0,1)
Vpos=np.arange(1,11)
xticks=np.arange(0,360,45)
xticklabs=xticks
yticks=np.arange(-30,40,10)
yticklabs=yticks.copy()

plt.clf()
plt.contourf(X,lags,compositeall,V,cmap='bwr')
plt.colorbar()
plt.contour(X,lags,compositeall850,V850[V850>=0],colors='r')
plt.contour(X,lags,compositeall850,V850[V850<=0],colors='b')

plt.plot([100,100],[-20,20],color='k',linewidth=1) 
plt.plot([150,150],[-20,20],color='k',linewidth=1) 
plt.title('All Events')
plt.yticks(yticks,yticks)
plt.xticks(xticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))
plt.xlabel('Longitude (Degrees East)')
plt.ylabel('Time Lag (Days)')

'''Build Figure 3:'''

plt.figure(figsize=(10,15))
plt.subplot(5,2,1)
plt.contourf(X,lags,composite90850,V850,cmap='bwr')
plt.colorbar()
plt.contourf(X,lags,sig90,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite90,V[V>=0],colors='r')
plt.contour(X,lags,composite90,V[V<=0],colors='b')
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)

plt.title('a. 90th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))


plt.subplot(5,2,2)
plt.contourf(X,lags,composite80850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig80,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite80,V[V>=0],colors='r')
plt.contour(X,lags,composite80,V[V<=0],colors='b')

plt.title('b. 80th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')

plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,3)
plt.contourf(X,lags,composite70850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig70,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite70,V[V>=0],colors='r')
plt.contour(X,lags,composite70,V[V<=0],colors='b')
plt.title('c. 70th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,4)
plt.contourf(X,lags,composite60850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig60,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite60,V[V>=0],colors='r')
plt.contour(X,lags,composite60,V[V<=0],colors='b')
plt.title('d. 60th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,5)
plt.contourf(X,lags,composite50850,V850,cmap='bwr')
plt.colorbar()
plt.contourf(X,lags,sig50,levels=[0.5,1],hatches='.',alpha=0)
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite50,V[V>=0],colors='r')
plt.contour(X,lags,composite50,V[V<=0],colors='b')
plt.title('e. 50th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,6)
plt.contourf(X,lags,composite40850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
#plt.contour(X,lags,composite40,V,colors='k')
plt.contourf(X,lags,sig40,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite40,V[V>=0],colors='r')
plt.contour(X,lags,composite40,V[V<=0],colors='b')
#plt.plot([100,100],[-20,20],color='k',linewidth=1) 
#plt.plot([150,150],[-20,20],color='k',linewidth=1) 
plt.title('f. 40th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,7)
plt.contourf(X,lags,composite30850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
#plt.contour(X,lags,composite30,V,colors='k')
plt.contourf(X,lags,sig30,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite30,V[V>=0],colors='r')
plt.contour(X,lags,composite30,V[V<=0],colors='b')
#plt.plot([100,100],[-20,20],color='k',linewidth=1) 
#plt.plot([150,150],[-20,20],color='k',linewidth=1) 
plt.title('g. 30th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,8)
plt.contourf(X,lags,composite20850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
#plt.contour(X,lags,composite20,V,colors='k')
plt.contourf(X,lags,sig20,levels=[0.5,1],hatches='.',alpha=0)
plt.contour(X,lags,composite20,V[V>=0],colors='r')
plt.contour(X,lags,composite20,V[V<=0],colors='b')
plt.title('h. 20th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,9)
plt.contourf(X,lags,composite10850,V850,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
#plt.contour(X,lags,composite10,V,colors='k')
plt.contourf(X,lags,sig10,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite10,V[V>=0],colors='r')
plt.contour(X,lags,composite10,V[V<=0],colors='b')
plt.title('i. 10th Percentile')
plt.xticks(xticks,xticks)
plt.yticks(yticks,yticks)
plt.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Time Lag (Days)')
plt.axis((0,180,-20,20))
plt.subplot(5,2,10)
plt.contourf(X,lags,composite0850,V850,cmap='bwr')
plt.colorbar()
plt.contourf(X,lags,sig0,levels=[0.5,1],alpha=0,hatches='.')
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite0,V[V>=0],colors='r')
plt.contour(X,lags,composite0,V[V<=0],colors='b')
plt.title('j. 0th Percentile')
plt.xticks(xticks,xticks)
plt.yticks(yticks,'')
plt.grid(True)
plt.xlabel('Longitude')
plt.axis((0,180,-20,20))
plt.savefig('/pr11/roundy/public_html/mjowindcompall.png')

Xolr=np.arange(0,360,2.5)
Volr=np.arange(-30,35,5)


'''Build Figure 4:'''

plt.figure(figsize=(10,15))
plt.subplot(5,2,1)
plt.contourf(Xolr,lags,composite90olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite90,V[V>=0],colors='r')
plt.contour(X,lags,composite90,V[V<=0],colors='b')

plt.title('a. 90th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))


plt.subplot(5,2,2)
plt.contourf(Xolr,lags,composite80olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite80,V[V>=0],colors='r')
plt.contour(X,lags,composite80,V[V<=0],colors='b')

plt.title('b. 80th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')

plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,3)
plt.contourf(Xolr,lags,composite70olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite70,V[V>=0],colors='r')
plt.contour(X,lags,composite70,V[V<=0],colors='b')
plt.title('c. 70th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,4)
plt.contourf(Xolr,lags,composite60olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite60,V[V>=0],colors='r')
plt.contour(X,lags,composite60,V[V<=0],colors='b')
plt.title('d. 60th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,5)
plt.contourf(Xolr,lags,composite50olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite50,V[V>=0],colors='r')
plt.contour(X,lags,composite50,V[V<=0],colors='b')
plt.title('e. 50th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))
plt.ylabel('Time Lag (Days)')

plt.subplot(5,2,6)
plt.contourf(Xolr,lags,composite40olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite40,V[V>=0],colors='r')
plt.contour(X,lags,composite40,V[V<=0],colors='b')
plt.title('f. 40th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,7)
plt.contourf(Xolr,lags,composite30olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite30,V[V>=0],colors='r')
plt.contour(X,lags,composite30,V[V<=0],colors='b')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.title('g. 30th Percentile')
plt.axis([0,180,-20,20])
plt.ylabel('Time Lag (Days)')
plt.subplot(5,2,8)
plt.contourf(Xolr,lags,composite20olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contour(X,lags,composite20,V[V>=0],colors='r')
plt.contour(X,lags,composite20,V[V<=0],colors='b')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.title('h. 20th Percentile')
plt.axis([0,180,-20,20])
plt.subplot(5,2,9)
plt.contourf(Xolr,lags,composite10olr,Volr,cmap='bwr')
plt.colorbar()
plt.contour(X,lags,composite10,V[V>=0],colors='r')
plt.contour(X,lags,composite10,V[V<=0],colors='b')
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.xticks(xticks,xticks)
plt.yticks(yticks,yticks)
plt.axis([0,180,-20,20])
plt.title('i. 10th Percentile')
plt.xlabel('Longitude')
plt.ylabel('Time Lag (Days)')
plt.subplot(5,2,10)
plt.contourf(Xolr,lags,composite0olr,Volr,cmap='bwr')
plt.colorbar()
plt.contour(X,lags,composite0,V[V>=0],colors='r')
plt.contour(X,lags,composite0,V[V<=0],colors='b')
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.xticks(xticks,xticks)
plt.yticks(yticks,'')
plt.xlabel('Longitude')
plt.title('j. 0th Percentile')
plt.axis([0,180,-20,20])
plt.savefig('/pr11/roundy/public_html/mjowindolrcompall.png')
plt.savefig('/pr11/roundy/public_html/mjowindolrcompall.eps',format='eps')


'''Build Figure 5:'''

plt.figure(figsize=(10,15))
plt.subplot(5,2,1)
plt.contourf(Xolr,lags,composite90olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig90,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite90850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite90850,V850[V850<=0],colors='b')

plt.title('a. 90th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))


plt.subplot(5,2,2)
plt.contourf(Xolr,lags,composite80olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig80,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite80850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite80850,V850[V850<=0],colors='b')

plt.title('b. 80th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')

plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,3)
plt.contourf(Xolr,lags,composite70olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig70,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite70850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite70850,V850[V850<=0],colors='b')
plt.title('c. 70th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.ylabel('Time Lag (Days)')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,4)
plt.contourf(Xolr,lags,composite60olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig60,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite60850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite60850,V850[V850<=0],colors='b')
plt.title('d. 60th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.subplot(5,2,5)
plt.contourf(Xolr,lags,composite50olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig50,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite50850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite50850,V850[V850<=0],colors='b')
plt.title('e. 50th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.grid(True)
plt.axis((0,180,-20,20))
plt.subplot(5,2,6)
plt.contourf(Xolr,lags,composite40olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig40,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite40850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite40850,V850[V850<=0],colors='b')
plt.title('f. 40th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))
plt.subplot(5,2,7)
plt.contourf(Xolr,lags,composite30olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig30,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite30850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite30850,V850[V850<=0],colors='b')
plt.title('g. 30th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,yticks)
plt.grid(True)
plt.axis((0,180,-20,20))
plt.subplot(5,2,8)
plt.contourf(Xolr,lags,composite20olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig20,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite20850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite20850,V850[V850<=0],colors='b')
plt.title('h. 20th Percentile')
plt.xticks(xticks,'')
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))
plt.subplot(5,2,9)
plt.contourf(Xolr,lags,composite10olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig10,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite10850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite10850,V850[V850<=0],colors='b')
plt.title('i. 10th Percentile')
plt.xticks(xticks,xticks)
plt.yticks(yticks,yticks)
plt.grid(True)
plt.axis((0,180,-20,20))
plt.subplot(5,2,10)
plt.contourf(Xolr,lags,composite0olr,Volr,cmap='bwr')
plt.colorbar()
plt.plot([45,90],[-5,5.56],color='k',linewidth=2)
plt.plot([45,90],[-9,9.27],color='r',linewidth=2)
plt.plot([45,90],[-3.6,3.6],color='b',linewidth=2)
plt.contourf(X,lags,sig0,levels=[0.5,1],alpha=0,hatches='.')
plt.contour(X,lags,composite0850,V850[V850>=0],colors='r')
plt.contour(X,lags,composite0850,V850[V850<=0],colors='b')
plt.title('j. 0th Percentile')
plt.xticks(xticks,xticks)
plt.yticks(yticks,'')
plt.grid(True)
plt.axis((0,180,-20,20))

plt.savefig('/pr11/roundy/public_html/mjowind850olrcompall.png')

plt.savefig('/pr11/roundy/public_html/mjowind850olrcompall.eps',format='eps')
