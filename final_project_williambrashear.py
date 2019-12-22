#!/usr/bin/env python
# coding: utf-8

# In[3]:


#William Brashear, CourseID: PHZ3150, Final Project, 11/27/2019
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle


# In[4]:


print("Problem 1: ")
#Actual function
def star(T,resolution):
    """Creates the stellar spectrum for a star of
    given temperature T (in K) between 0.5μm and 10μm. """
    lambda1 = np.linspace(0.5*1e-6,10*1e-6, num = 300)
    lambda2 = np.array((resolution*lambda1)/(resolution-1))
    Spect = (2*6.62607004*10**-34*3.00*10**8)/(lambda2**5*    (np.exp((6.62607004*10**-34*3.00*10**8)/(lambda2*1.38064852 * 10**-23*T))-1))
    return Spect
lambda1 = np.linspace(0.5*1e-6,10*1e-6,num=300)
# plot
plt.figure(figsize=(12,12))
plt.title('The stellar spectrum for a star given Tempurature (in K) between 0.5μm and 10μm.', fontsize='20', color='white')
plt.xlabel('Wavelength in m', fontsize='20', color='white')
plt.ylabel('Energy per unit wavelength λ', fontsize='20', color='white')
plt.xticks(fontsize='15', color='white')
plt.yticks(fontsize='20', color='white')
a=plt.plot(lambda1,star(2000,100),label='T = 2000 K',linestyle='--')
b=plt.plot(lambda1,star(5000,100), label = 'T = 5000 K',linestyle='-')
c=plt.plot(lambda1,star(6000,100), label = 'T = 6000 K',linestyle=':')
d=plt.plot(lambda1,star(7000,100), label = 'T = 7000 K',linestyle='-.')
e=plt.plot(lambda1,star(8000,100), label = 'T = 8000 K')
f=plt.plot(lambda1,star(10000,100), label = 'T = 10000 K')
plt.xlim(0, 10*1e-6)
SpectKnightro = np.loadtxt("Knightro-2019.spec")
Wavelen=SpectKnightro[:,0]*1e-6
Spect=SpectKnightro[:,1]*(10**-3)
plt.plot(Wavelen,Spect, label='Knightro-2019' )
plt.legend(prop={'size':'20'})


# In[5]:


print("Problem 1b: ")
# inital load
SpectKnightro = np.loadtxt("Knightro-2019.spec")
Wavelen=SpectKnightro[:,0]*1e-6
Spect=SpectKnightro[:,1]*1e-3

# plotted figure
plt.figure(figsize=(12,12))
plt.title('The stellar spectrum of Knightro-2019', fontsize='20', color='white')
plt.xlabel('Wavelength in m', fontsize='20', color='white')
plt.ylabel('Energy per unit wavelength λ', fontsize='20', color='white')
plt.xticks(fontsize='15', color='white')
plt.yticks(fontsize='20', color='white')
plt.plot(Wavelen,Spect)
plt.xlim(0,10*1e-6)


# In[6]:


print("Problem 2a: ")
# list of Albedos
A=[0.1,0.3,0.6]
fin = Spect
fout1 = A[0] * fin
fout2 = A[1] * fin
fout3 = A[2] * fin
lambda1 = np.arange(0.5*1e-6, 10*1e-6, 0.1)

# Figure plotted here
plt.figure(figsize=(12,12))
plt.title('The stellar spectrum of Knightro-2019 with differing Albedos', fontsize='20', color='white')
plt.xlabel('Wavelength in m', fontsize='20', color='white')
plt.ylabel('Energy per unit wavelength λ', fontsize='20', color='white')
plt.xticks(fontsize='15', color='white')
plt.yticks(fontsize='20', color='white')
plt.plot(Wavelen,Spect, label = 'Original spectrum')
plt.plot(Wavelen, fout1, label= 'Spectrum with Albedo of 0.1')
plt.plot(Wavelen, fout2,label= 'Spectrum with Albedo of 0.3')
plt.plot(Wavelen, fout3, label= 'Spectrum with Albedo of 0.6')
plt.legend(prop={'size':'20'})
plt.xlim(0,5*1e-6)




# In[7]:


print("Problem 2b part 1: ")

SpectKnightro = np.loadtxt("Knightro-2019.spec")
Wavelen=SpectKnightro[:,0]
region1 = Wavelen
Spect=SpectKnightro[:,1]
tau = tuple(np.arange(0,100,0.5))
# The exact flux at 0.7 micrometers
f01= 1.38800e3
# The exact flux at 1.1 micrometers
f02 = 6.02500e2
fout1 = f01 *(1/np.exp(tau))
fout2 = f02 * (1/np.exp(tau))
print(fout1)
print(fout2)


# In[8]:


print("Problem 2b part 2 and 3: ")

def base_model(lightin, tau, tau2, A):
    fout1 = lightin *(1/np.exp(tau))
    fout2 = fout1 * A
    fout3 = fout2 * A
    fout4 = fout3 *(1/np.exp(tau2))
    return fout4
print(base_model(1.85900e3, 12,10,0.3))

    


# In[24]:


print("Problem 2 Bonus Question: ")
Wavelen=SpectKnightro[:,0]
forest = np.loadtxt('forest_surface.dat')
spectadj = np.interp(forest[:,0], Wavelen, Spect)
# Pretty easy question not gonna lie
fout = spectadj * forest[:,1]
print(fout)


# In[55]:


print("Problem 2b part 4: ")
import scipy.stats as stats
import math

mu1 = 0.765
sigma = .04

x1 = np.linspace(mu1 - 3*sigma, mu1 + 3*sigma, 100)

mu2 = 0.688
sigma = .04

x2 = np.linspace(mu2 - 1*sigma, mu2 + 1*sigma, 100)

mu3 = .325
sigma1 = .20

def fres(x,x2,mu1,mu2,sigma):
    fres = 1
    if x == (mu1 - 3*sigma):
        fres = 1-x
        if x == (mu2 - 1*sigma):
            fres = 1-x2
            
# plotted figure, but clearly I can't figure out how to plot absorbtion lines
x3 = np.linspace(mu3-5*sigma1 ,mu3 + 5*sigma1, 100)
plt.figure(figsize= (12,12))
plt.plot(x3,stats.norm.pdf(x3,mu3,sigma1) )
plt.xlim(0.3, 1.0)
plt.xlabel('Wavelength [μm]', fontsize='20', color='white')
plt.ylabel('Flux [W/m/Hz]', fontsize='20', color='white')
plt.xticks(fontsize='15', color='white')
plt.yticks(fontsize='20', color='white')


# In[ ]:





# In[10]:


print("Problem 3a.: ")
bc = np.loadtxt("planet_lightcurves.dat")

# Calculating using Keplers third law
radiusb = np.sqrt(0.000045*.95**2)
print(radiusb)
radiusc = np.sqrt(0.00002*0.95**2)
print(radiusc)

orbdistb=(350)**1.5
print(orbdistb)
orbdistc=(350)**1.5
print(orbdistc)


# In[11]:


print("Problem 3c: ")
import pickle
from scipy.interpolate import interp1d
from scipy.stats import chisquare
models = open('model_runs.pickle','rb')
model_pickle = pickle.load(models)

spectbc = np.loadtxt('knightro_2019_bc_photometry.dat')
yb = spectbc[:,1]
yc = spectbc[:,2]
x = spectbc[:,0]
x1 = model_pickle['wavelength']
y_int = np.interp(x1, x, yb)

# All the models!!
y1 = model_pickle['forest_clear_atmosphere.dat']
y2 = model_pickle['forest_t10_atmosphere.dat']
y3 = model_pickle['grass_clear_atmosphere.dat']
y4 = model_pickle['grass_t10_atmosphere.dat']
y5 = model_pickle['ice_clear_atmosphere.dat']
y6 = model_pickle['ice_t10_atmosphere.dat']
y7 = model_pickle['ocean_clear_atmosphere.dat']
y8 = model_pickle['ocean_t30_atmosphere.dat']
y9 = model_pickle['sand_clear_atmosphere.dat']
y10 = model_pickle['sand_t10_atmosphere.dat']

chisquaresb = [chisquare( y_int, f_exp=y1),chisquare( y_int, f_exp=y2),              chisquare( y_int, f_exp=y3),chisquare( y_int, f_exp=y4),chisquare( y_int, f_exp=y5),chisquare( y_int, f_exp=y6),chisquare( y_int, f_exp=y7),chisquare( y_int, f_exp=y8),chisquare( y_int, f_exp=y9),chisquare( y_int, f_exp=y10)]
# best fit for b is y10, sand_t10_atmosphere

y_intc = np.interp(x1, x, yc)
chisquaresc = [chisquare( y_intc, f_exp=y1),chisquare( y_intc, f_exp=y2),              chisquare( y_intc, f_exp=y3),chisquare( y_intc, f_exp=y4),chisquare( y_intc, f_exp=y5),chisquare( y_intc, f_exp=y6),chisquare( y_intc, f_exp=y7),chisquare( y_intc, f_exp=y8),chisquare( y_intc, f_exp=y9),chisquare( y_intc, f_exp=y10)]
for i in range(len(chisquaresc)):
    print(chisquaresc[i])
# best fit for c is y9, sand_clear_atmosphere


# In[ ]:




