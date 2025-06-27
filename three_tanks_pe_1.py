#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:27:33 2022

@author: c184156
"""
import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import pyphi as phi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

rng = np.random.default_rng()  #Random Number Generator


t_meas=[
0.0,
5.0,
10.0,
15.0,
20.0,
30.0,
40.0,
50.0,
60.0,
70.0,
80.0,
90.0,
100.0
]

conc_2_clean=np.asarray([
0.0000,
0.0354,
0.0671,
0.0847,
0.0927,
0.0979,
0.0975,
0.0958,
0.0946,
0.0938,
0.0934,
0.0932,
0.0931,
    ])

height_2_clean=np.asarray([
10.0,
10.2889,
10.6678,
11.0181,
11.3037,
11.8164,
12.2706,
12.7112,
13.0734,
13.4495,
13.7698,
14.1328,
14.4938])

std_dev_conc_noise    = .0025
std_dev_height_noise  = 0.01  


conc_w_uncertainty   = conc_2_clean   + rng.normal(0, std_dev_conc_noise,  len(conc_2_clean))
height_w_uncertainty = height_2_clean + rng.normal(0, std_dev_height_noise, len(height_2_clean))


conc_2_clean       = phi.np1D2pyomo(conc_2_clean,indexes=t_meas)
conc_w_uncertainty = phi.np1D2pyomo( conc_w_uncertainty,indexes=t_meas)

height_2_clean       = phi.np1D2pyomo(height_2_clean,indexes=t_meas )
height_w_uncertainty = phi.np1D2pyomo(height_w_uncertainty,indexes=t_meas )


conc_2_meas=conc_w_uncertainty
height_2_meas=height_w_uncertainty



m            = pyo.ConcreteModel()
m.tf         = pyo.Param(initialize=100)
#probably better to create a set and then initialize on the sate
m.t_meas     = pyo.Set(initialize = t_meas)

m.t          = dae.ContinuousSet(bounds=(0,m.tf),initialize=m.t_meas)

# Time invariant constants
m.k          = pyo.Var(domain=pyo.NonNegativeReals,initialize=0.1)

m.diam1      = pyo.Param(initialize=3)   # m
m.diam2      = pyo.Param(initialize=2)   # m
m.diam3      = pyo.Param(initialize=1.5) # m
m.area1      = pyo.Param(initialize = np.pi*(m.diam1**2)/4)
m.area2      = pyo.Param(initialize = np.pi*(m.diam2**2)/4)
m.area3      = pyo.Param(initialize = np.pi*(m.diam3**2)/4)

m.vol_flow_2 = pyo.Param(initialize=1.5) # m3/hr
m.vol_flow_3 = pyo.Param(initialize=1.6) # m3/hr

#Time invariante variables (functions of constants could be pre-calculted also)


#Time dependent variables
m.height1      = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize=7)
m.height2      = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize=10)
m.height3      = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize=15)

m.vol1          = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize = 8)
m.vol2          = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize = 8)
m.vol3          = pyo.Var(m.t, domain=pyo.NonNegativeReals,initialize = 8)

m.conc_solute_1 = pyo.Var(m.t, domain=pyo.NonNegativeReals)
m.conc_solute_2 = pyo.Var(m.t, domain=pyo.NonNegativeReals)
m.conc_solute_3 = pyo.Var(m.t, domain=pyo.NonNegativeReals)

m.mass_solute_1 = pyo.Var(m.t, domain=pyo.NonNegativeReals)
m.mass_solute_2 = pyo.Var(m.t, domain=pyo.NonNegativeReals)
m.mass_solute_3 = pyo.Var(m.t, domain=pyo.NonNegativeReals)


#Derivative variables
m.dheight1 = dae.DerivativeVar(m.height1,wrt=m.t)
m.dheight2 = dae.DerivativeVar(m.height2,wrt=m.t)
m.dheight3 = dae.DerivativeVar(m.height3,wrt=m.t)

m.dmass_solute_1 = dae.DerivativeVar(m.mass_solute_1,wrt=m.t)
m.dmass_solute_2 = dae.DerivativeVar(m.mass_solute_2,wrt=m.t)
m.dmass_solute_3 = dae.DerivativeVar(m.mass_solute_3,wrt=m.t)

#Differential equations
def _dheight1(m,t):
    return   m.dheight1[t] == (m.vol_flow_3 - m.k * m.height1[t])/m.area1
m.dheight1eq = pyo.Constraint(m.t, rule = _dheight1)

def _dheight2(m,t):
    return  m.dheight2[t] == (m.k * m.height1[t] - m.vol_flow_2)/m.area2
m.dheight2eq = pyo.Constraint(m.t, rule = _dheight2)

def _dheight3(m,t):
    return  m.dheight3[t] == (m.vol_flow_2 - m.vol_flow_3)/m.area3 
m.dheight3eq = pyo.Constraint(m.t, rule = _dheight3)

def _dmass_solute_1(m,t):
    return m.dmass_solute_1[t] == m.vol_flow_3 * m.conc_solute_3[t] - m.k * m.height1[t] * m.conc_solute_1[t]
m.dmass_solute_1eq = pyo.Constraint(m.t, rule = _dmass_solute_1)

def _dmass_solute_2(m,t):
    return m.dmass_solute_2[t] == m.k * m.height1[t] * m.conc_solute_1[t] - m.vol_flow_2 * m.conc_solute_2[t]
m.dmass_solute_2eq = pyo.Constraint(m.t, rule = _dmass_solute_2)

def _dmass_solute_3(m,t):
    return m.dmass_solute_3[t] == m.vol_flow_2 * m.conc_solute_2[t] - m.vol_flow_3 * m.conc_solute_3[t]
m.dmass_solute_3eq = pyo.Constraint(m.t, rule = _dmass_solute_3)


def _vol1(m,t):
    return m.vol1[t]==m.area1 * m.height1[t]
m.vol1eq = pyo.Constraint(m.t, rule = _vol1)

def _vol2(m,t):
    return m.vol2[t]==m.area2 * m.height2[t]
m.vol2eq = pyo.Constraint(m.t, rule = _vol2)

def _vol3(m,t):
    return m.vol3[t]==m.area3 * m.height3[t]
m.vol3eq = pyo.Constraint(m.t, rule = _vol3)

def _conc_solute_1(m,t):
    return m.conc_solute_1[t] * m.vol1[t] == m.mass_solute_1[t]
m.conc_solute_1eq = pyo.Constraint(m.t, rule =_conc_solute_1 )

def _conc_solute_2(m,t):
    return m.conc_solute_2[t] * m.vol2[t] == m.mass_solute_2[t]
m.conc_solute_2eq = pyo.Constraint(m.t, rule =_conc_solute_2 )

def _conc_solute_3(m,t):
    return m.conc_solute_3[t] * m.vol3[t] == m.mass_solute_3[t]
m.conc_solute_3eq = pyo.Constraint(m.t, rule =_conc_solute_3 )


#Provide boundary conditions at t=0
m.mass_solute_1[0].fix(10)
m.mass_solute_2[0].fix(0)
m.mass_solute_3[0].fix(0)
m.height1[0].fix(7)
m.height2[0].fix(10)
m.height3[0].fix(15)

def _obj  (m):
     return sum((m.conc_solute_2 [i]- conc_2_meas[i])**2 +
                 (m.height2[i]- height_2_meas[i])**2
                 for i in t_meas)


# def _obj (model):
#     return sum((m.height2[i]- height_2_meas[i])**2 for i in t_meas)

# def _obj (model):
#     return sum((m.conc_solute_2 [i]- conc_2_meas[i])**2  for i in t_meas)


m.obj = pyo.Objective(rule = _obj)


# Simulate with Pyomo
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=20,ncp=5,scheme='LAGRANGE-RADAU')

pyo.SolverFactory('ipopt').solve(m,tee=True)


#Retreive the values from the model

time=[]
for i in np.arange(1,len(m.t)+1):
    time.append(m.t.at(i))
time=np.array(time)

def get_pyo_var(time,pyovar):
    x=[]
    for i in time:
        x.append(pyovar[i].value)
    x=np.array(x)
    return x
conc_solute_2=get_pyo_var(time,m.conc_solute_2 )


# height_1=get_pyo_var(time,m.height1 )
height_2=get_pyo_var(time,m.height2 )
# height_3=get_pyo_var(time,m.height3 )

c2m=[]
for ts in conc_2_meas.keys():
    c2m.append(conc_2_meas[ts])
    
h2m=[]
for ts in height_2_meas.keys():
    h2m.append(height_2_meas[ts])
        
fig,ax=plt.subplots(1,2)
ax[0].plot(time,conc_solute_2,label='Concentration in Tank 2' )
ax[0].plot(t_meas,c2m,'.',label='Measured Conc in Tank 2' )
ax[0].legend()
ax[0].set_xlabel('Time (hr)')
ax[0].set_ylabel ('Concentration (kg/m3)')
fig.suptitle('K estimated :'+str(np.round(m.k.value,4) ))


ax[1].plot(time,height_2,label='Height in Tank 2' )
ax[1].plot(t_meas,h2m,'.',label='Measured Height in Tank 2' )
ax[1].legend()
ax[1].set_xlabel('Time (hr)')
ax[1].set_ylabel ('Hight (m)')

#%%


def plot_colored_lines(X, z, x=None, xlabel='Index', ylabel='Value',
                       title='Color-coded Lines using Viridis Colormap',
                       show_grid=True, colorbar_label='Scalar value (z)'):
    """
    Plots each column of matrix X as a line, color-coded using the viridis colormap
    according to the corresponding value in vector z. Includes a colorbar and optional labels.

    Parameters:
    X (ndarray): 2D array where each column represents a line to be plotted.
    z (ndarray): 1D array of scalar values used for color-coding each line.
    x (ndarray or list): Optional array of x-values for plotting. Must match the number of rows in X.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    show_grid (bool): Whether to show vertical and horizontal gridlines.
    colorbar_label (str): Label for the colorbar.
    """
    X = np.asarray(X)
    z = np.asarray(z)
    if x is None:
        x_vals = np.arange(X.shape[0])
    else:
        x_vals = np.asarray(x)
        if x_vals.shape[0] != X.shape[0]:
            raise ValueError("Length of x must match the number of rows in X.")

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(X.shape[1]):
        ax.plot(x_vals, X[:, i], color=cmap(norm(z[i])))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_grid:
        ax.grid(True, color='lightgray')

    plt.show()



ofvals=[]
kvals=np.linspace(0.1,0.7,50)
heights=[]
concs=[]
for kval in kvals :
    def kconst_(m): return m.k == kval
    m.kconst =  pyo.Constraint(rule=kconst_(m))
        
    pyo.SolverFactory('ipopt').solve(m,tee=True)

    ofvals.append(pyo.value(m.obj))
    time=[]
    for i in np.arange(1,len(m.t)+1):
        time.append(m.t.at(i))
    time=np.array(time)
    
    def get_pyo_var(time,pyovar):
        x=[]
        for i in time:
            x.append(pyovar[i].value)
        x=np.array(x)
        return x
    conc_solute_2=get_pyo_var(time,m.conc_solute_2 )
    
    
    # height_1=get_pyo_var(time,m.height1 )
    height_2=get_pyo_var(time,m.height2 )
    # height_3=get_pyo_var(time,m.height3 )
    
    c2m=[]
    for ts in conc_2_meas.keys():
        c2m.append(conc_2_meas[ts])
        
    h2m=[]
    for ts in height_2_meas.keys():
        h2m.append(height_2_meas[ts])

    
    heights.append(height_2)
    concs.append(conc_solute_2)
    
    m.del_component(m.kconst)

plt.figure()
plt.plot(kvals,ofvals )
plt.xlabel('Objective Function')
plt.ylabel('K parameter value')
plt.title('Error Function Map')

heights=np.asarray(heights).T
concs=np.asarray(concs).T

plot_colored_lines(heights,ofvals,x=time,
                   xlabel='Time',
                   ylabel='Height (m)',
                   title='Height as K varies',
                   colorbar_label='K Value')
ax=plt.gca()
ax.plot(t_meas,h2m,'ro')

plot_colored_lines(concs,ofvals,x=time,
                   xlabel='Time',
                   ylabel='Height (m)',
                   title='Concentration 2 as K varies',
                   colorbar_label='K Value')
ax=plt.gca()
ax.plot(t_meas,c2m,'ro')





