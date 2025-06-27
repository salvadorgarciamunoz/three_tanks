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


m            = pyo.ConcreteModel()
m.tf         = pyo.Param(initialize=100)
#probably better to create a set and then initialize on the sate
m.t          = dae.ContinuousSet(bounds=(0,m.tf))

# Time invariant constants
m.k          = pyo.Param(initialize=0.25)

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
    
m.dmass_solute_1eq[m.t.first()].deactivate()
m.dmass_solute_2eq[m.t.first()].deactivate()
m.dmass_solute_3eq[m.t.first()].deactivate()
m.dheight1eq[m.t.first()].deactivate()
m.dheight2eq[m.t.first()].deactivate()
m.dheight3eq[m.t.first()].deactivate()

#Provide boundary conditions at t=0
m.mass_solute_1[0].fix(10)
m.mass_solute_2[0].fix(0)
m.mass_solute_3[0].fix(0)
m.height1[0].fix(7)
m.height2[0].fix(10)
m.height3[0].fix(15)
m.obj = pyo.Objective(expr = 1)
#%%
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

conc_solute_1=get_pyo_var(time,m.conc_solute_1 )
conc_solute_2=get_pyo_var(time,m.conc_solute_2 )
conc_solute_3=get_pyo_var(time,m.conc_solute_3 )

height_1=get_pyo_var(time,m.height1 )
height_2=get_pyo_var(time,m.height2 )
height_3=get_pyo_var(time,m.height3 )


plt.figure()
plt.plot(time,conc_solute_1,label='Concentration in Tank 1' )
plt.plot(time,conc_solute_2,label='Concentration in Tank 2' )
plt.plot(time,conc_solute_3,label='Concentration in Tank 3' )
plt.legend()
plt.xlabel('Time (hr)')
plt.ylabel ('Concentration (kg/m3)')

plt.figure()
plt.plot(time,height_1,label='Height in Tank 1' )
plt.plot(time,height_2,label='Height in Tank 2' )
plt.plot(time,height_3,label='Height in Tank 3' )
plt.legend()
plt.xlabel('Time (hr)')
plt.ylabel ('Height (m)')

