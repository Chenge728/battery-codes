# -*- coding: utf-8 -*-
"""
Created on Apr 22nd, 2025

@author:gcg
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import csv
import random
from io import StringIO
from scipy.optimize import minimize

#%% Global settings

random.seed(42)  # set the random seed
np.random.seed(42)  # set the random seed
N_t = 24  # The number of time intervals in one day
monthlist = np.array(['June','July','August','September','October','November','December','January',
                      'February','March','April','May'])  # From 2024.06 to 2025.05
monthnumlist = np.array(['06','07','08','09','10','11','12','01','02','03','04','05'])  # From 2024.06 to 2025.05
num = 11  # Choose the month from the monthlist
N_day = 30  # Adjust the variable with the month choice
month = monthlist[num]  # Load the month
monthnum = monthnumlist[num]  # Load the month
carbonmonth = num  # Load the month
eta = 0.95 # Set the efficiency of the equivalent battery
Smax = 1 # Set the maximum SOC of the equivalent battery
Smin = 0 # Set the minimum SOC of the equivalent battery

#%% Load data of CAISO

# 20240627 20240829 20240924 20250309 20250603 # The data of these days is null from CAISO 
def readdata(Ntall,Nt,Nday):
    df_price = pd.read_csv('2025' + monthnum + ' CAISO Average Price.csv')  # real time average LMP
    price = df_price['price'].values
    load = np.zeros((Ntall*Nday))  # real time load
    battery = np.zeros((Ntall*Nday))  # real time battery discharge power (>0 discharge;<0 charge)
    gas = np.zeros((Ntall*Nday))  # real time gas power

    
    for d in range(Nday): # cut the data by days
        if d+1<=9: 
            date = monthnum+'0'+str(d+1)
        else:
            date = monthnum+str(d+1)
        print(date)
        if date != '0627' and date != '0829' and date != '0309' and date != '0924': # skip the null data
            df_load = pd.read_csv('load/CAISO-netdemand-2025'+date+'.csv')  # read the real time load
            df_load = df_load.set_index(df_load.columns[0])  
            load[Ntall*d:Ntall*(d+1)] = df_load.loc['Net demand'].iloc[:-1].values
            df_ess = pd.read_csv('battery_data/CAISO-batteries-2025'+date+'.csv')  # read the real time battery power
            df_ess = df_ess.set_index(df_ess.columns[0])
            battery[Ntall*d:Ntall*(d+1)] = df_ess.loc['Total batteries',:]
            
            df_cu = pd.read_csv('curtailment/curtailment_'+month+'.csv')  # read the curtailment of renewables
            curtailment = df_cu['total_curtailment_mwh'].to_numpy()
            
    # transform data sampled per 5 minutes to data sampled per hour
    points_per_hour = 12 
    n_hours_price = len(price) // points_per_hour 
    n_hours_load = len(load) // points_per_hour 
    n_hours_battery = len(battery) // points_per_hour 
    n_hours_gas = len(gas) // points_per_hour 
    price = price[:n_hours_price * points_per_hour]
    load = load[:n_hours_load * points_per_hour]
    battery = battery[:n_hours_battery * points_per_hour]
    gas = gas[:n_hours_gas * points_per_hour]
    hourly_price = price.reshape(n_hours_price, points_per_hour).mean(axis=1)
    hourly_load = load.reshape(n_hours_load, points_per_hour).mean(axis=1)
    hourly_battery = battery.reshape(n_hours_battery, points_per_hour).mean(axis=1)
    hourly_gas = gas.reshape(n_hours_gas, points_per_hour).mean(axis=1)
    
    # calculate the maximum discharge power and charge power
    Pdmax = np.ceil(max( hourly_battery)/100)*100 
    Pcmax = Pdmax
    
    # calculate the capacity and the SOC at 0:00 on the first day of the month
    E = np.zeros((Nt*Nday+1))
    for i in range(1,Nt*Nday+1):
        if  hourly_battery[i-1] < 0: 
            E[i] = E[i-1] -  hourly_battery[i-1]*eta
        else:
            E[i] = E[i-1] -  hourly_battery[i-1]/eta
    Emax = max(E)
    Emin = min(E)
    Cap = 1000 *np.ceil((Emax-Emin)/1000)
    SOC = E/Cap
    smin_oringinal = np.min(SOC)
    smax_oringinal = np.max(SOC)
    if smin_oringinal<0:
        SINI = np.ceil(-smin_oringinal/0.05)*0.05
    elif smax_oringinal>1:
        SINI = np.ceil(smax_oringinal-1/0.05)*0.05
    else:
        SINI = 0
    return hourly_price, hourly_load, hourly_battery, Pdmax, Pcmax, Smax, Smin, Cap, eta, SINI, hourly_gas,curtailment


#%% M1: the demand function bidding -- the proposed method
def biddingNEW(N_t, N_day, eta,CAP,Smax,Smin,Pdmax,Pcmax,price,SINI):
    # initialize the ESSs
    Cap = CAP
    SOC_ini = SINI
    # print(SINI)
    Res0 = np.zeros((N_day)) 
    # get prices
    pri = price[1:N_t]
    pri0 = np.linspace(min(price), max(price), N_day)
    
    # initialize variables for the stair recognization
    iup= np.zeros((N_day)) 
    ilow= np.zeros((N_day))
    numncd0= np.zeros((3,N_day)) 
    Tb= np.zeros((N_day))
    Elast = np.zeros((N_day))
    ResE= np.zeros((N_t,N_day)) 
    Res= np.zeros((N_t,N_day)) 
    ResPd= np.zeros((N_t,N_day)) 
    ResPc= np.zeros((N_t,N_day)) 
    
    # traverse the price at t=0 from the minimum to the maximum
    for K in range(N_day):        
        # build the model 
        model = gp.Model()    
        # variable definition
        SOC = model.addVars( N_t+1, vtype=GRB.CONTINUOUS, lb=Smin, ub=Smax, name='SOC')
        Pc = model.addVars( N_t, vtype=GRB.CONTINUOUS, lb=0, ub=Pcmax, name='Pc')
        Pd = model.addVars( N_t, vtype=GRB.CONTINUOUS, lb=0, ub=Pdmax, name='Pd')
        # Constraints
        # SOC initialization
        model.addConstr((SOC[0] == SOC_ini), 'SOC_1')
        # SOC update
        model.addConstrs((Cap*SOC[t] == Cap*SOC[t-1] 
                                            - (Pd[t-1]/eta - eta*Pc[t-1])*(24/N_t) for t in range(1, N_t+1)), 'SOC')
        # SOC bounds
        model.addConstrs(((SOC[t] >= Smin) for t in range(1, N_t+1)), 'SOC_lower_bound')
        model.addConstrs(((SOC[t] <= Smax) for t in range(1, N_t+1)), 'SOC_upper_bound')

        # objective function
        model.setObjective(((pri0[K]*((Pd[0]-Pc[0]))*(24/N_t) +(sum(pri[t-1]*((Pd[t]-Pc[t])*(24/N_t)) 
                                                        for t in range(1,N_t)))) ), GRB.MAXIMIZE)   
        # solve the model
        model.setParam('OutputFlag', 0)
        model.optimize()
        # get power at time 0
        Res0[K] = Pd[0].X-Pc[0].X
        if Pd[0].X*Pc[0].X !=0:
            if Pd[0].X>Pc[0].X*eta**2:
                Res0[K] = Pd[0].X-Pc[0].X*eta**2
            else:
                Res0[K] = Pd[0].X/(eta**2)-Pc[0].X
                
        # stair recognization        
        Elast[K] = Cap*SOC[N_t-1].X - (Pd[N_t-1].X/eta - eta*Pc[N_t-1].X)
        for c in range(N_t):
            ResPd[c,K] = Pd[c].X
            ResPc[c,K] = -Pc[c].X
            Res[c,K] = Pd[c].X-Pc[c].X
        for c in range(N_t-1):
            ResE[c,K] = SOC[c+1].X
        uplabel = len(np.argwhere(ResE[:,K].T == Smax))
        if uplabel == 0:
            iup[K] = 24
        else:
            idx = np.where(ResE[:, K].T == Smax)[0][0]  
            iup[K] = idx + 1  
        lowlabel = len(np.argwhere(ResE[:,K].T == Smin))
        if lowlabel == 0:
            ilow[K] = 24
        else:
            idx = np.where(ResE[:, K].T == Smin)[0][0]  
            ilow[K] = idx + 1
        Tb[K]=min(iup[K],ilow[K])
        aa=int(Tb[K])
        numncd0[0,K] = np.sum(ResPc[:aa,K]==-Pcmax)
        numncd0[1,K] = np.sum(ResPd[:aa,K]==Pdmax)
        numncd0[2, K] = np.sum((ResPc[:aa, K] + ResPd[:aa, K] >= -1e-3) & 
                       (ResPc[:aa, K] + ResPd[:aa, K] <= 1e-3))
    Res0_rounded = np.round(Res0, 4)   
    
    # Build the staircase bidding
    i = 0
    pricesstair = [min(pri0)]
    while i < len(Res0_rounded):
        start_power = Res0_rounded[i]
        max_price = pri0[i]
        while i + 1 < len(Res0_rounded) and Res0_rounded[i + 1] == start_power:
            i += 1
            max_price = max(max_price, pri0[i])  
        pricesstair.append(max_price)
        i += 1  
    powerstair = np.unique(Res0_rounded)
    NstairESS = len(powerstair)
    maxdis = max(Res0)
    maxch = min(Res0)
    maxdis_r = max(Res0_rounded)
    maxch_r = min(Res0_rounded)
    if maxdis_r > maxdis:
        powerstair[-1] = maxdis
    if maxch_r < maxch:
        powerstair[0] = maxch
    maxdis = max(powerstair)
    maxch = min(powerstair)
    if NstairESS ==1:
        if powerstair[0] >0:
            powerstair = np.insert(powerstair,-1,maxdis-1e-6)
        else:
            powerstair = np.insert(powerstair,1,maxch+1e-6)
    else:
        NstairESS = len(powerstair)+1
        powerstair = np.insert(powerstair,1,maxch+1e-6)
        powerstair = np.insert(powerstair,-1,maxdis-1e-6)
    points = np.zeros((NstairESS,3))
    NstairESS = len(powerstair)-1
    for i in range(NstairESS):
        points[i,0] = powerstair[i]
        points[i,1] = powerstair[i+1]
        points[i,2] = pricesstair[i]
    return [numncd0,points, maxdis, maxch]

#%% calculate the profit of the battery that bid by M1
def calculate_profit(N_t, Nday, N_day_ess, eta,CAP,Smax,Smin,Pdmax,Pcmax,price, Sinitial,meanstd):
    # Initialize the battery
    SINI = Sinitial
    ESSOC_status = np.zeros((N_t*Nday))
    profit = np.zeros((N_t*Nday))
    P_cleared = np.zeros((N_t*Nday))
    ncd = np.zeros((N_t*Nday,3,N_day_ess))
    
    # for each time interval bid a demand-supply function
    for t in range(N_t*Nday):
        print(t)
        price96 = price[t:t+N_t]
        noise = np.zeros((N_t))
        price96std = np.zeros((N_t))
        # generate prices with prediction error
        for tt in range(N_t):
            sigma = tt *0.001 + (meanstd / 100)
            noise[tt] = random.uniform(-sigma,+sigma)
            price96std[tt] = np.array(price96[tt]*(1 + noise[tt]))
            if price96std[tt]<np.min(price):
                price96std[tt] =np.min(price)
            if price96std[tt]>np.max(price):
                price96std[tt]=np.max(price)
        
        # bid a demand-supply function
        [numncd0,y, maxdis, maxch] = biddingNEW(N_t, N_day_ess, eta,CAP,Smax,Smin,Pdmax,Pcmax,price96std,SINI)
        
        # clearing and profit calculation
        x = price96[0]
        for i in range(len(y) - 1):
            y_low = y[i, 2]
            y_high = y[i + 1, 2]
            if y_low <= x <= y_high or y_high <= x <= y_low: 
                P_cleared[t] = y[i, 1]
        if P_cleared[t] >=0:
            SINI = SINI - P_cleared[t]/eta/CAP*(24/N_t)
        else:
            SINI = SINI - P_cleared[t]*eta/CAP*(24/N_t)
        ESSOC_status[t] = SINI
        profit[t] = x*P_cleared[t]*(24/N_t)
        ncd[t,:,:] = numncd0
    aveprice = np.mean(price[0:N_t*Nday])
    ESSend = ESSOC_status[-1]*CAP
    end_value = ESSend*aveprice
    return profit, ESSOC_status,P_cleared,end_value,ncd

#%% calculate the profit of the battery that bid by M2 (the actual data)
def calculate_profit_actual(eta,CAP,price, N_t,Nday, netoutput,Sinitial):
    # Initialize the battery
    SINI = Sinitial
    ESSOC_status = np.zeros((N_t*Nday))
    profit = np.zeros((N_t*Nday))
    # profit calculation
    for t in range(N_t*Nday):
        x = price[t]
        profit[t] = x*netoutput[t]*(24/N_t)
        if netoutput[t] >=0:
            SINI = SINI - netoutput[t]/eta/CAP*(24/N_t)
        else:
            SINI = SINI - netoutput[t]*eta/CAP*(24/N_t)
        ESSOC_status[t] = SINI  
    aveprice = np.mean(price[0:N_t*Nday])
    ESSend = ESSOC_status[-1]*CAP
    end_value = ESSend*aveprice
    
    return profit, ESSOC_status,netoutput,end_value

#%% Generation cost and carbon emission

# calculate the generation cost and the carbon emission
def calculate_cost_and_carbon(net_load_actual,curtailment, PESS_method,PESS_actual,N_t,Nday):
    # the function of the generation cost to net demand (fitted by data fron NYISO)
    a=0.067
    b=-1801.441
    c=12195662.599
    # calculate the net demand in three cases: column1--M1, column2--M2, column3--M2_without_battery_participation
    Pnet = np.zeros((N_t*Nday,3))
    Pnet[:,1] = net_load_actual
    Pnet[:,2] = (net_load_actual+PESS_actual)
    absorbed = np.zeros((N_t*Nday))
    
    # consider the effect of the curtailment
    diff = PESS_method-PESS_actual
    for t in range(N_t*Nday): 
        if diff[t]<0:
            absorbed[t] = min(-diff[t],curtailment[t])
        else:
            if net_load_actual[t]<0:
                absorbed[t] = -diff[t]
            else:
                absorbed[t] = -max(diff[t]-net_load_actual[t],0)
        
    net_load_new = net_load_actual - absorbed + (PESS_actual - PESS_method)
    Pnet[:,0] = net_load_new
    
    # calculated the generation cost
    Cost = np.zeros((N_t*Nday,3))
    for t in range(N_t*Nday): 
        for j in range(3):
            Cost[t,j] = a*(Pnet[t,j])**2 + b*(Pnet[t,j])**1 + c 
    return Pnet, Cost, absorbed, diff

def calculate_main(meanstd,price, load, battery, curtailment, Pdmax, Pcmax, Smax, Smin, Cap, eta, SINI):
    # calculate the generation cost, the battery profit and the carbon emission
    profit, ESSOC_status,P_cleared, end_value,ncd = calculate_profit(N_t, N_day, Nday_ess, eta,Cap,Smax,Smin,Pdmax,Pcmax,price, SINI,meanstd)
    profit_actual, ESSOC_status_actual,P_cleared_actual, end_value_actual = calculate_profit_actual(eta,Cap,price, N_t,N_day, battery,SINI)
    total_profit = sum(profit)+end_value
    total_profit_actual = sum(profit_actual)+end_value_actual
    Pnet, Cost, absorbed, diff = calculate_cost_and_carbon(load,curtailment,P_cleared,P_cleared_actual,N_t,N_day)
    
    # summarize the results
    total_cost = sum(Cost[:,0])
    total_cost_actual = sum(Cost[:,1])
    total_cost_withoutESS = sum(Cost[:,2])
    res_date = {'profit': profit, 'profit_actual': profit_actual, 'P_ESS': P_cleared,'P_ESS_actual': P_cleared_actual,
    'P_net_load':Pnet[:,0],'P_net_load_actual':Pnet[:,1], 'P_renewable_absorbed':absorbed,
    'cost': Cost[:,0], 'cost_actual': Cost[:,1], 'cost_withoutESS': Cost[:,2],
    'total_profit': total_profit, 'total_profit_actual': total_profit_actual, 
    'total_cost': total_cost, 'total_cost_actual': total_cost_actual, 'total_cost_withoutESS': total_cost_withoutESS,
    'total_absorb': sum(absorbed)}

    df_results = pd.DataFrame(res_date)
    df_results.to_csv('results_program/'+month+'2024_eta95%_std'+str(int(meanstd))+'.csv', index=False)
    return [total_profit,total_profit_actual, total_cost,total_cost_actual,total_cost_withoutESS,Cost,absorbed,P_cleared,Pnet,ncd]


#%% main

price_glo, load_glo, battery_glo, Pdmax_glo, Pcmax_glo, Smax_glo, Smin_glo, Cap_glo, eta_glo, SINI_glo, gas_glo,curtailment_glo = readdata(N_t*12,N_t,N_day)
Nday_ess = 100 # the number of traversed prices
meanstd = 2 # prediction error variable
# matrice initialzation
Capacity = np.zeros((1,N_day))
Pdmax = np.zeros((1,N_day))
Pcmax = np.zeros((1,N_day))
SINI = np.zeros((1,N_day))
Profits = np.zeros((1,2))
Cost = np.zeros((1,3))

# get results and save
result = calculate_main(meanstd, price_glo, load_glo, battery_glo, curtailment_glo,Pdmax_glo, Pcmax_glo, Smax_glo, Smin_glo, Cap_glo, eta_glo, SINI_glo)
[Profits[0, 0], Profits[0, 1], Cost[0,0], Cost[0,1],Cost[0,2],costdetails,absorbed, P_cleared,Pnet,ncd] = result
np.save('results of 202406 to 202505/with_prediction_error/ncd_'+month+'2024.npy',ncd)
np.save('results of 202406 to 202505/with_prediction_error/Pcleared_'+month+'2024.npy',P_cleared)
totalabsorbed = sum(absorbed)
print('total absorbed curtailment = '+str(totalabsorbed))
df = pd.read_csv("load/load"+month+".csv")
totalload = pd.to_numeric(df["load.load"], errors="coerce").sum()
rate_load = totalabsorbed/totalload

# calculate the carbon reduction
carbon_reduce = (8500/1000)*0.053165*totalabsorbed #(8500/1000)*0.053165 is collected from the CAISO carbon emission document
totalmonth_carbon = [2655335.59,4669362.14,4308940.88,4077587.38,4189340.87,3620906.08,4241056.99,3655694.77,2541019.37,2337717.50,1880394.26,1962830.77]
rate_carbon = carbon_reduce/totalmonth_carbon[carbonmonth]
resultsfinal = np.array([Profits[0, 0],Profits[0, 1], Cost[0,0], Cost[0,1],totalabsorbed,totalload,rate_load,carbon_reduce,rate_carbon])

