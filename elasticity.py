
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS


class els:

    #Define class attributes  
    #Quarterly Inflation data
    c= pd.read_excel('cpi.xlsx', sheet_name='Data1')
    #cpi 1998/9 - 2018/3  quarterly
    cpi = pd.DataFrame(c.iloc[209:-4,8].astype(float))  
    #Unemployment data
    u = pd.read_csv('data.csv',index_col='Time')
    #Unemployment rate 1998/9 - 2018/3 quarterly
    U=u['Unemployment rate']    
    #Monthly bill data
    babs= pd.read_csv('3mo_bills.csv',header=2,index_col=0)
    #3 month bank bills, 1998/9 - 2018/3 quarterly 
    bb= babs['Bank Accepted Bills/Negotiable Certificates of Deposit-3 months; monthly average'][359:-14].astype(float)
    #Quarterly Bills
    bills=np.array(bb[::3].astype(float)) 

    #Three class functions. 

    #Run function produces the r2, regression beta/elasticity, corresponding p-value and plots the output from the rolling regression.  
    #Run function takes y,x and window size for the rolling regression
    def run(y,x,window):
        Y = np.log(y)   
        X= np.log(x)   
        model = PandasRollingOLS(y=Y, x=X, window=window) 
        p= model.predicted    
        p_value=model.pvalue_beta             
        r2=model.rsq     
        fig = plt.figure(figsize=(25,10))               
        ax = plt.axes() 
        ax.xaxis.set_major_locator(plt.MaxNLocator(21)) 
        ax.plot(model.beta, lw=5,color= 'red')
        ax.plot(p_value,lw=1,color= 'magenta')
        plt.plot(r2, lw=1,color='blue') 
        ax.legend(['Elasticity','P values','R2'],fontsize=15)
        plt.show() 
        print('                Number of quarters where the elasticity was significant at 95pc CI = %.1f' % 
          (p_value < 0.05).sum())
         
    #Difference data if needed, data must be 1d for this function.
    #Difference_data function takes the raw data as input and returns the differenced variable with the first NAN element removed. 
    def  difference_data(self): 
        X_diff=pd.Series(np.diff(self,axis=0))
        return X_diff 

    #Lag data if needed
    #Shift_data takes as input the data to be shifted or lagged and also number of periods by which the data is lagged(integer).
    def shift_values(self,periods):
        X_shift= pd.Series(self).shift(periods) 
        #Return lagged data and clip top of series with NAN's
        return X_shift.dropna()




 
