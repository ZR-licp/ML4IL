import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt   
from sklearn.metrics import r2_score    
from sklearn.model_selection import train_test_split    
  
data = pd.read_csv('Opt_RFE_descriptor.csv')  
  
X = data[['ABC', 'LogEE_A', 'VR1_A', 'VR2_A', 'VR3_A', 'nN', 'nS']]
y = data['diffusion']    
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)    
  
# 定义复杂的计算公式为lambda函数  
def viscosity_formula(row):
    ABC = row['ABC']     
    LogEE_A = row['LogEE_A']  
    VR1_A = row['VR1_A']
    VR2_A = row['VR2_A']
    VR3_A = row['VR3_A']
    nN = row['nN']  
    nS = row['nS']  
       
    return ((10.784009 - np.log(VR1_A + (93.8193 / ((LogEE_A - (nN + VR2_A)) + LogEE_A)))) - (VR2_A * 0.30362245)) - nS
  
y_pred_train = X_train.apply(viscosity_formula, axis=1)   
y_pred_test = X_test.apply(viscosity_formula, axis=1)  
  
r2_train = r2_score(y_train, y_pred_train)    
r2_test = r2_score(y_test, y_pred_test)  

plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 42, 
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 42,
}
plt.figure(figsize=(12.0,11.5)) 

y_train = y_train.ravel()
y_pred_train = y_pred_train.ravel()
y_test = y_test.ravel()
y_pred_test = y_pred_test.ravel()

plt.plot(y_train,y_pred_train,color='#C0C0C0',marker='o',linestyle='', markersize=20, markerfacecolor='#80C149',alpha=1) #8
plt.plot(y_test,y_pred_test,color='#C0C0C0',marker='o',linestyle='', markersize=20, markerfacecolor='b',alpha=0.5)

plt.text(0.05, 0.88, f'R² Train: {r2_train:.2f}', transform=plt.gca().transAxes, fontsize=44) # 26    
plt.text(0.05, 0.75, f'R² Test: {r2_test:.2f}', transform=plt.gca().transAxes, fontsize=44)  

plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=38, frameon=True) # 22
title='MD calculated viscosity [mpa.s]' 
title1='SR Predicted viscosity [mpa.s]'

plt.xlabel(title,font1)
plt.ylabel(title1,font1)

plt.xlim((0, 10))
plt.ylim((0, 10))
plt.plot([0, 10],[0, 10], color='k', linewidth=5.0, linestyle='--')
my_x_ticks = np.arange(0, 10, 1.0)
my_y_ticks = np.arange(0, 10, 1.0)  

plt.xticks(my_x_ticks,size=38)
plt.yticks(my_y_ticks,size=38)
plt.tick_params(width=4.0, length=10.0) 
bwith = 4.0 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig('./viscosity_rs_accuracy.png')
