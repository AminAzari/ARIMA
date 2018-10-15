#IHN


#Importing packages
import pickle
import shelve
import random
import warnings
import numpy as np
import pandas as pd 
import seaborn; seaborn.set()
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

#Reading data
data = pd.read_csv("datan.csv")
#print(data.head(10))
#print(data.tail(5))
data.dtypes
data.info()
data.describe()
data = data['Close']
 

ts = data
ts_logtrsfed = np.log(ts)
ts_diff_logtrans = ts_logtrsfed -ts_logtrsfed.shift(1)
ts_diff_logtrans.dropna(inplace=True)
ts_diff_logtrans = ts_diff_logtrans.fillna(0)


 
from statsmodels.tsa.arima_model import ARIMA





print('*Grid search for pq')

p_values = range(0,10,1)
d_values = range(0,3,1)  
q_values = range(0,10,1) 


def evaluate_arima_model(data_set, arima_order):
    model = ARIMA(data_set, order=arima_order)
    results_ARIMA = model.fit(disp=-1) 
    RSS_diff = results_ARIMA.fittedvalues-ts_diff_logtrans
    RSS = RSS_diff**2
    ar_cf, ma_cf = results_ARIMA.arparams, results_ARIMA.maparams
    return RSS, results_ARIMA.arparams, results_ARIMA.maparams

def evaluate_arima_model_par(data_set, arima_order):
    model = ARIMA(data_set, order=arima_order)
    results_ARIMA = model.fit(disp=-1) 
    ar_cf, ma_cf = results_ARIMA.arparams, results_ARIMA.maparams
    return ar_cf, ma_cf
def evaluate_models(dataset, p_values, q_values):
    best_score, best_cfg = float("inf"), None
    k=-1
    A=[]
    for p in p_values:
        for q in q_values:
           for d in d_values:
		k=k+1
        	order = (p,d,q) 
		if p==0 and q==0:
			A.insert(k,float('nan'))
		elif q> p:
			A.insert(k,float('nan'))
		else:
			try: 
		  		rss,ar_cf, ma_cf =evaluate_arima_model(dataset, order) 
				rss.dropna(inplace=True)			
				Rss=sum(rss/len(rss) )
				A.insert(k,Rss)
				f = open('data_ar(%d).txt' % k, 'wb')
				pickle.dump(ar_cf, f)
				f.close()
				f = open('data_ma(%d).txt' % k, 'wb')
				pickle.dump(ma_cf, f)
				f.close()

				#print(order,Rss)	
		  		if Rss < best_score:
		     			best_score, best_cfg = Rss, order 
		     		 
		  	except:
				A.insert(k,float('nan'))
				continue
		print(p,d,q)
    print('Best ARIMA%s RSS=%.3f' % (best_cfg, best_score)) 
    return A

warnings.filterwarnings('ignore') 
A=evaluate_models(ts_logtrsfed,p_values,q_values)
 

f = open('storeA.txt', 'wb')
pickle.dump(A, f)
f.close()




model = ARIMA(ts_logtrsfed, order=(9, 1, 4))  
results_ARIMA = model.fit(trend= 'nc', disp=-1)  
print('*************************************arima_mod')
plt.plot(ts_diff_logtrans)
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'p =9, q =4')
RSS =results_ARIMA.fittedvalues-ts_diff_logtrans
ExRSS =np.exp(results_ARIMA.fittedvalues)-np.exp(ts_diff_logtrans)
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc='best')
#plt.show()
print('*************************************arima_end')

#model summary
print(results_ARIMA.summary())

#model splot
plt.plot(ts_logtrsfed, label = 'log_tranfromed_data')
plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')
plt.title('ARIMA Model Residual plot')
plt.legend(loc = 'best')



results_ARIMA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_ARIMA.resid.describe())
#plt.show()

print(np.mean(results_ARIMA.resid))
print(np.sqrt(np.var(results_ARIMA.resid)))


ar_cf, ma_cf = results_ARIMA.arparams, results_ARIMA.maparams





#ar_cf=[ 0.29778829, -0.23707003,  0.2437071,   0.71296213,  0.04217428, -0.00150594,  -0.02483154,  0.03071463,-0.06401419]
#ma_cf = [-0.28235576,  0.25410623, -0.22378936, -0.7479611 ]

#print(ar_cf)
#print(ma_cf)
 
f = open('storeA.txt' , 'rb')
A = pickle.load(f)
f.close()

Min_ha=[]
ITmse=[]
for u in range(0,20):

	#print(A)
	Tmse=[]
	k=-1
	AN=[]
	for p in p_values:
	    for q in q_values:
		k=k+1
		if A[k] > -1 and np.maximum(p,q)<w:
			AN=np.insert(AN,k,A[k])
			#print('(p,d,q):(', p,1,q,'), k:',k,'RSS:',A[k])		
			f = open('data_ar(%d).txt' % k, 'rb')
			ar_cf = pickle.load(f)
			f.close()
		
			f = open('data_ma(%d).txt' % k, 'rb')
			ma_cf = pickle.load(f)
			f.close()
			#ar_cf, ma_cf =evaluate_arima_model_par(ts_logtrsfed, (p,1,q)) 
			#print(ma_cf)
			#print(ar_cf)		
	
			LrG=random.sample(range(1, 95), 50)
			LrG=[0.01*LrG[i] for i in range(len(LrG))]				
			#plt.show()
			#LrG=[0.1 ,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85,0.9]


			delt=0.04#np.sqrt(np.var(results_ARIMA.resid))
			Lmse=0
			size=len(ts_logtrsfed)
			wpq=w-np.maximum(p,q)
			 
			#print('******************************************************************')
			e_ha00= [delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt,delt]
			e_ha0= e_ha00[0:q]
			Lmse=[]
			for loc in LrG:
				tmse=[]
				for ptrS in range(0,wpq,1):
					#print('-Start---------------------------------')		
					#print(loc,ptrS)
					test =   (ts_logtrsfed[int(loc*size):int(loc*size)+w])
					ptrL=np.maximum(p,q)
					if p!=0 and q!=0 and  p >= q: 				
						N_MAcf=ma_cf#[ma_cf[0],ma_cf[1],ma_cf[2],ma_cf[3]]
						N_ARcf1=ar_cf[1:p]			
						N_ARcf1=np.insert(N_ARcf1,0,1)
						N_ARcf1=np.insert(N_ARcf1,p,0)
						N_ARcf2=ar_cf[0:p]
						N_ARcf2=np.insert(N_ARcf2,0,np.negative(ar_cf[0]))		
						N_ARcf=[N_ARcf1[i]-N_ARcf2[i] for i in range(len(N_ARcf1))]		


						y_haf=test[ptrS:ptrS+ptrL+1:1]
						y_ha=np.flip(y_haf)
						e_ha=e_ha0[0:q]
						yhat= np.dot(N_ARcf, y_ha)+ np.dot(N_MAcf, e_ha)
					if p!=0 and q!=0 and  q > p: 				
						N_MAcf=ma_cf#[ma_cf[0],ma_cf[1],ma_cf[2],ma_cf[3]]
						N_ARcf1=ar_cf[1:p]			
						N_ARcf1=np.insert(N_ARcf1,0,1)
						N_ARcf1=np.insert(N_ARcf1,p,0)
						N_ARcf2=ar_cf[0:p+1]
						N_ARcf2=np.insert(N_ARcf2,0,np.negative(ar_cf[0]))		
						N_ARcf=[N_ARcf1[i]-N_ARcf2[i] for i in range(len(N_ARcf1))]		


						y_haf=test[ptrS:ptrS+ptrL+1:1]
						y_ha=np.flip(y_haf)
						y_ha=y_ha[0:p:1]
						e_ha=e_ha0[0:q]

						yhat= np.dot(N_ARcf, y_ha)+ np.dot(N_MAcf, e_ha)

					elif p!=0 and q==0:
						N_ARcf1=ar_cf[1:p]			
						N_ARcf1=np.insert(N_ARcf1,0,1)
						N_ARcf1=np.insert(N_ARcf1,p,0)
						N_ARcf2=ar_cf[0:p]
						N_ARcf2=np.insert(N_ARcf2,0,np.negative(ar_cf[0]))		
						N_ARcf=[N_ARcf1[i]-N_ARcf2[i] for i in range(len(N_ARcf1))]		

	 
						y_haf=test[ptrS:ptrS+ptrL+1:1]
	 
						y_ha=np.flip(y_haf)
						e_ha=e_ha0[0:q]

						yhat= np.dot(N_ARcf, y_ha)
					elif p==0 and q!=0:
						N_MAcf=ma_cf#[ma_cf[0],ma_cf[1],ma_cf[2],ma_cf[3]]
						y_haf=test[ptrS:ptrS+ptrL+1:1]
						y_ha=np.flip(y_haf)
						y_ha=y_ha[0]
						e_ha=e_ha0[0:q]
						yhat= np.dot(1, y_ha)+ np.dot(N_MAcf, e_ha)
					#print(test)
					#print(y_ha)
					#print(int(loc*size) +  ptrS+ptrL+1)		
					print(  'moghayese log'  )
					print(  yhat  )
					print(  float(ts_logtrsfed[int(loc*size) +ptrS+ptrL+1]))
					print(  'moghayese exp'  )
					print( np.exp(yhat)  )
					print(  float(np.exp(ts_logtrsfed[int(loc*size) +ptrS+ptrL+1]))) 
					mse=(np.exp(yhat) - float(np.exp(ts_logtrsfed[int(loc*size) +ptrS+ptrL+1])) )**2
					e_ha0=np.insert(e_ha0,0, float(ts_logtrsfed[int(loc*size) +ptrS+ptrL+1]-yhat))
					#print(' --_EnD-----  -')
					tmse=np.insert(tmse,0,mse)
				Lmse=np.insert(Lmse,0,(np.exp(yhat) - float(np.exp(ts_logtrsfed[int(loc*size) +ptrS+ptrL+1])) )**2)
				#print(  't-MSE'  )
				#print(np.around(tmse,1))

			#print(  'L-MSE'  )
			#print(sum(Lmse)/len(Lmse))
			Tmse=np.insert(Tmse,k,sum(Lmse)/len(Lmse))
			#print(np.around(Lmse,1))
			#plt.show()
			plt.plot(Lmse)
			#plt.show()


		else:
		
			Tmse=np.insert(Tmse,k,float('nan'))
			AN=np.insert(AN,k,float('nan'))

	print(  'T-MSE'  )
	print(Tmse)

	print(np.nanmin(AN),np.nanargmin(AN),np.nanmin(Tmse),np.nanargmin(Tmse))
 
	ITmse.append(Tmse)
	Min_ha=np.insert(Min_ha,u,np.nanmin(Tmse))
	print(u,'res',np.nanmin(Tmse))
#print(ITmse)
MITmse=np.nanmean(ITmse,0)
print(np.nanmin(AN),np.nanargmin(AN),np.nanmin(MITmse),np.nanargmin(MITmse))
print(np.mean(Min_ha))








 


