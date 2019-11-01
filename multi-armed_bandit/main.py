import matplotlib.pyplot as plt
import numpy as np

#####Loading and preprocessing data---------------------------------------------
txt = open("data.txt")
text = []
i = 0
for word in txt:
    text.append((word.split(':')[1].split(';')) + word.split(':')[2].split(';'))
    text[i][14] = text[i][14][:-2]
    i+=1
data = np.array(text).astype(np.float)
n = data.shape[0]
d = 5 #context dimension
actions = 10 #number of possible actions
context,rewards = np.split(data,[d],axis=1)


#####Bandit algorithms----------------------------------------------------------
#1- Random choice algorithm
random = [rewards[k][x] for k,x in zip(range(n),np.random.randint(10,size=n))]


#2- Staticbest algorithm
best = np.argmax(rewards.mean(axis=0))
staticbest = [x[best] for x in rewards]


#3- Optimal algorithm
optimal = [x.max() for x in rewards]


#4- UCB algorithm
#Auxiliary function
def update_UCB(k,exp,upperbound,actions,choice):
    actions[choice]+=1
    exp[choice] = ( exp[choice]*(actions[choice]-1) + rewards[k][choice] ) /actions[choice]
    upperbound = exp +  np.sqrt(2*np.log(k)/actions)
    return exp,upperbound,actions
#initialization
UCB = []
UCB_exp = np.array([rewards[k][k] for k in range(actions)])
UCB_upperbound = UCB_exp
UCB_actions =np.ones(actions)
#running algorithm
for k in range(actions,n):
    choice = UCB_upperbound.argmax()
    UCB.append(rewards[k][choice])
    UCB_exp,UCB_upperbound,UCB_actions = update_UCB(k,UCB_exp,UCB_upperbound,UCB_actions,choice)


#5- UCBV algorithm
#Auxiliary function
def update_UCBV(k,exp,var,upperbound,actions,choice):
    actions[choice]+=1
    exp[choice] = ( exp[choice]*(actions[choice]-1) + rewards[k][choice] ) /actions[choice]
    var[choice] = ( var[choice]*(actions[choice]-1) + (rewards[k][choice]-exp[choice])**2 ) /actions[choice]
    upperbound = exp +  np.sqrt(2 * np.log(k) * var / actions) + np.log(k) / ( actions * 2)
    return exp,var,upperbound,actions

#initialization
UCBV = []
UCBV_exp = np.array([rewards[k][k] for k in range(actions)])
UCBV_var = np.zeros(10)
UCBV_upperbound = UCBV_exp
UCBV_actions =np.ones(actions)

#running algorithm
for k in range(actions,n):
    choice = UCBV_upperbound.argmax()
    UCBV.append(rewards[k][choice])
    UCBV_exp,UCBV_var,UCBV_upperbound,UCBV_actions = update_UCBV(k,UCBV_exp,UCBV_var,UCBV_upperbound,UCBV_actions,choice)


#6- LinUCB algorithm
#Auxiliary functions
def compute_teta(_A,_b):
    return np.dot(np.linalg.inv(_A),_b)

def compute_estimator(_teta,_x,_alpha,_A):
    return np.dot(_x.T,_teta) + _alpha * np.sqrt(np.dot(_x.T,compute_teta(_A,_x)))

def update_LinUCB(_A,_b,_LinUCB_est,_LinUCB,r,x):
    arm = _LinUCB_est.argmax()
    _A[arm] = _A[arm] + np.dot(x.T,x)
    _b[arm] = _b[arm] + r[arm] * x
    _LinUCB.append(r[arm])
    return _A,_b,_LinUCB

def compute_LinUCB(alpha=1):
    #initialization
    LinUCB = []
    A = [np.identity(d)] * actions
    b = [np.zeros(d)] * actions

    #running algorithm
    for k in range(n):
        LinUCB_est = np.array([compute_estimator(compute_teta(A[i],b[i]),context[k],alpha,A[i]) for i in range(actions)])
        A,b,LinUCB = update_LinUCB(A,b,LinUCB_est,LinUCB,rewards[k],context[k])
    return LinUCB

#Gridsearch to take the best alpha
alpha = [i/100 for i in range(10,21)]
LinUCBs = [compute_LinUCB(alpha=a) for a in alpha]
LinUCB = max(LinUCBs,key= lambda x: sum(x))


#####Visualization--------------------------------------------------------------
print("Cumulative rewards per policy :")
print('- Random : ',round(sum(random),2))
print('- Staticbest : ',round(sum(staticbest),2))
print('- Optimal : ',round(sum(optimal),2))
print('- UCB-V : ',round(sum(UCBV),2))
print('- LinUCB : ',round(sum(LinUCB),2))

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.plot(alpha,[sum(LinUCB) for LinUCB in LinUCBs])
plt.title("LinUCB : evolution of final gain w.r.t alpha")
plt.xlabel("alpha")
plt.ylabel("Gain")
plt.show()

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.plot(np.cumsum(random),label = "Random")
plt.plot(np.cumsum(staticbest),label = "Static_Best")
plt.plot(np.cumsum(optimal),label = "Optimal")
plt.plot(np.cumsum(UCB),label = "UCB")
plt.plot(np.cumsum(UCBV),label = "UCB-V")
plt.plot(np.cumsum(LinUCB),label = "LinUCB")
plt.title("Evolution of cumulative rewards per policy")
plt.xlabel("Iterations")
plt.ylabel("Gain")
plt.xlim((0,n))
plt.ylim((0,1600))
plt.legend()
plt.show()
