import numpy as np

def createSample(size):
    x=np.zeros((size,9))
    y=np.zeros((size,2))#0 = Kreuz; 1=Kreis
    for i in range(0,size):
        random = np.random.rand()
        if random<0.5:
            x[i]=createPic("Kreuz")
            y[i][0]=1.0
        else:
            x[i]=createPic("DIAMAND")
            y[i][1]=1.0
    return (x,y)
def createPic(form):
    pic=np.zeros(9)
    if form == "Kreuz":
        for i in range(0,9,2):
            pic[i]= 0.75+np.random.rand()/4
        for i in range(1,9,2):
            pic[i]=np.random.rand()/4
    else:#DIAMAND
        for i in range(0,9,2):
            pic[i]= np.random.rand()/4
        for i in range(1,9,2):
            pic[i]=0.75+np.random.rand()/4
    return pic
def printPic(pic):
    for i in range(0,int(len(pic)/3)):
        print(" %f %f %f"% (pic[3*i],pic[3*i+1],pic[3*i+2]))
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
def ReLU(x):
    return max(0,x)
def softmax(x):
    mat = np.zeros((x.shape[0],x.shape[1]))
    for i in range(0,x.shape[0]):
        sumRow=0
        for j in range(0,x.shape[1]):
            sumRow+=np.exp(x[i][j])
        for j in range(0,x.shape[1]):
            mat[i][j]=np.exp(x[i][j])/sumRow
    return mat
def matMul(x,y):
    z = np.zeros((x.shape[0],y.shape[1]))
    if(x.shape[1]!=y.shape[0]):
        pass
    else:
        for i in range(0,x.shape[0]):
            for j in range(0,y.shape[1]):
                for k in range(0,x.shape[1]):
                    z[i][j]+=x[i][k]*y[k][j]
    return z
def gradients(x,y_,weights,biases):
    rows = weights.shape[0]#9
    cols = weights.shape[1]#2

    nabla_w = np.zeros((rows,cols))
    for i in range(0,cols):#0 bis 1
        for j in range(0,rows):#0 bis 8
            dC_dm_ij = 0
            for k in range(0,len(x)):
                deltay_i=(sigmoid(np.dot(x[k],W)+biases) - y_[k])[i]
                sigma_prime_i=sigmoid_prime(np.dot(x[k],weights)+biases)[i]
                dC_dm_ij+=deltay_i*sigma_prime_i*x[k][j]
            nabla_w[j][i]=dC_dm_ij

    nabla_b = np.zeros(cols)
    for i in range(0,cols):#0 bis 2
        dC_db_ij = 0
        for k in range(0,len(x)):
            deltay_i=(sigmoid(np.dot(x[k],W)+biases) - y_[k])[i]
            sigma_prime_i=sigmoid_prime(np.dot(x[k],weights)+biases)[i]
            dC_db_ij+=deltay_i*sigma_prime_i
        nabla_b[i]=dC_db_ij


    return (nabla_w,nabla_b)

x,y_ = createSample(50)

W = (2*np.random.rand(9,2)-1)/4
b = 2*np.random.rand(2)-1



epochs = 100
eta = 0.5

#Training-----------------------------------------------------
print("Training initiallized...")
for i in range(0,epochs):
    if i%10==0:
        eta=eta*0.8
    y=sigmoid(np.dot(x,W)+b)
    cost=np.sum(np.square(y-y_)/2)
    if i%10==0:
        print("Cost: %f" % cost)

    nw,nb=gradients(x,y_,W,b)
    W=W-eta*nw
    b=b-eta*nb
print("Training done")
print("-----------------------\n")
#Testing-----------------------------------------------------
x_test,y_test = createSample(5)
y=sigmoid(np.dot(x_test,W)+b)
for i in range(0,len(x_test)):
    printPic(x_test[i])
    if np.argmax(y[i]) == 0:
        print("Vorraussage: %s"%"KREUZ")
    else:
        print("Vorraussage: %s"%"DIAMAND")
    print("-----------------------")