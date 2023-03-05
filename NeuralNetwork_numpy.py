
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this


X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)


X_train=X_train/255
X_test=X_test.T/255

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_diff(x):
    return sigmoid(x)*(1-sigmoid(x))
def ReLU(x):
    return np.array([np.maximum(0,i) for i in x])
def ReLU_diff(x):
    #return np.array([np.maximum(0,i>0) for i in x])
    return x>0
def tanh(x):
    return np.tanh(x)
def tanh_diff(x):
    t=np.tanh(x)
    return 1-t*t
diff={
    ReLU:ReLU_diff,
    tanh:tanh_diff,
    sigmoid:sigmoid_diff
}
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

class NeuralNetwork:
    def __init__(self,n,ldim,X,Y,activations,lr):
        self.X=X
        self.Y=Y
        self.n=n
        self.w=[]
        self.b=[]
        self.inp_dim,self.nrows=X.shape
        self.activations=activations
        
        self.lr=lr
        
        for i in range(n):
            self.w.append(np.random.rand(ldim[i],ldim[i-1] if i!=0 else self.inp_dim)-0.5)
            self.b.append(np.random.rand(ldim[i],1)-0.5)
        return
    def forward_propagation(self,ba,batch_size):
        self.Z=[]
        self.A=[]
        self.Z.append(self.w[0]@self.X[:,ba*batch_size:(ba+1)*batch_size] + self.b[0])
        act=self.activations[0]
        self.A.append(act(self.Z[0]))
        for i in range(1,self.n):
            self.Z.append(self.w[i]@self.A[i-1] + self.b[i])
            act=self.activations[i]
            self.A.append(act(self.Z[i]))
        #self.FinalA=self.A[self.n -1]
        return
    def back_propagation(self,ba,batch_size):
        one_hot_Y = one_hot(self.Y[ba*batch_size:(ba+1)*batch_size])
        self.dw=[0.]*self.n
        self.db=[0.]*self.n
        self.dZ=[0.]*self.n
        n=self.n
        self.dZ[n-1]=self.A[n-1]-one_hot_Y
        self.dw[n-1]=(1/self.nrows)*self.dZ[n-1]@self.A[n-2].T
        self.db[n-1]=(1/self.nrows)*np.sum(self.dZ[n-1])
        
        for i in range(self.n-1,0,-1):
            act_diff=diff[self.activations[i-1]]
            self.dZ[i-1]=self.w[i].T@self.dZ[i] * act_diff(self.Z[i-1])#W2.T.dot(dZ2) * ReLU_deriv(Z1)
            if i!=1:
                self.dw[i-1]=(1/self.nrows)*self.dZ[i-1]@self.A[i-2].T
            else:
                self.dw[i-1]=(1/self.nrows)*self.dZ[i-1]@self.X[:,ba*batch_size:(ba+1)*batch_size].T
            self.db[i-1]=(1/self.nrows)*np.sum(self.dZ[i-1])
        
        return
    def update_parameters(self):
        for i in range(self.n):
            self.w[i]=self.w[i]-self.lr*(self.dw[i])
            self.b[i]=self.b[i]-self.lr*(self.db[i])
        return
    def fit(self,epochs=100,batches=100):
        #batches=100
        batch_size=self.nrows//batches
        for i in tqdm(range(1,epochs+1)):
            for ba in range(batches):
                self.forward_propagation(ba,batch_size)
                self.back_propagation(ba,batch_size)
                self.update_parameters()
        print("Done fitting.")
    def predict(self,x):
        Z=[]
        A=[]
        Z.append(self.w[0]@x + self.b[0])
        act=self.activations[0]
        A.append(act(Z[0]))
        for i in range(1,self.n):
            Z.append(self.w[i]@A[i-1] + self.b[i])
            act=self.activations[i]
            A.append(act(Z[i]))
        return np.argmax(A[self.n -1],0)
    def accuracy(self,y_pred,y):
        return np.sum(y==y_pred)/y.size

#train_x,test_x,train_y,test_y=model_selection.train_test_split(X_train,y_train,test_size=0.2)
train_x=X_train.T
train_y=y_train.T
# test_x=test_x.T
# test_y=test_y.T

n=3
nn=NeuralNetwork(n,[50,20,10],train_x,train_y,[ReLU, ReLU, softmax],0.6)
nn.fit(1500)

train_x.shape

# y_pred=nn.predict(test_x)
# nn.accuracy(y_pred,test_y)

y_pred=nn.predict(X_test)
submissions_df['label']=pd.DataFrame(y_pred)


submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))


