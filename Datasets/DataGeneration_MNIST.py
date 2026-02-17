#needs the python environment venv in this directory
#activate: source venv/bin/activate
#deactivate: deactivate


#import pennylane as qml
import numpy as np

# write a test/train/val data set to a file
def writeCSV(x,y,num_samples,typ):
	out_X=open(f'X{typ}_{num_samples}.csv','w')
	out_Y=open(f'Y{typ}_{num_samples}.csv','w')
	for i in range(len(x)):
		nFeature=len(x[i])
		for j in range(nFeature+1):
			if j<nFeature: out_X.write(f'{x[i][j]},')
			else: out_X.write(f'1\n')
		out_Y.write(f'{y[i]}\n')
	out_X.close()
	out_Y.close()

########################################################################
print('mnist')

'''Train_I = Np.Load("Mnist_Fixed/Mnist_0_1_7x7_N_1000.Npz")["X"]
Train_L = Np.Load("Mnist_Fixed/Mnist_0_1_7x7_N_1000.Npz")["Y"]

Test_I = Np.Load("Mnist_Fixed/Mnist_0_1_7x7_N_1000_Test.Npz")["X"]
Test_L = Np.Load("Mnist_Fixed/Mnist_0_1_7x7_N_1000_Test.Npz")["Y"]

Feature_Dim = '0_1_7x7_N'

For I In Range(Len(Train_L)):
	If Train_L[I] == -1: Train_L[I] = 0
For I In Range(Len(Test_L)):
	If Test_L[I] == -1: Test_L[I] = 0
Print('Write Data')
Writecsv(Train_I,Train_L,Len(Train_I),F'Mnist_Train_{Feature_Dim}')
Writecsv(Test_I,Test_L,Len(Test_I),F'Mnist_Test_{Feature_Dim}')'''

########################################################################

'''Train_I = np.load("MNIST_fixed/mnist_0_1_14x14_N_1000.npz")["X"]
Train_L = np.load("MNIST_fixed/mnist_0_1_14x14_N_1000.npz")["y"]

Test_I = np.load("MNIST_fixed/mnist_0_1_14x14_N_1000_TEST.npz")["X"]
Test_L = np.load("MNIST_fixed/mnist_0_1_14x14_N_1000_TEST.npz")["y"]

Feature_Dim = '0_1_14x14_N'

for I in range(len(Train_L)):
	if Train_L[I] == -1: Train_L[I] = 0
for I in range(len(Test_L)):
	if Test_L[I] == -1: Test_L[I] = 0
print('Write Data')
writeCSV(Train_I,Train_L,len(Train_I),F'mnist_train_{Feature_Dim}')
writeCSV(Test_I,Test_L,len(Test_I),F'mnist_test_{Feature_Dim}')

########################################################################

Train_I = np.load("MNIST_fixed/mnist_0_1_28x28_N_1000.npz")["X"]
Train_L = np.load("MNIST_fixed/mnist_0_1_28x28_N_1000.npz")["y"]

Test_I = np.load("MNIST_fixed/mnist_0_1_28x28_N_1000_TEST.npz")["X"]
Test_L = np.load("MNIST_fixed/mnist_0_1_28x28_N_1000_TEST.npz")["y"]

Feature_Dim = '0_1_28x28_N'

for I in range(len(Train_L)):
	if Train_L[I] == -1: Train_L[I] = 0
for I in range(len(Test_L)):
	if Test_L[I] == -1: Test_L[I] = 0
print('Write Data')
writeCSV(Train_I,Train_L,len(Train_I),F'mnist_train_{Feature_Dim}')
writeCSV(Test_I,Test_L,len(Test_I),F'mnist_test_{Feature_Dim}')
'''



####################################################################

'''train_i = np.load("MNIST_fixed/mnist_3_5_7x7_N_1000.npz")["X"]
train_l = np.load("MNIST_fixed/mnist_3_5_7x7_N_1000.npz")["y"]

test_i = np.load("MNIST_fixed/mnist_3_5_7x7_N_1000_TEST.npz")["X"]
test_l = np.load("MNIST_fixed/mnist_3_5_7x7_N_1000_TEST.npz")["y"]

feature_dim = '3_5_7x7_N'

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
	elif train_l[i] == 3: train_l[i] = 0
	elif train_l[i] == 5: train_l[i] = 1
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
	elif test_l[i] == 3: test_l[i] = 0
	elif test_l[i] == 5: test_l[i] = 1
print('write data')
writeCSV(train_i,train_l,len(train_i),f'mnist_train_{feature_dim}')
writeCSV(test_i,test_l,len(test_i),f'mnist_test_{feature_dim}')'''


feature_dim = '3_5_14x14_N'

train_i = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000.npz")["X"]
train_l = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000.npz")["y"]

test_i = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000_TEST.npz")["X"]
test_l = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000_TEST.npz")["y"]



for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
	elif train_l[i] == 3: train_l[i] = 0
	elif train_l[i] == 5: train_l[i] = 1
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
	elif test_l[i] == 3: test_l[i] = 0
	elif test_l[i] == 5: test_l[i] = 1
print('write data')
writeCSV(train_i,train_l,len(train_i),f'mnist_train_{feature_dim}')
writeCSV(test_i,test_l,len(test_i),f'mnist_test_{feature_dim}')

feature_dim = '3_5_28x28_N'

train_i = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000.npz")["X"]
train_l = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000.npz")["y"]

test_i = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000_TEST.npz")["X"]
test_l = np.load(f"MNIST_fixed/mnist_{feature_dim}_1000_TEST.npz")["y"]



for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
	elif train_l[i] == 3: train_l[i] = 0
	elif train_l[i] == 5: train_l[i] = 1
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
	elif test_l[i] == 3: test_l[i] = 0
	elif test_l[i] == 5: test_l[i] = 1
print('write data')
writeCSV(train_i,train_l,len(train_i),f'mnist_train_{feature_dim}')
writeCSV(test_i,test_l,len(test_i),f'mnist_test_{feature_dim}')
