#needs the python environment venv in this directory
#activate: source venv/bin/activate
#deactivate: deactivate


import pennylane as qml


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
print('mnist-pca')
[ds] = qml.data.load("other", name="mnist-pca")
feature_dim=10
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),f'mnist-pca_train_{feature_dim}d')
writeCSV(test_i,test_l,len(test_i),f'mnist-pca_test_{feature_dim}d')
'''########################################################################
print('hyperplanes')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=4
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test')
########################################################################'''
print('bars-and-stripes')
[ds] = qml.data.load("other", name="bars-and-stripes")
feature_dim=8
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),f'bars-and-stripes_{feature_dim}d')
writeCSV(test_i,test_l,len(test_i),f'bars-and-stripes_{feature_dim}d')
'''########################################################################
print('hidden-manifold')
[ds] = qml.data.load("other", name="hidden-manifold")
feature_dim=4
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hidden-manifold')
writeCSV(test_i,test_l,len(test_i),'hidden-manifold')
########################################################################'''
print('two-curves')
[ds] = qml.data.load("other", name="two-curves")
feature_dim=10
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),f'two-curves_{feature_dim}d')
writeCSV(test_i,test_l,len(test_i),f'two-curves_{feature_dim}d')
########################################################################
print('linearly-separable')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=20
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),f'linearly-separable_{feature_dim}d')
writeCSV(test_i,test_l,len(test_i),f'linearly-separable_{feature_dim}d')
########################################################################



'''########################################################################
print('hyperplanes 6')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=6
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_6')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_6')
########################################################################
print('hyperplanes 8')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=8
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_8')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_8')
########################################################################
print('hyperplanes 10')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=10
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_10')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_10')
########################################################################
print('hyperplanes 5')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=5
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_5')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_5')
########################################################################
print('hyperplanes 7')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=7
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_7')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_7')
########################################################################
print('hyperplanes 9')
[ds] = qml.data.load("other", name="hyperplanes")
feature_dim=9
train_i=ds.diff_train[f'{feature_dim}']['inputs']
train_l=ds.diff_train[f'{feature_dim}']['labels']
test_i=ds.diff_test[f'{feature_dim}']['inputs']
test_l=ds.diff_test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.diff_train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'hyperplanes_train_9')
writeCSV(test_i,test_l,len(test_i),'hyperplanes_test_9')



print('linearly-separable 5')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=5
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable_5')
writeCSV(test_i,test_l,len(test_i),'linearly-separable_5')
########################################################################
print('linearly-separable 6')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=6
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable_6')
writeCSV(test_i,test_l,len(test_i),'linearly-separable_6')
########################################################################
print('linearly-separable 7')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=7
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable_7')
writeCSV(test_i,test_l,len(test_i),'linearly-separable_7')
########################################################################
print('linearly-separable 8')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=8
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable_8')
writeCSV(test_i,test_l,len(test_i),'linearly-separable_8')
########################################################################
print('linearly-separable 9')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=9
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable 9')
writeCSV(test_i,test_l,len(test_i),'linearly-separable 9')
########################################################################
print('linearly-separable 10')
[ds] = qml.data.load("other", name="linearly-separable")
feature_dim=10
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
print('write data')
writeCSV(train_i,train_l,len(train_i),'linearly-separable 10')
writeCSV(test_i,test_l,len(test_i),'linearly-separable 10')
########################################################################

























print('bars-and-stripes')
[ds] = qml.data.load("other", name="bars-and-stripes")
feature_dim=4
train_i=ds.train[f'{feature_dim}']['inputs']
train_l=ds.train[f'{feature_dim}']['labels']
test_i=ds.test[f'{feature_dim}']['inputs']
test_l=ds.test[f'{feature_dim}']['labels']

for i in range(len(train_l)):
	if train_l[i] == -1: train_l[i] = 0
for i in range(len(test_l)):
	if test_l[i] == -1: test_l[i] = 0
print(len(ds.train[f'{feature_dim}']['inputs'][0]))
#print('write data')
#writeCSV(train_i,train_l,len(train_i),'bars-and-stripes_2D')
#writeCSV(test_i,test_l,len(test_i),'bars-and-stripes_2D')
import matplotlib.pyplot as plt
import numpy as np
#ds.train['8']['inputs'] # vector representations of 4x4 pixel images
#ds.train['8']['labels'] # labels for the above images
for i in range(50):
	plt.imshow(np.reshape(ds.train['4']['inputs'][i], (4,4)))
	print(ds.train['4']['inputs'][i])
	plt.show()'''
