import numpy as np
N=10
band_train=np.zeros(10*N)
dir='../output_std_nonoise_'
for i in range(10):
	grad = np.load(dir+str(i)+'/train_grads.npy')
	print(grad.shape)
	grad_abs = np.absolute(grad)
	avg_grad = np.mean(grad_abs, axis=(0))
	print(avg_grad.shape)
	idxs = (-np.sum(avg_grad,axis=0)).argsort()[:N]
	print('Most important train indexes: ',idxs, ' Model: ', i)
	band_train[i*N:(i+1)*N]=idxs

def make_hist(idxs):
    values, counts = np.unique(idxs, return_counts=True)
    print(np.stack((values, counts), axis=-1))
    inds = counts.argsort()
    print(np.stack((values[inds], counts[inds]), axis=-1))
make_hist(band_train)

