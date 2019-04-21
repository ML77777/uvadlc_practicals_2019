import cifar10_utils

cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

print(cifar10)
x,y = cifar10['train'].next_batch(5)
print(x)
print(y)

x_test,y_test = cifar10.test.images, cifar10.test.labels