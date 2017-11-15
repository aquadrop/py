import numpy as np

arr = [[['我', '2'], ['3', '4']], [['5', '6'], ['7', '8']]]
arr = np.asarray(arr)
a = arr.ravel()

for i, c in enumerate(a):
    print(str(c))
    if str(c) == '我':
        print('...')
print(a)
print(arr)
