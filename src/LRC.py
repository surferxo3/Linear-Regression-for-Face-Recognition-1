import theano
import theano.tensor as T
import numpy as np

# image class matrix
x = T.matrix('x')

# image which will be classified
y = T.vector('y')

# LRC
hat_matrix = T.dot(T.dot(x, T.nlinalg.pinv(T.dot(x.T, x))), x.T)
hat_y_i = T.dot(hat_matrix, y)
distance = ((y - hat_y_i) ** 2).sum() ** 0.5

# distance to specific class
distance_i = theano.function([x, y], distance, allow_input_downcast=True)
