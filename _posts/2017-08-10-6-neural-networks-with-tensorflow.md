---
layout: post
author: Matthias Nitsche
title: Neural Networks with Tensorflow
keywords: [tensorflow, neural networks, deep learning]
index: 8
img: brain.jpg
---

In this post I would like to explore the concepts of Tensorflow. As explained before Tensorflow is a numerical computation engine using data flow graphs developed and maintained by Google.

<blockquote>Tensorflow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.<cite>- <a href="https://www.tensorflow.org/" target="_blank">Tensorflow.org</a></cite></blockquote>

{% include image.html url="/images/brain.jpg" description="source: https://upload.wikimedia.org/wikipedia/commons/c/c3/Neuronal_activity_DARPA.jpg" %}

### Tensorflow Setup

Working with Tensorflow in Python we need to setup some dependencies first. In a post before we used Docker to setup a machine learning environment. To actually leverage Docker based notebook session with Tensorflow we would need some computational servers to run our models on. GPU clusters are costly be it in my backyard or in the cloud. Thats why we need to prepare Tensorflow for a local session. The requirements for Python are

{% highlight python %}
$ cat requirements.txt
ipykernel
jupyter
scipy
numpy
pandas
matplotlib
sklearn
glances
pillow
tensorflow
keras
{% endhighlight %}

Lets install it

{% highlight python %}
pip install -r requirements.txt
{% endhighlight %}

The current setup of Tensorflow is without any GPU support because it is discontinued since `1.12` on Mac. On a Linux box NVIDIA GPU support can be enabled by installing [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker){:target="_blank"}. Then one could run a Docker container in a Jupyter notebook listening on `localhost:8888`

{% highlight python %}
nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu
{% endhighlight %}

No matter what direction you choose, be it Tensorflow for CPU/GPU locally or in a Docker container you should be able to import tensorflow in a file like so

{% highlight python %}
import numpy as np
import tensorflow as tf
import sklearn
{% endhighlight %}

### Neural Networks

Feedforward neural networks are the most basic constructions. A constant amount of layers with a set amount of units per layer, several activation functions and good old backpropagation for feeding the error back with respect to a cost function. The training has three steps. 

{% include image.html url="/images/feedforward.png" %}

1. Consume data and feed it forward through the network applying non linear activation functions and creating output suitable to the question.
2. Calculate the error in the forward pass with respect to a cost function. 
3. Backpropagate that error by taking the derivatives with respect to the activation functions and the error of subsequent layers. On the way back update all the weight by that margin.

Additionally it is possible to add some fancy parameters that drastically reduce the error. The goal for a good classifier is to keep the generalization error low e.g. the error on how well the data performs to unseen examples of different data distributions and the training error low e.g. how well does the classifier classify the underlying data distribution. Both errors are in a constant battle if you fit too well in the training, the generalization is often not very good and vice versa. 

### Tensorflow

{% highlight python %}
import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
{% endhighlight %}

After setting up the imports, lets get the Fisher's Iris Data. The dataset contains 3 kinds of flowers with 4 features for each flower.

<table class="table table-md table-striped table-bordered">
<thead>
  <tr>
  <th>Species</th>
  <th>Sepal length</th>
  <th>Sepal width</th>
  <th>Petal length</th>
  <th>Petal width</th>
  </tr>
</thead>
<tbody>
  <tr>
  <td>setosa</td>
  <td>5.1</td>
  <td>3.5</td>
  <td>1.4</td>
  <td>0.2</td>
  </tr>
  <tr>
  <td>versicolor</td>
  <td>7.0</td>
  <td>3.2</td>
  <td>4.7</td>
  <td>1.4</td>
  </tr>
  <tr>
  <td>virginica</td>
  <td>6.3</td>
  <td>3.3</td>
  <td>6.0</td>
  <td>2.5</td>
  </tr>
</tbody>
</table>

Loading the dataset from the `sklearn` library and splitting into test and train set looks like this

{% highlight python %}
iris = datasets.load_iris()
data, target = iris["data"], iris["target"]

n, m  = data.shape
x_raw = np.ones((n, m + 1))
x_raw[:, 1:] = data

num_labels = len(np.unique(target))
y_raw = np.eye(num_labels)[target]

x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.4)
{% endhighlight %}

Next we would like to setup a classifier. As seen above a simple neural network structure looks like the one above. To keep it simple we have two hidden layery with 256 hidden units each. 4 input units for each flower feature and 3 output units for each of the 3 species classes. The following code builds up a Tensorflow graph. We are using a ReLU node as non-linear activation function and create three weight matrices between the input, hidden layers and output layer. In the end we define how we would like to calculate the loss or error and what optimizer is used to update the gradients of the error on the way.

{% highlight python %}
x_size = x_train.shape[1]   # 4 features, 1 bias
h_size = 256                # hidden units
y_size = y_train.shape[1]   # 3 flowers
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])

w_1 = tf.Variable(tf.random_normal((x_size, h_size), stddev=0.1))
w_2 = tf.Variable(tf.random_normal((h_size, h_size), stddev=0.1))
w_3 = tf.Variable(tf.random_normal((h_size, y_size), stddev=0.1))

# Input layer to hidden layer
h_1 = tf.nn.relu(tf.matmul(X, w_1))
h_1_drop = tf.nn.dropout(h_1, keep_prob)

# Hidden to hidden layer
h_2 = tf.nn.relu(tf.matmul(h_1_drop, w_2))
h_2_drop = tf.nn.dropout(h_2, keep_prob)

# Hidden to Output layer
y_hat = tf.matmul(h_2_drop, w_3)

predict = tf.argmax(y_hat, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
gradients = tf.train.AdamOptimizer(0.001).minimize(loss)
{% endhighlight %}

After we defined our computational graph we are ready to build the training routine. This is done by creating a Tensorflow `session`. We will also be starting a graph with it for our Tensorboard. Here we initialize all variables and run the training routine a hundret times and feed each flower datapoint into our graph. We evaluate the classifier after each epoch with the test and train set.

{% highlight python %}
def evaluate_step(sess, y, x_test, y_test):
  correct_predictions = tf.equal(predict, tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
  return sess.run(accuracy, feed_dict={X: x_test, y: y_test, keep_prob: 1.0})

with tf.Session().as_default() as sess:
  sess.run(tf.global_variables_initializer())

  steps = 50
  accuracy_value = 0

  for epoch in range(steps):
    for i in range(len(x_train)):       
      sess.run(gradients, feed_dict={
        X: x_train[i: i + 1], 
        y: y_train[i: i + 1],
        keep_prob: 0.5
      })
    
    new_accuracy_value = evaluate_step(sess, y, x_test, y_test, accuracy_value)
    if new_accuracy_value > accuracy_value:
      accuracy_value = new_accuracy_value
      print("In Epoch {} Accuracy {:10.5f}".format(epoch + 1, accuracy_value))
{% endhighlight %}

This should yield between 96% and 98% accuracy. A few things to note: we applied dropout, which means that at random nodes in the network are removed to improve the generalization error. Instead of using a simple gradient descent we used the Adam optimizer, which leads in some cases to improved results. In this particular example we could have got away with 1 hidden layer, for the fun of it we used 2.

### Keras

Tensorflow can be tedious to write. Here is how you would define the same model above in Keras, a high level abstraction ontop of Tensorflow. We will not rewrite the entire code but only the useful fragments.

{% highlight python %}
import tensorflow as tf
from keras.layers import Dropout
from keras import backend as K
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras.layers import Dense

# Data setup here...

# Model
x = Dense(256, activation='relu')(x_train)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
y_hat = Dense(3, activation='softmax')(x)

# Loss and optimization
loss = tf.reduce_mean(categorical_crossentropy(y, y_hat))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Training here...

# Evaluate accuracy
accuracy_value = accuracy(labels, preds)
with sess.as_default():
  acc_value.eval(feed_dict={
    X: x_test,
    y: y_test,
    K.learning_phase(): 0
  })
{% endhighlight %}

It is easy to see that working with keras can be much easier when quickly evaluating models. The abstraction is more suitable to think about design than about code. Tensorflow models can get to 2-3k lines of code quickly.

### Wrap up

In this post we have seen how to use Tensorflow as a neural network engine classifying the Iris data. It shows also what can be done with the data at our disposal. Big data questions arise more frequently due to companies saving more user data. However most companies do not leverage these kinds of data in other useful ways. Maybe there are a few classifiers in your application already, you just have not noticed! 

### Sources

- [Tensorflow](https://www.tensorflow.org/){:target="_blank"}
- [Keras](https://keras.io/){:target="_blank"}
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker){:target="_blank"}
