---
layout: post
author: Matthias Nitsche
title: Machine Learning with Spark
keywords: [spark, machine learning, hadoop, mapreduce]
index: 5
img: server-farm.jpg
---

Spark is an engine for large-scale data processing running on the MapReduce paradigm 100x faster in memory and 10x faster on disk than Hadoop. Spark can be used for parallel machine learning applications leveraging the hardware efficiently. In this post we will explore Spark and how it can be used to train a machine learning classifier.

{% include image.html url="/images/server-farm.jpg" description="source: https://c1.staticflickr.com/1/686/33371413545_360923fbd7_b.jpg" %}

### Spark Setup

Spark relies on the MapReduce paradigm. The backend of Spark relies on Hadoop the beforegoing MapReduce engine. The setup can be hard and unsatisfactory. Setting up Spark and Hadoop on a MacOsx you have to install the dependencies

{% highlight bash %}
brew install hadoop
brew install apache-spark
{% endhighlight %}

Next we need to install `pyspark` for python.

{% highlight bash %}
pip install --upgrade pip
pip install pyspark
# test it with
pyspark
# open browser at http://localhost:4040
{% endhighlight %}

Next we want to connect Spark to a `Jupyter notebook` so jupyter would need to be installed

{% highlight bash %}
pip install jupyter
# test it with
jupyter notebook
# open browser at http://localhost:8888
{% endhighlight %}

Now what we actually want is a fully connected and functioning notebook with Spark, Hadoop and Python environment ready. To do this create a file `jupyspark.sh` and add the following

{% highlight bash %}
#!/bin/bash
export SPARK_HOME=`brew info apache-spark | grep /usr | tail -n 1 | cut -f 1 -d " "`/libexec
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export HADOOP_HOME=`brew info hadoop | grep /usr | head -n 1 | cut -f 1 -d " "`/libexec
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --NotebookApp.open_browser=True --NotebookApp.ip='localhost' --NotebookApp.port=8888"

${SPARK_HOME}/bin/pyspark \
--master local[4] \
--executor-memory 1G \
--driver-memory 1G \
--conf spark.sql.warehouse.dir="file:///tmp/spark-warehouse" \
--packages com.databricks:spark-csv_2.11:1.5.0 \
--packages com.amazonaws:aws-java-sdk-pom:1.10.34 \
--packages org.apache.hadoop:hadoop-aws:2.7.3
{% endhighlight %}

The file contains several important configurations. The `$SPARK_HOME` and `$HADOOP_HOME` callable from the shell. With this we make sure that the environment is properly setup. Next we set the `$PYSPARK_DRIVER_PYTHON` driver and options for it `$PYSPARK_DRIVER_PYTHON_OPTS`. With this we tell `pyspark` when it is started how it should be started. We would like to setup a jupyter notebook. The last command `${SPARK_HOME}/bin/pyspark` starts the Spark cluster and the jupyter notebook at once, so giving the right permissions and running it yields

{% highlight bash %}
chmod +x jupyspark.sh
./jupyspark.sh
# Spark at http://localhost:4040
# jupyter notebooks at http://localhost:8888
{% endhighlight %}

Now we are ready to do some actual coding and modelling.

{% include image.html url="/images/spark-browser.png" description="Spark browser <code>http://localhost:4040</code>" %}

### Spark Machine Learning

In order to train a machine learning algorithm with Python and Spark we need to do some proper imports. `pyspark` is the library we would like to import. The jupyter session started before makes it possible to call a `SparkContext` automatically aliased by `sc`.

{% highlight python %}
import pyspark as ps

from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
{% endhighlight %}

We first read the titanic data `CSV` file into a resilient distributed dataframe (RDD).

{% highlight python %}
def read_csv(filename, header=True):
  rdd = sc.textFile(filename)
  if not header:
    return rdd
  header = rdd.first()
  return rdd.filter(lambda line: line != header)

rdd = read_csv("./titanic.csv", header=True)
print("count: {}".format(rdd.count()))
{% endhighlight %}

The `RDD` contains the titanic data with wich we try to predict which passenger survived. We get several variables (we only use 3 here). The data contains the following entries and attributes

<table class="table table-md table-striped table-bordered">
<thead>
  <tr>
  <th>#</th>
  <th>Class</th>
  <th>Age</th>
  <th>Sex</th>
  <th>Survived</th>
  </tr>
</thead>
<tbody>
  <tr>
  <td>1</td>
  <td>"1st class"</td>
  <td>"adults"</td>
  <td>"man"</td>
  <td>"no"</td>
  </tr>
  <tr>
  <td>2</td>
  <td>"3rd class"</td>
  <td>"adults"</td>
  <td>"woman"</td>
  <td>"yes"</td>
  </tr>
</tbody>
</table>

We need to normalize the text data into machine readable formats.

{% highlight python %}
def normed_row(line):
  passenger_id, klass, age, sex, survived = [segs.strip('"') for segs in line.split(',')]
  klass = int(klass[0]) - 1
  
  features = [
    klass,
    (1 if age == 'adults' else 0),
    (1 if sex == 'women' else 0)
  ]

  return LabeledPoint(1 if survived == 'yes' else 0, features)

normed_rdd = rdd.map(normed_row)
train_rdd, test_rdd = normed_rdd.randomSplit([0.7, 0.3], seed = 0)
print("train: {} - test: {}".format(train_rdd.count(), test_rdd.count()))
{% endhighlight %}

In order to use the `RDD` for a Multilayer Perceptron we have to convert it to a `ML` vector.

{% highlight python %}
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils

sqlContext = SQLContext(sc)
train_df = MLUtils.convertVectorColumnsToML(sqlContext.createDataFrame(train_rdd))
test_df = MLUtils.convertVectorColumnsToML(sqlContext.createDataFrame(test_rdd))
{% endhighlight %}

Next we define the `MultilayerPerceptronClassifier` that is a simple feedforward neural network with discrete binary outputs on each node.

{% highlight python %}
layers = [3, 10, 5, 2]
trainer = MultilayerPerceptronClassifier(maxIter=300, layers=layers, blockSize=256)
{% endhighlight %}

Then we train the classifier and predict the accuracy of the model.

{% highlight python %}
model = trainer.fit(train_df)
result = model.transform(test_df)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
{% endhighlight %}

The accuracy is around 80%, the best accuracy that can be predicted without feeding in prior knowledge about the passengers is around 82%. We only used 3 of the 20 variables to predict it though. Seems reasonable.

### Wrap up

In this post we have seen how Spark can be leveraged to classify the survival data of the Titanic. While the algorithms are very solid and could be - in theory - executed completely in parallel, they are not state of the art. Connecting Spark with libraries like Tensorflow is a non trivial task that could cover several blog posts alone. In context of large data collections Spark is a good option as long as the computational servers have enough RAM.

### Sources

- [Apache Spark](https://spark.apache.org/){:target="_blank"}
- [Apache Spark documentantion](https://spark.apache.org/docs){:target="_blank"}
- [Python Jupyter](http://jupyter.org/){:target="_blank"}
