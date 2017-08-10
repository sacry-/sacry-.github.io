---
layout: post
author: Matthias Nitsche
title: Machine Learning in the Cloud
keywords: [machine learning, cloud, research]
index: 4
img: data-science-cover.png
---

Machine learning is a hot topic nowadays. Every big player quickly scaled up their research labs within the last years. Naming a few Googles Deep Mind and Google Brain, Microsoft Research AI, Facebook FAIR, OpenAI (sponsored by Elon Musk) and NVIDIA sponsoring GPU clusters to many. In this post I would like to explore what the big players offer with their machine learning solutions. We will take a look on Google and Microsoft.

{% include image.html url="/images/data-science-cover.png" description="source: https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Social_Network_Analysis_Visualization.png/1024px-Social_Network_Analysis_Visualization.png" %}

### Google Machine Learning

Googles latest ecosystem evolves around the Google Cloud Platform (GCP). `GCP` offers central support for hardware, deployment software, cluster management software, containerization and registry, build in solutions for `Kubernetes`, vast analytics platforms like `Big Query` and machine learning. Machine learning needs a lot of distributed GPU power and Google has become one of the largest suppliers for this. Their Tensor Processing Unit (`TPU`) is one of the most evolved GPUs for high throughput learning applications that generally need to solve numerical optimization problems on floating point numbers. 

Google is probably the most involved researching company for general AI and the very promising subfield called machine learning with its very hot subfield deep learning. In deep learning we take universal approximators and stack layers upon layers of non linear weight approximations as deep and wide the hardware is possible to solve. In general there are thresholds from which adding weight matrices and layers does not do much for the accuracy. For the complex algorithms that run behind the scences Google has opened to a variety of development tools and SaaS products.

<b>Tensorflow</b> is probably the most promising deep learning platform within the last years. We will look into the framework in a later post.

<blockquote>Tensorflow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.<cite>- <a href="https://www.tensorflow.org/" target="_blank">Tensorflow.org</a></cite></blockquote>

It is not only a deep learning tool but a mathematical modelling tool for powerful graphical abstractions. It scales extremely well and easily to GPUs with CUDA support. The sole purpose for Tensorflow is to accelerate research within Google (though it is believed that they use an optimized version).

<b>Googles machine learning solutions</b> are a great way to get going with nothing more than a company account and an oauth token. Whether it is <a href="https://cloud.google.com/ml-engine/" target="_blank">general purpose ML</a>, <a href="https://cloud.google.com/jobs-api/" target="_blank">Job postings</a>, <a href="https://cloud.google.com/video-intelligence/" target="_blank">Video intelligence</a>, <a href="https://cloud.google.com/vision/" target="_blank">Image analysis</a>, <a href="https://cloud.google.com/speech/" target="_blank">Speech recognition</a>, <a href="https://cloud.google.com/natural-language/" target="_blank">Natural language processing</a> or <a href="https://cloud.google.com/translate/" target="_blank">Google translate</a>, there is a service with pretrained - on billions of datapoints - state of the art ML models. All you have to do is sent your data to them and pay a little money. Unfortunately individual users in the European Union will have a hard time subscribing for the trial periods. 

<blockquote>
In the European Union, Google Cloud Platform services can be used for business purposes only.
<cite>- <a href="https://cloud.google.com/free/docs/frequently-asked-questions" target="_blank">GCP FAQ</a></cite></blockquote>

<blockquote>
If you are located in the European Union and the sole purpose for which you want to use Google Cloud Platform services has no potential economic benefit you should not use the service. If you have already started using Google Cloud Platform, you should discontinue using the service. See Create, modify, or close your billing account to learn how to disable billing on your projects.
<cite>- <a href="https://cloud.google.com/free/docs/frequently-asked-questions" target="_blank">GCP FAQ</a></cite></blockquote>

If you are with a company the trial includes 12 months of usage with 300$ credits (which is enough for playing around).

<div class="panel panel-default">
  <div class="panel-heading"><b>Google natural language processing</b></div>
  <div class="panel-body">
  Before running anything with GCP you should download the <a href="https://cloud.google.com/sdk/gcloud/" target="_blank">Google Cloud SDK</a> for your local shell usage. Thusly you should be able to run

{% highlight python %}$ gcloud auth application-default login{% endhighlight %}

  To setup library usage for Python or Golang you will need a <mark>service_account_file</mark> from google exported in your env 

{% highlight python %}$ export GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>{% endhighlight %}

  As we will run the examples with python, you should be able to run

{% highlight python %}
$ python --version
# Python 3.6.1
$ pip install --upgrade google-cloud-language{% endhighlight %}
  
  Afterwards we would like to create a file called <code>sentiments.py</code> and import the google libraries

{% highlight python %}
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
{% endhighlight %}
  
  Next we would like to do some basic sentiment analysis.

{% highlight python %}
client = language.LanguageServiceClient()

content = "The good people are all dead and gone. Now it is your turn!"
document = types.Document(content=content, type=enums.Document.Type.PLAIN_TEXT)
annotations = client.analyze_sentiment(document=document)

score = annotations.document_sentiment.score
magnitude = annotations.document_sentiment.magnitude
print('Sentiment of document: {} with magnitude {}'.format(score, magnitude))

for sentence in annotations.sentences:
  sentence_sentiment = sentence.sentiment.score
  print('{}\n\t sentiment score: {}'.format(sentence, sentence_sentiment))
{% endhighlight %}

After this you should see the sentiments for the sentences.
  </div>
</div>

### Microsoft Azure Machine Learning

Azure is the equivalent solution to Googles GCP. Microsofts Azure has roughly anything to offer GCP offers their costumers. To properly distinguish one from the other is hard. Exceptions to this are their eco systems and development targets e.g. C# vs Golang, Windows vs. Linux. Lets skip all the noise enumerating endless perks and benefits Azure has to offer and take the GCP introduction as a reference.

Microsoft Azure seems much more friendly in the beginning. Going to <a href="https://azure.microsoft.com/en-us/services/machine-learning/" target="_blank">Microsoft Azure machine learning</a> you can immediately start with a free 8 hour trial period. You are then navigated to the Microsoft Azure Machine Learning Studio which is a similiar concept to <a href="https://www.rstudio.com/" target="_blank">R Studio</a>, having something like notebooks as well. Much more interesting even is the fact that you can model computational graph flows in a graphical editor. Go sign up for it if you want to follow along!

In a previous post about docker and machine learning notebooks were a concept where you created cells containing code that could be run over and over again. Everything the cells before mutated was taken into account. In data processing this is especially useful if you first write code to load data, then to process data, then to transform the data.

Azure Studio is ordered into <b>workspaces</b> that can be understood as projects. A workspace can have regions (e.g. your servers). Lets start a new <b>experiment</b> with "+ new". I choose an example for <mark>Clustering: Find similar companies</mark>. 

Each experiment has access to <b>datasets</b> and <b>modules</b>. There are example datasets but you can also create your own, making it really easy to just use one from the beginning. In our case it is the <mark>Wikipedia SP 500 Dataset</mark> provided by Microsoft. Each dataset can be easily downloaded, lets have a peek at the dataset before we continue

{% highlight bash %}
$ cat Wikipedia\ SP\ 500\ Dataset.csv
# Title,Category,Text
# Apple Inc.,Information Technology, nasdaq 100 component s p 500 component ...
{% endhighlight %}

A module can be thought of as anything you would like to do with data. Be it data transformation or model classification. To name a few important ones "Data Transformation", "Feature Selection", "Data Input and Output" and of course "Machine Learning" which is further divided into evaluation, scoring, intialization and training. Lets see how the example model looks like

{% include image.html url="/images/azure-clustering.png" description="source: https://studio.azureml.net/" %}

The experiment can be abstracted by three components. 

<ol>
  <li>
    <b>Read data and transform it</b>
    <p>
    In our case we read the data and feature hash the text data to numeric features using the Vowpal Wabbit library. Then the R script selects the features with the best PCA scores, e.g. the ones with the highest principal components that explain the data best. This is most often done with a singular value decomposition (SVD) chosing the values with their highest eigenvalues squared e.g. singular values. Ironically in the text domain using truncated SVD, e.g. with a maximum amount of eigenvalues to keep is called Latent Semantic Indexing/Analysis (LSI/LSA). Lot of words for very similar concepts.
    </p>
  </li>
  <li>
    <b>Select the features of the data, choose a model with its parameters and train it</b>
    <p>
    In our example we choose all columns of our dataset and set the clustering algorithm to K-Means. Upon clicking on the K-Means, 3 parameters can be set e.g. Number of centroids (the number of clusters that can be maximaly assigned), Initialization (e.g. Random, maximum variance etc.) and the Random seed (for reproducability). The core weakness of K-Means is that we have to know the maximum number of centroids. What came first the chicken or the egg? How shall we know how many hidden useful categories can be clustered from our data before? How should our algorithm know that beforehand? The alternative approaches are K-nearest neighbour methods, that take the locality of the data into account. The parameters are then shifted to density estimation. Very problematic as well. Both procedures suffer by the curse of dimnesionality problem. In our example we have two configurations (K-Means with 10 and 3 centroids), thats why the second step is doubled.
    </p>
  </li>
  <li>
    <b>Capture results and display them</b>
    <p>
    The third step produces what kind of metadata shall be returned from the training. E.g. categorical, probabilities etc. and an output node for CSV results.
    </p>
  </li>
</ol>

Due to the limited knowledge I decided to download the dataset and evaluate the model myself. You could however also use the scoring functions Azure has to offer. Our example data has 10 centroids, mapping to the 10 categories of the business industry the company is in.

{% highlight python %}
import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score

df = pd.read_csv("results_10.csv", usecols=[0,1,2,4099])

print(df.describe())
categories = df.Category.unique()

mapping = {}
for index, category in enumerate(categories):
  mapping[category] = index

print(mapping)

centroids = df.Assignments.unique()
print("centroids", centroids)
print("v-measure:", v_measure_score(df.Assignments, df.Category))
{% endhighlight %}

In our example there was a mere 0.09 v measure score. That is extremely bad and we can only account for roughly 10% of the data making any kind of sense together. A point to note, it is not at all clear if that is really bad. The hypothesis however that our Azure clustering model assigns datapoints to a cluster that also happen to have the same business category is false. 

The model could be improved with some costumization, for which we would need at least the standard subscription or free trial. The major problem here is that we did not really do anything with the text except feature hashing. In text domains we need to tone down the number of words used. A simple but effective improvement would be `tf-idf` in combination with `LSA`, maybe stemming and stopword removal. A better improvement would be a neural language model such as `word2vec`.

### Wrap up

Machine learning in the cloud looks promising. With paid variants this could be a game changer for companies looking for machine learning solutions without the necessary knowledge to build up inhouse solutions. The problem however would be data privacy: Is a company actually authorized or willing to share its data to Googles, Microsofts or for that matter any cloud machine learning solution?

### Sources

- [Google Cloud Platform](https://cloud.google.com){:target="_blank"}
- [Tensorflow](https://www.tensorflow.org/){:target="_blank"}
- [Microsoft Azure](https://azure.microsoft.com/en-us/services/machine-learning/){:target="_blank"}
- [R Studio](https://www.rstudio.com/){:target="_blank"}
