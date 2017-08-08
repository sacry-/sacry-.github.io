---
layout: post
author: Matthias Nitsche
title: Machine Learning in the Cloud
keywords: [machine learning, cloud, research]
index: 4
img: data-science-cover.png
---

Machine learning is a hot topic nowadays. Every big player quickly scaled up their research labs within the last years. Naming a few Googles Deep Mind and Google Brain, Microsoft Research AI, Facebook FAIR, OpenAI (sponsored by Elon Musk) and NVIDIA sponsoring GPU clusters to many. In this post I would like to explore what the big players offer with their machine learning solutions. We will take a look on Google, Microsoft and NVIDIA.

{% include image.html url="/images/data-science-cover.png" description="source: https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Social_Network_Analysis_Visualization.png/1024px-Social_Network_Analysis_Visualization.png" %}

### Google Machine Learning

Googles latest ecosystem evolves around the Google Cloud Platform (GCP). `GCP` offers central support for hardware, deployment software, cluster management software, containerization and registry, build in solutions for `Kubernetes`, vast analytics platforms like `Big Query` and machine learning. Machine learning needs a lot of distributed GPU power and Google has become one of the largest suppliers for this. Their Tensor Processing Unit (`TPU`) is one of the most evolved GPUs for high throughput learning applications that generally need to solve numerical optimization problems on floating point numbers. 

Google is probably the most involved researching company for general AI and the very promising subfield called machine learning with its very hot subfield deep learning. In deep learning we take universal approximators and stack layers upon layers of non linear weight approximations as deep and wide the hardware is possible to solve. In general there are thresholds from which adding weight matrices and layers does not do much for the accuracy. For the complex algorithms that run behind the scences Google has opened to a variety of development tools and SaaS products.

<b>Tensorflow</b> is probably the most promising deep learning platform within the last years.

<blockquote>Tensorflow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.<cite>- <a href="https://www.tensorflow.org/" target="_blank">Tensorflow.org</a></cite></blockquote>

It is not only a deep learning tool but a mathematical modelling tool for powerful graphical abstractions. It scales extremely well and easily to GPUs with CUDA support. The sole purpose for Tensorflow is to accelerate research within Google (though it is believed that they use an optimized version).

<b>Googles machine learning solutions</b> are a great way to get going with nothing more than a company account and an oauth token. Whether it is <a href="https://cloud.google.com/ml-engine/" target="_blank">general purpose ML</a>, <a href="https://cloud.google.com/jobs-api/" target="_blank">Job postings</a>, <a href="https://cloud.google.com/video-intelligence/" target="_blank">Video intelligence</a>, <a href="https://cloud.google.com/vision/" target="_blank">Image analysis</a>, <a href="https://cloud.google.com/speech/" target="_blank">Speech recognition</a>, <a href="https://cloud.google.com/natural-language/" target="_blank">Natural language processing</a> or <a href="https://cloud.google.com/translate/" target="_blank">Google translate</a>, there is a service with pretrained - on billions of datapoints - state of the art ML models. All you have to do is sent your data to them and pay a little money.

<div class="panel panel-default">
  <div class="panel-heading"><b>Google Natural language processing</b></div>
  <div class="panel-body">
    Natural language processing a small guide.
  </div>
</div>

### Microsoft Azure Machine Learning

<a href="https://azure.microsoft.com/en-us/services/machine-learning/" target="_blank">Microsoft Azure machine learning</a>

@TODO

### NVIDIA Vision and Deep Learning

@TODO

### Sources

- [link1](https://google.com)
- [link2](https://google.com)
