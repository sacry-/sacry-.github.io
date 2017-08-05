---
layout: post
author: Matthias Nitsche
title: Introduction and Overview
keywords: [tti, introduction, overview]
index: 1
img: wordcloud-cloud.png
---

The need for scalable services and rapid development cycles calls for new ways of thinking about infrastructure, platforms and services. Cloud computing aims exactly at this and enables users to conveniently scale and publish their ideas into the world wide web. This blog aims to provide insights into deep learning providing instructions and presenting ideas on how to train, deploy and provision such models.

{% include image.html url="/images/wordcloud-cloud.png" description="source: http://www.wordclouds.com/" %}

In the following I will lay out basic terminology as well as a "table of content" for the upcoming posts. This blog is the final assigment of the lecture "Technik und Technologie verteilter Informationssysteme" ("Methods and technology of distributed information systems") in short "TTI" which is part of the M.Sc. Informatics curriculum of the university of applied sciences Hamburg held by [Prof. Steffens](http://users.informatik.haw-hamburg.de/~steffens/){:target="_blank"}.

### Table of content

1. <b>About the lecture and BPaaS project</b>

    Some general thoughts and summaries on the TTI lecture and Business-process-as-a-service project. Successes, failures and what I have learned during the lecture.

2. <b>Containers, Docker and data science</b>

    Containers are immutable objects that directly interact with the OS Host system. They are used to standardize and encapsulate the dependencies used for most applications. One of the most used container engine providers is [Docker](https://www.docker.com/){:target="_blank"} with the Dockerfile format. In this post I will elaborate the mechanisms of containers and how they can be used in data science applications.

3. <b>Kubernetes and Google Cloud Engine</b>

    [Kubernetes](https://kubernetes.io/){:target="_blank"} can scale roughly anything as long as it is run in containers. Googles open source platform written in Go is one of the most used frameworks to deploy services on 3rd party cloud providers such as AWS or GCE. [Google Cloud Engine](https://cloud.google.com/compute/){:target="_blank"} on the other hand comes with its own hardware, container registry and build in resources for managing Kubernetes clusters.

4. <b>Training a neural network with Spark and Tensorflow</b>

    [Apache Spark](https://spark.apache.org/){:target="_blank"} is a great framework for concurrent data processing using Resilient Distributed Datasets (RDDs) or in short dataframes. It is entirely in memory and can scale up to large datasets. [Tensorflow](https://www.tensorflow.org/){:target="_blank"} on the other hand is a computational graph engine that does automatic differention on mathematical models, such as deep learning algorithms. As Tensorflow needs a lot of computational power and Spark stores and accesses huge datasets efficiently how can we leverage both technologies to build a reasonable pipeline?

5. <b>Deploying Tensorflow models</b>

    A topic often ignored in the literature is how to actually serve trained models. We will take a deeper look on how to deploy and serve Tensorflow models in the cloud. For this [gRPC](http://www.grpc.io/){:target="_blank"} - a high-performance, open-source RPC framework - and [Google Protobuf](https://developers.google.com/protocol-buffers/){:target="_blank"} - a fast data interchange format - will be used.


The table is long and there is much todo. This basic introduction should have made you familiar with the concepts I would like to cover. Every article will have a longer sources section, possibly papers as well.

### Sources

- [Docker](https://www.docker.com/){:target="_blank"}
- [Kubernetes](https://kubernetes.io/){:target="_blank"}
- [Google Cloud Engine](https://cloud.google.com/compute/){:target="_blank"}
- [Apache Spark](https://spark.apache.org/){:target="_blank"}
- [Tensorflow](https://www.tensorflow.org/){:target="_blank"}
- [gRPC](http://www.grpc.io/){:target="_blank"}
- [Google Protobuf](https://developers.google.com/protocol-buffers/){:target="_blank"}
- [Google Cloud Machine Learning](https://cloud.google.com/products/machine-learning/){:target="_blank"}
- [Microsoft Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/){:target="_blank"}
- [Prof. Steffens](http://users.informatik.haw-hamburg.de/~steffens/){:target="_blank"}

