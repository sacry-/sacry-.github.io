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

2. <b>Containers, Docker and Machine Learning</b>

    Containers are immutable objects that directly interact with the OS Host system. They are used to standardize and encapsulate the dependencies used for most applications. One of the most used container engine providers is [Docker](https://www.docker.com/){:target="_blank"} with the Dockerfile format. In this post I will elaborate the mechanisms of containers and how they can be used in machine learning applications.

3. <b>Machine Learning in the Cloud</b>

    All the global players jumped on the AI train. Whether it is [Google](https://cloud.google.com/products/machine-learning/), [Microsoft](https://azure.microsoft.com/en-us/services/machine-learning/), [Facebook](https://research.fb.com/category/facebook-ai-research-fair/) with NVIDIA as their computational backend for high end GPU clusters. They set up research labs with the brightest people the field has to offer. Only big players can afford to run the algorithms with state of the art results and more importantly to ship products and applications to the market that scale to the billions of people interested. Here we will explore whats on the market and what developers of today can do with it. 

4. <b>Machine Learning with Spark</b>

    [Apache Spark](https://spark.apache.org/){:target="_blank"} is a great framework for concurrent data processing using Resilient Distributed Datasets (RDDs) or in short dataframes. It is entirely in memory and can scale up to large datasets. Spark is a perfect engine for trying out lots of different machine learning models. The libraries and its eco system made most of the standard algorithms like Random forest, SVM or linear regression available.

5. <b>Neural Networks with Tensorflow</b>

    [Tensorflow](https://www.tensorflow.org/){:target="_blank"} is a computational graph engine that does automatic differention on mathematical models, such as deep learning algorithms. Todays data is large, unstructured and often entirely unnormalized. Tensorflow leverages clusters of GPU to approximate different cost functions automatically learning to represent data in a different dimension. The setup and debugging of such models can be tedious. Here we will have a basic introduction to Tensorflow models.

The table is long and there is much todo. This basic introduction should have made you familiar with the concepts I would like to cover. Every article will have a longer sources section, possibly papers as well.

### Sources

- [Docker](https://www.docker.com/){:target="_blank"}
- [Google Cloud Engine](https://cloud.google.com/compute/){:target="_blank"}
- [Apache Spark](https://spark.apache.org/){:target="_blank"}
- [Tensorflow](https://www.tensorflow.org/){:target="_blank"}
- [gRPC](http://www.grpc.io/){:target="_blank"}
- [Google Protobuf](https://developers.google.com/protocol-buffers/){:target="_blank"}
- [Google Cloud Machine Learning](https://cloud.google.com/products/machine-learning/){:target="_blank"}
- [Microsoft Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/){:target="_blank"}
- [Facebook FAIR](https://research.fb.com/category/facebook-ai-research-fair/)
- [Prof. Steffens](http://users.informatik.haw-hamburg.de/~steffens/){:target="_blank"}

