---
layout: post
author: Matthias Nitsche
title: Introduction and Overview
keywords: [tti, introduction, overview]
index: 1
img: 404.jpg
---

The need for scalable services and rapid development cycles calls for new ways of thinking about infrastructure, platforms and services. Cloud computing aims exactly at this and enables users to conveniently scale and publish their ideas into the world wide web. This blog aims to provide insights into deep learning providing instructions and presenting ideas on how to train, deploy and provision such models.

In the following I will lay out basic terminology as well as a "table of content" for the upcoming posts. This blog is the final assigment of the lecture "Technik und Technologie verteilter Informationssysteme" ("Methods and technology of distributed information systems") in short "TTI" which is part of the M.Sc. Informatics curriculum of the university of applied sciences Hamburg held by professor Steffens.

### Table of content

1. About the lecture and BPaaS project

    Some general thoughts and summaries on the TTI lecture and Business-process-as-a-service project.

2. Containers, Docker and data science

    Containers are immutable objects that directly interact with the OS Host system. They are used to standardize and encapsulate the dependencies used for most applications. One of the most used container engine providers is Docker with the Dockerfile format. In this post I will elaborate the mechanisms of containers and how they can be used in data science applications.

3. Kubernetes and Google Cloud Engine

    Kubernetes can scale roughly anything as long as it is run in containers. Googles open source platform written in Go is one of the most used frameworks to deploy services on 3rd party cloud providers such as AWS or GCE. Google Cloud Engine on the other hand comes with its own hardware, container registry and build in resources for managing Kubernetes clusters.

4. Machine learning in the cloud

    Machine learning and especially deep learning is incredbly hard. Doing it right and correct on large scale datasets that can surpass the 10 billion mark is prone to a lot of trial and error. Automated machine learning services provided by Google or Microsoft become a viable alternative. What do you need to look out for and how to costumize when everything is a black box? What are the legal culprits?

5. Training a neural network with Spark and Tensorflow

    Spark is a great framework for concurrent data processing using Resilient Distributed Datasets (RDDs) or in short dataframes. It is entirely in memory and can scale up to large datasets. Tensorflow on the other hand is a computational graph engine that does automatic differention on mathematical models, such as deep learning algorithms. As Tensorflow needs a lot of computational power and Spark stores and accesses huge datasets efficiently how can we leverage both technologies to build a reasonable pipeline?

6. Training models on GPU clusters

    Deep learning models are all about the depth and width. More deeper layers and more hidden units per layer. The biggest models developed at Facebook or Google are as deep as 150 layers. Training such networks takes several weeks and an incredible amount of ressources. Most modern deep learning architectures leverage parallelization and lossy floating point arithmetic on GPU clusters. In this post I will explore what it takes to train on GPUs with Tensorflow.

7. Deploying Tensorflow models

    A topic often ignored in the literature is how to actually serve trained models. We will take a deeper look on how to deploy and serve Tensorflow models in the cloud. For this GRPC - a high-performance, open-source RPC framework from Google and Protobuf - Googles fast data interchange format - will be used.

8. Continuous Deployment of machine learning models

    @TODO


### Sources

- [link1](https://google.com)
- [link2](https://google.com)
