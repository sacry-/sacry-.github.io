---
layout: post
author: Matthias Nitsche
title: About the lecture and BPaaS-project
keywords: [tti]
index: 2
img: wordcloud-cloud.png
---

The overall topic in the TTI lecture can be summarized with the two words "cloud computing". Create distributed architectures, deploy and provision services horizontally and scale them as needed. In the following I will summarize the lecture and make remarks to the BPaaS project/assigment of the team I worked in.

{% include image.html url="/images/wordcloud-cloud.png" description="source: http://www.wordclouds.com/" %}

### Cloud Computing

<blockquote>Cloud computing is a model for enabling convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction. <cite>- NIST</cite></blockquote>

For me the most striking point of cloud computing is that by definition the setup and usability should be "convenient", "rapid", with "minimal management effort" working on "shared resources" that should be replaceable e.g. "configurable" and "on-demand". Services like Heroku are prime examples of what this means. Deploying a service with Heroku and Github is not much more than
    
    cd app
    heroku login # authenticate here
    git init
    git add .
    git commit -m "initial commit"
    heroku create
    git push heroku master

After this everything is setup up. Depending on your requirements and framework you are using e.g. Ruby on Rails, Django or Node.js you need to provide a way to start the service. If the Heroku defaults are insufficient, tools like Docker help in making your services deployable.

#### BPaaS project

For the project we used a barebone server with Docker installed. This is sufficient for some cases but not for all. In retrospec we should have standardized this early to see how it behaves and works in production. Solutions like Kubernetes would have been a good choice instead of e.g. Heroku or barebone servers to deploy services and workers. Our biggest mistake was starting too late and using a programming language / framework (javascript/node.js) nobody ever worked with before.

### Virtualization vs Containers



#### BPaaS project

### IaaS and PaaS

#### BPaaS project

### Multi Tenancy, Web Services and REST

#### BPaaS project

### SOA, API-Economies

#### BPaaS project

### Hadoop and MapReduce

#### BPaaS project

### NoSQL

#### BPaaS project

### GraphDB

#### BPaaS project

### Web Search - Information Retrieval

#### BPaaS project

### Semantic Web

#### BPaaS project


### Sources

- [Heroku Setup](https://devcenter.heroku.com/articles/git)
- [link2](https://google.com)
