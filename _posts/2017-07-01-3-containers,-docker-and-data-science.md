---
layout: post
author: Matthias Nitsche
title: Containers, Docker and Data Science
keywords: [containers, docker, data science, machine learning, data engineering]
index: 3
img: containerization.jpg
---

Hardware virtualization and virtual machines were yesterday! Today is the age of containers. Just kidding, but containers are here to stay for a long time. 

{% include image.html url="/images/containerization.jpg" description="source: https://c1.staticflickr.com/4/3678/10921728045_4de014f856_b.jpg" %}

<blockquote>A container image is a lightweight, stand-alone, executable package of a piece of software that includes everything needed to run it: code, runtime, system tools, system libraries, settings. 
<cite>- https://www.docker.com/what-container</cite></blockquote>

Docker is one of the many providers to offer support for containers through various tools like the Docker cli, Dockerhub and the Dockerfile standard. The best thing about them is simple: If you need to install several libraries and software tools to support your own use case Docker helps in defining this platform in a simple Dockerfile format. From third party tools to runtime setups, operating systems and how the code is loaded most of the things you think of are possible. Containers also offer a standardized interface to the outside world meaning that IaaS or PaaS providers are capable of offering their services given your Dockerfile alone.

In this post I would like to create a small runtime for a data science / machine learning workflow. The hard thing about these workflows is that you need about everything. Programming environments, dozens of programming libraries, dozens of system libraries at best running on a linux, at worst with GPU accelerated and system specific support, visualization tools and access to a broad range of databases. This seems like a cool use case!

### Interesting Aspect

### Additional Information

### Papers, Journals and Books

### Tooling/Programming

### Examples

### Conclusion

### Sources

- [link1](https://google.com)
- [link2](https://google.com)
