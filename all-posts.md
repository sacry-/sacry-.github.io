---
layout: page
title: All Posts
permalink: /all-posts/
---

<p>Links to all posts published in this blog.</p>

<table class="table table-sm table-striped">
  {% assign sorted = (site.posts | sort: 'index') %}
  {% for post in sorted %}
    <tr>
      <td>{{ forloop.index }}.</td>
      <td><a href="{{ post.url }}">{{ post.title }}</a></td>
      <td>
        <small>{{ post.date | date: "%B %e, %Y" }}</small>
      </td>
    </tr>
  {% endfor %}
</table>