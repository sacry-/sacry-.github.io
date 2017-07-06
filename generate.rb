require 'date'


def generate_file(title)
  normalized_title = title.gsub(/\s+/, "-").downcase
  return "#{Date.today.to_s}-#{normalized_title}.md"
end

def generate_text(title, index)
  parts = []
  parts << "---"
  parts << "layout: post"
  parts << "author: Matthias Nitsche"
  parts << "title: #{title}"
  parts << "keywords: [tti]"
  parts << "tags: tti"
  parts << "index: #{index}"
  parts << "---"
  parts << '
This is the first paragraph. It should give a small introduction to the topic. The reader should want to read on after this. Be precise/provoking and direct.

<b>Here comes</b> the introduction..

{% include image.html url="/images/wordcloud-cloud.png" description="source: http://quotesnhumor.com/top-30-funny-cat-memes/" %}

### Interesting Aspect

### Additional Information

### Papers, Journals and Books

### Tooling/Programming

### Examples

### Conclusion

### Sources

- [link1](https://google.com)
- [link2](https://google.com)
'
  return parts.join("\n")
end


def create_post(title, index)
  file = generate_file(title)
  exists = File.exist?(file)
  unless exists
    text = generate_text(title.gsub(/\d+\s*/, ""), index)
    File.open("./_posts/#{file}", "w+") {|f| f.write(text) }
  end
  return (not exists)
end

def new_post(title, index: -1)
  unless create_post(title, index)
    print("Post already exists: #{title}")
  end
end

def prepare_all_posts
  titles = []
  headline = /###\s+General/
  headword_line = /^\s+-\s.+$/
  File.open("./README.md", "r") do |f|
    lines = f.read.split("\n")
    lines.each_with_index do |line, index|
      if line =~ headline
        lines[index,lines.size].each do |title|
          next if title.strip.empty? || title =~ headline || title =~ headword_line
          title = title.gsub(/\./, "").strip
          titles << title
        end

        break
      end
    end
  end

  titles.each_with_index do |title, index|
    new_post(title, index: index + 1)
  end
end

def new_post_with_args
  if ARGV.size < 1
    raise StandardError.new("Must provide (multi) argument 'Title' as argument.")
  end

  title = ARGV.join(" ")
  new_post(title)
end


# new_post_with_args
prepare_all_posts






