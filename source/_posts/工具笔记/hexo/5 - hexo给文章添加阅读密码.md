---
title: 5 - hexo给文章添加阅读密码(password=123)
tags:
  - hexo
password: 123
abstract: 'Welcome to my blog, enter password to read.'
message: 'Welcome to my blog, enter password to read.'
categories:
  - 工具笔记
  - hexo
abbrlink: 2899485557
date: 2023-02-20 12:05:50
index_img:
---







## [√] 安装hexo-blog-encrypt插件

---

- 在hexo目录下`npm install hexo-blog-encrypt`
- 在`/Hexo/_config.yml`文件中添加内容:

```yml
encrypt:
	enable:true
```

> alec：
>
> - encrypt，v. 加密,将…译成密码

## [√] 使用插件

---

- 在想要使用加密功能的Blog头部加上对应文字：

```yaml
---
title: Hexo加密功能
date: 2019-09-04 23:20:00   
tags: [学习笔记,Hexo]
categories: Hexo      
password: smile   
abstract: Welcome to my blog, enter password to read. 
message: 密码输入框上描述性内容
---
```

- 其中：
    - password: 该Blog使用的密码
    - abstract: Blog摘要文字（少量）
    - message: 密码框上的描述性文字





