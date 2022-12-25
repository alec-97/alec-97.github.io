---
title: 2 - fluid主题添加'一言'
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 151223124
---


## [√] 介绍

---

- 一言网（[hitokoto.cn](http://hitokoto.cn/)）创立于 2016 年，隶属于萌创团队，网站主要提供一句话服务。
- ‘一言’ 的初衷——动漫也好、小说也好、网络也好，不论在哪里，我们总会看到有那么一两个句子能穿透你的心。我们把这些句子汇聚起来，形成一言网络，以传递更多的感动。
- 简单来说，一言指的就是一句话，可以是动漫中的台词，也可以是网络上的各种小段子。 或是感动，或是开心，有或是单纯的回忆。来到这里，留下你所喜欢的那一句句话，与大家分享，这就是一言存在的目的。

官网链接：https://developer.hitokoto.cn/



## [√] 实践

---

#### [√] 在首页添加随机slogan

---

在 Fluid 主题配置文件修改 `index/slogan` 配置的 `url` 和 `keys`

```yaml
# 首页副标题的独立设置
# Independent config of home page subtitle
slogan:
enable: true

# 为空则按 hexo config.subtitle 显示
# If empty, text based on `subtitle` in hexo config
text: "要走起来，你才知道方向。"

# 通过 API 接口作为首页副标题的内容，必须返回的是 JSON 格式，如果请求失败则按 text 字段显示，该功能必须先开启 typing 打字机功能
# Subtitle of the homepage through the API, must be returned a JSON. If the request fails, it will be displayed in `text` value. This feature must first enable the typing animation
api:
enable: true

# 请求地址
# Request url
url: "https://v1.hitokoto.cn/"

# 请求方法
# Request method
# Available: GET | POST | PUT
method: "GET"

# 请求头
# Request headers
headers: {}

# 从请求结果获取字符串的取值字段，最终必须是一个字符串，例如返回结果为 {"data": {"author": "fluid", "content": "An elegant theme"}}, 则取值字段为 ['data', 'content']；如果返回是列表则自动选择第一项
# The value field of the string obtained from the response. For example, the response content is {"data": {"author": "fluid", "content": "An elegant theme"}}, the expected `keys: ['data','content']`; if the return is a list, the first item is automatically selected
keys: ['hitokoto']
```



#### [√] 在footer添加slogan

---

（1）修改主题配置文件

嵌入代码为：

```xml
 <div class="statistics">
 	<a href="https://developer.hitokoto.cn/" id="hitokoto_text"><span style="color: #DDD;"  id="hitokoto"></span></a>
<script src="https://v1.hitokoto.cn/?encode=js&select=%23hitokoto" defer></script>
 </div>
```

在主题配置文件中，`footer`处添加`content2`：

```yaml
#---------------------------
# 页脚
# Footer
#---------------------------
footer:
  # 页脚第一行文字的 HTML，建议保留 Fluid 的链接，用于向更多人推广本主题
  # HTML of the first line of the footer, it is recommended to keep the Fluid link to promote this theme to more people
  content: '
    <a href="https://hexo.io" target="_blank" rel="nofollow noopener"><span>Hexo</span></a>
    <i class="iconfont icon-love"></i>
    <a href="https://github.com/fluid-dev/hexo-theme-fluid" target="_blank" rel="nofollow noopener"><span>Fluid</span></a>
  '
  content2: ' <div class="statistics">
 	<a href="https://developer.hitokoto.cn/" id="hitokoto_text"><span style="color: #DDD;"  id="hitokoto"></span></a>
<script src="https://v1.hitokoto.cn/?encode=js&select=%23hitokoto" defer></script>
 </div>'

```

（2）修改`footer.ejs`文件（或者也可以在`bolg/themes/fluid/layout/_partials/footer/statistics.ejs`中添加这个代码）

要添加的代码为：

```ejs
  <% if (theme.footer.content2) { %>
    <div class="footer-content">
      <%- theme.footer.content2 %>
    </div>
  <% } %>
```

在`bolg/themes/fluid/layout/_partials/footer.ejs`中添加上述自定义的div块,

同时统一此footer处的字体大小：

```ejs
<div class="footer-inner" style="font-size: 0.85rem;">
```

如图所示：

<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212222217085.png" alt="image-20221222221711339" style="zoom:67%;" />

## [√] 效果图展示

---

![image-20221222222210016](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212222222688.png)

![image-20221222221903010](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212222222689.png)



## [√] 参考资料

---

> https://www.zywvvd.com/notes/hexo/theme/fluid/fluid-yiyan/yiyan/

