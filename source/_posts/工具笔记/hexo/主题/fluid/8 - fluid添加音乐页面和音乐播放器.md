---
title: 8 - fluid添加音乐页面和音乐播放器
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 3564283204
---

## [√] 添加悬浮音乐播放器

---

在fluid的主题配置文件中，提供了自定义html的位置，因此直接在主题配置文件中添加html代码

```yaml
custom_html: '
 <!--音乐-->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.css">
  <script src="https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/meting@1.2/dist/Meting.min.js"></script>
  <div id="player" class="aplayer aplayer-withlist aplayer-fixed" data-id="3025663508" data-server="netease" data-type="playlist" data-order="random" data-fixed="true" data-listfolded="true" data-theme="#2D8CF0"></div>
'
```

## [√] 添加音乐页面

---

（1）在主题配置文件中的`nemu`处，添加音乐配置：

```yaml
- { key: "音乐", link: "/playlist/", icon: "iconfont icon-music" }
```

具体为：

```yaml
  menu:
    - { key: "home", link: "/", icon: "iconfont icon-home-fill" }
    - { key: "archive", link: "/archives/", icon: "iconfont icon-archive-fill" }
    - { key: "category", link: "/categories/", icon: "iconfont icon-category-fill" }
    - { key: "tag", link: "/tags/", icon: "iconfont icon-tags-fill" }
    - { key: "音乐", link: "/playlist/", icon: "iconfont icon-music" }
    - { key: "about", link: "/about/", icon: "iconfont icon-user-fill" }
```

（2）使用命令创建音乐界面，比如命名为playlist

```shel
hexo new page playlist
```

（3）打开网站根目录source\playlist\index.md根据[hexo-tag-aplayer](https://github.com/MoePlayer/hexo-tag-aplayer)文档书写即可

```markdown
示例：
{% meting "7729098320" "netease" "playlist" %}

{% meting "2305794885" "netease" "playlist" %}

{% meting "7724497259" "tencent" "playlist" "theme:#3F51B5" "mutex:true" "preload:auto" %}
```

## [√] 参考

---

https://cloud.tencent.com/developer/article/1953317（√）