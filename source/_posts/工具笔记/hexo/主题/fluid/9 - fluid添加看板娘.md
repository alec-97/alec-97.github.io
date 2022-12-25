---
title: 9 - fluid添加看板娘
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 2499736958
---

（1）

下载 [张书樵大神的项目](https://github.com/stevenjoezhang/live2d-widget)，解压到本地博客目录的 `themes/fluid/source/alec_diy/` 下，修改文件夹名为 `live2d-widget`，修改项目中的 `autoload.js` 文件，如下将：

```js
const live2d_path = 'https://cdn.jsdelivr.net/gh/stevenjoezhang/live2d-widget/'
```

改为

```js
const live2d_path = '/alec_diy/live2d-widget/'
```

（2）在主题配置文件的`custom_js`和`custom_css`中加入：

```yaml
custom_js:
  # live2d的js文件（2）
  - /alec_diy/live2d-widget/autoload.js
  - //cdn.jsdelivr.net/npm/jquery/dist/jquery.min.js

custom_css:
  # live2d的css文件（1）
  - //cdn.jsdelivr.net/npm/font-awesome/css/font-awesome.min.css
```

也可以将上面两个依赖文件下载到本地然后再引入

（3）在主题配置文件中,新增如下内容：

```yaml
live2d: enable: true
```

（4）想修改看板娘大小、位置、格式、文本内容等，可查看并修改 `waifu-tips.js` 、 `waifu-tips.json` 和 `waifu.css`

（5）效果展示：

![image-20221225141923800](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212251419659.png)