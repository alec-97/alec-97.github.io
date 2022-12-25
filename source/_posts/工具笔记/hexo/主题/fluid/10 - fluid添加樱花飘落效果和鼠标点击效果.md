---
title: 10 - fluid添加樱花飘落效果和鼠标点击效果
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 703315339
---

## [√] 步骤

---

（1）创建路径，并在`blog_test\themes\fluid\source\alec_diy\mouse_click\`路径下载鼠标点击效果

![image-20221225145308788](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212251454939.png)

（2）修改主题配置文件，添加

```yaml
custom_js:
  #############
  #鼠标点击特效#
  #############
  # （1）鼠标移动星星特效
  - /alec_diy/mouse_click/star.js
  # （2）鼠标点击爱心特效
  # - /alec_diy/mouse_click/love.js
  # （3）鼠标点击文字特效
  # - /alec_diy/mouse_click/dianjichuzi.js
  
  #############
  #满屏飘落特效#
  #############  
  # （1）樱花飘落
  - //cdn.jsdelivr.net/gh/bynotes/texiao/source/js/yinghua.js
  
```

> tips:
>
> - 站点下的资源文件和主题下的资源文件同时在custom中的时候，有冲突，或者可能主题配置文件中的自定义js文件，无法搜索到站点下的source文件