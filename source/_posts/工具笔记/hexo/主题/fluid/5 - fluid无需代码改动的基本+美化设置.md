---
title: 5 - fluid无需代码改动的基本+美化设置
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 135347359
---



## [√] 主题配置文件

---



#### [√] 修改浏览器标签的图标

---

（1）在`blog/themes/fluid/source/img/`中，添加自己喜欢的图片，并命名为`photo.png`

![image-20221223112505185](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212241646455.png)

（2）修改主题配置文件

```yaml
#---------------------------
# 全局
# Global
#---------------------------

# 用于浏览器标签的图标
# Icon for browser tab
favicon: /img/photo.png

# 用于苹果设备的图标
# Icon for Apple touch
apple_touch_icon: /img/photo.png
```

（3）效果展示：

![image-20221223112704543](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212241646456.png)

#### [√] 副标题美化

---

修改打字机效果速度：80

指定页面使用打字机效果

```yaml
# 一些好玩的功能
# Some fun features
fun_features:
  # 为 subtitle 添加打字机效果
  # Typing animation for subtitle
  typing:
    enable: true

    # 打印速度，数字越大越慢
    # Typing speed, the larger the number, the slower
    typeSpeed: 80

    # 游标字符
    # Cursor character
    cursorChar: "_"

    # 是否循环播放效果
    # If true, loop animation
    loop: false

    # 在指定页面开启，不填则在所有页面开启
    # Enable in specified page, all pages by default
    # Options: home | post | tag | category | about | links | page | 404
    scope: [home]
```



#### [√] 为文章内容中的当前标题添加锚图标

---

（1）设置

```yaml
  # 为文章内容中的标题添加锚图标
  # Add an anchor icon to the title on the post page
  anchorjs:
    enable: true
    element: h1,h2,h3,h4,h5,h6
    # Options: left | right
    placement: left
    # Options: hover | always | touch
    visible: hover
    # Options: § | # | ❡
    icon: "->"

```



（2）效果展示

![image-20221223220959466](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212241646457.png)



#### [√] 加载进度条调整宽度和颜色

---

```yaml
  # 加载进度条
  # Progress bar when loading
  progressbar:
    enable: true
    height_px: 5
    color: "#29d"
    # See: https://github.com/rstacruz/nprogress
    options: { showSpinner: false, trickleSpeed: 100 }

```

效果展示

![image-20221223221218379](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212241646458.png)

