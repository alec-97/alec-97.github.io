---
title: 5 - fluid无需代码改动的基本+美化设置
---



## [] 主题配置文件

---



#### [√] 修改浏览器标签的图标

---

（1）在`blog/themes/fluid/source/img/`中，添加自己喜欢的图片，并命名为`photo.png`

![image-20221223112505185](D:\坚果云\Alec - backup files\typora pictures\image-20221223112505185.png)

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

![image-20221223112704543](D:\坚果云\Alec - backup files\typora pictures\image-20221223112704543.png)

#### [] 副标题美化

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









