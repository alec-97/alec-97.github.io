---
title: 3 - fluid主题自定义CSS样式：debug footer字体颜色问题
---

## [√] 相关知识

---

#### [√] div和span的区别是什么

---

在html页面布局时，我们经常会使用到div标签和span标签，那么div标签和span标签之间有什么区别？



**div**

- 占一行：
    - div标签是块级元素，每个div标签都会从新行开始显示，占据一行
- 嵌套：
    - div标签内可以添加其他的标签元素（行内元素、块级元素都行），比如：span标签，p标签，也可以是div标签
- 支持自定义CSS样式：
    - div标签可以通过css样式来设置自身的宽度（也可省略，当没有使用css自定义宽度时，div标签的宽度为其的容器的100%）、高度，且还可以设置标签之间的距离（外边距和内边距）



**span**

- 行内元素：
    - span标签是行内元素，会在一行显示；span标签元素会和其他标签元素会在一行显示（块级元素除外），不会另起一行显示
    - span标签内只能添加行内元素的标签或文本，span标签里只能容纳文本或者是其他的行内元素，不能容纳块级元素。
- 不支持CSS，只能在标签内定义格式：
    - span标签的宽度、高度都无法通过css样式设置，它的宽高受其本身内容（文字、图片）控制，随着内容的宽高改变而改变；span标签无法控制外边距和内边距，虽然可以设置左右的外边距和内边距，但上下的外边距和内边距无法设置。



总结，div标签可以单独的自定义CSS样式，且独占一行，支持嵌套，span只能在标签内通过如`<span style="color: #DDD;"  id="hitokoto"></span>`的方式改变样式，且为行内元素，不能嵌套。





#### [√] 无侵入式自定义CSS样式

---

hexo中经常需要修改网页中的样式，为了无侵入地修改CSS样式可以使用 Fluid 自定义 CSS样式的功能，本文记录使用方法。



方法为：

- 创建`blog/source/css/`路径，并在这个路径创建自定义名称如`alec_custom.css`文件
- 主题配置文件中加入该文件相对路径：

```yaml
custom_css:
  - /css/custom.css
  - //at.alicdn.com/t/font_1736178_ijqayz9ro8k.css
```

- 在该css文件中自定义css样式即可
- 具体实现步骤可以往下看





## [√] debug修改footer字体颜色步骤

---

在上一文中在页表`footer`添加了`一言`之后，出现了如下切换黑夜、日间模式时字体颜色不统一问题：

![image-20221222225634604](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137471.png)

![image-20221222225657666](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137472.png)

因此想要通过自定义CSS样式无侵入式的解决这个问题。

首先在网页页面，通过`ctrl + shift + c`快捷键或者`F12`快捷键，定位到想要修改样式的地方，

![image-20221223010222198](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137473.png)

![image-20221223010315398](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137474.png)

如图所示，发现 footer 此处的总的div是`class="footer-inner"`，因此可以在自定义的CSS文件中，统一修改颜色：

![image-20221223010451794](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137475.png)

在修改了颜色之后，发现上述问题解决，但是出现了小bug，如下图所示：

![image-20221223010708930](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230107284.png)

即仍有两个地方颜色未变，查看源代码：

```html
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
```

发现`hexo`和`fluid`文字是通过`span`定义的，查阅资料知道，`span`标签，不支持自定义CSS文件修改样式。因此在此处直接在`span`标签上修改样式，改动为为：

```html
<span>Hexo</span>
<span>Fluid</span>
# ↓
<span style="color: #DDD;">Hexo</span>
<span style="color: #DDD;">Fluid</span>
```

![image-20221223011414766](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137476.png)



最终，问题得以解决！





## [√] 参考

---

https://m.php.cn/article/413753.html（√）

https://www.zywvvd.com/notes/hexo/theme/fluid/fluid-custom-css/fluid-custom-css/（√）