---
title: 6 - fluid主题无侵入式方式添加页面焦点监控文字
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 3644508848
---

## [√] 步骤

---

#### [√] 添加自定义monitortext.ejs文件

---

在`blog\source\_inject\`文件夹中，新增文件`monitortext.ejs`，用于向页面中添加焦点监控代码，其内容为：

```ejs
<% if(theme.fun_features.monitortext.enable) { %>
	<script type="text/javascript">
	  /*窗口监视*/
	  var originalTitle = document.title;
	  window.onblur = function(){document.title = "<%- theme.fun_features.monitortext.text %>"};
	  window.onfocus = function(){document.title = originalTitle};
	</script>
  <% } %>
```

#### [√] 注入上述代码

---

在文件`blog\scripts\page.js`中，添加以下代码：

```js
// 添加页面焦点监控文字
injects.bodyBegin.file('monitortext', "source/_inject/monitortext.ejs");
```

快照为：

```js
hexo.extend.filter.register('theme_inject', function(injects) {
    injects.bodyBegin.file('default', "source/_inject/bodyBegin.ejs");
    injects.header.file('video-banner', 'source/_inject/header.ejs', { key: 'value' }, -1);
    // 添加页面焦点监控文字(here)
    injects.bodyBegin.file('monitortext', "source/_inject/monitortext.ejs");
  });
```





#### [√] 编辑主题配置文件

---

- 编辑fluid 主题配置文件，在 `fun_features` 项下添加

```yaml
monitortext:
  enable: true
  text: 失去焦点显示该文字
```

<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212241646975.png" alt="image-20221224113744839" style="zoom: 80%;" />