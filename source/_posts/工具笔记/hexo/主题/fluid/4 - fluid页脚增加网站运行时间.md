---
title: 4 - fluid页脚增加网站运行时间
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 2108047675
---

Fluid 1.8.4 版本支持自定义页脚内容了，本文记录页脚添加网站运行时间的方法。

## [√] 步骤

---

#### [√] 添加js文件

---

（1）js文件内容：

```js
!(function() {
  /** 计时起始时间，自行修改 **/
  var start = new Date("2020/01/01 00:00:00");

  function update() {
    var now = new Date();
    now.setTime(now.getTime()+250);
    days = (now - start) / 1000 / 60 / 60 / 24;
    dnum = Math.floor(days);
    hours = (now - start) / 1000 / 60 / 60 - (24 * dnum);
    hnum = Math.floor(hours);
    if(String(hnum).length === 1 ){
      hnum = "0" + hnum;
    }
    minutes = (now - start) / 1000 /60 - (24 * 60 * dnum) - (60 * hnum);
    mnum = Math.floor(minutes);
    if(String(mnum).length === 1 ){
      mnum = "0" + mnum;
    }
    seconds = (now - start) / 1000 - (24 * 60 * 60 * dnum) - (60 * 60 * hnum) - (60 * mnum);
    snum = Math.round(seconds);
    if(String(snum).length === 1 ){
      snum = "0" + snum;
    }
    document.getElementById("timeDate").innerHTML = "本站安全运行&nbsp"+dnum+"&nbsp天";
    document.getElementById("times").innerHTML = hnum + "&nbsp小时&nbsp" + mnum + "&nbsp分&nbsp" + snum + "&nbsp秒";
  }

  update();
  setInterval(update, 1000);
})();
```

在调用该js代码之后，会执行每1秒循环调用`update()`这个函数，

在这个函数中，比如执行如下语句：

`document.getElementById("timeDate").innerHTML = "本站安全运行&nbsp"+dnum+"&nbsp天";`

通过执行这个语句，可以例如将页面内容`<span id="timeDate">载入天数...</span>`替换，替换为计算出来的天数内容。



（2）将`var start = new Date("2020/01/01 00:00:00");`一行修改为自己的网站开始时间。



（3）js文件存放位置

需要说明的是，如果将这个js 文件直接放在 hexo 目录的source 文件夹中，会报错无法渲染站点，此处有两种解决方案

方案1：

- 在主题 themes -> fluid -> source -> js 文件夹中添加文件 duration.js

方案2：

- 在hexo站点 hexo -> source -> alec_js 文件夹中添加文件 duration.js

- 同时在站点配置文件 hexo/_config.yml 中为 alec_js 文件夹添加跳过渲染的选项：

```yaml
skip_render:
    - alec_js/**
```











#### [√] 修改主题配置文件

---

在主题配置中的 footer: content 添加div：

```yaml
    <div style="font-size: 0.85rem">
      <span id="timeDate">载入天数...</span>
      <span id="times">载入时分秒...</span>
      <script src="/vvd_js/duration.js"></script>
    </div>
```

具体为：

```yaml
footer:
  content: '
    <a href="https://hexo.io" target="_blank" rel="nofollow noopener"><span>Hexo</span></a>
    <i class="iconfont icon-love"></i>
    <a href="https://github.com/fluid-dev/hexo-theme-fluid" target="_blank" rel="nofollow noopener"><span>Fluid</span></a>
    <div style="font-size: 0.85rem">
      <span id="timeDate">载入天数...</span>
      <span id="times">载入时分秒...</span>
      <script src="/vvd_js/duration.js"></script>
    </div>
  '
```







## [√] 效果展示

---

![image-20221223013654140](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202212230137981.png)





## [√] 参考

---

https://www.zywvvd.com/notes/hexo/theme/fluid/fluid-run-how-long/fluid-run-how-long/（√）