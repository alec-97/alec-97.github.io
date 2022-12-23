---
title: 1 - fluid主题无侵入式方式添加视频背景
categories:
  - 工具笔记
  - hexo
  - 主题
  - fluid
abbrlink: 4083129976
---
---
1 - fluid主题无侵入式方式添加视频背景
---

> 参考：[Fluid -20- 使用 Fluid 注入功能实现背景视频（√）](https://www.zywvvd.com/notes/hexo/theme/fluid/fluid-inject/fluid-inject/)

> 通过代码注入的方式修改主题，可以实现无侵入式的修改。hexo中开发者开发的各种主题，会不断的优化改进、迭代新的版本。侵入式的方式在`hexo博客目录/themes/fluid/`文件夹中修改源代码，当主题升级之后，可能就会出现兼容性问题。因此推荐使用代码注入的方式实现无侵入式的修改。

## 1 - 代码注入介绍



### 1.1 - 概述

- 代码注入是在项目之外将需要修改的代码动态插入到项目中的技术手段
- 直接修改源码是完全可以达到目的的，但是源码修改会破坏仓库的代码完整性，问题主要出现在需要对仓库进行更新的时候
- 修改过的仓库很容易在更新时引入冲突，那时候很可能需要面对自己都不记得为什么改的代码和完全不懂的项目代码做出取舍，实在是很危险、痛苦而且不优雅的
- 也就是说，我们又要调整项目代码功能，又要保持项目足够“干净”，以便享受将来的更新，此时代码注入的价值便显现出来了



### 1.2 - hexo代码注入

- [Hexo 注入器](https://hexo.io/zh-cn/api/injector.html) 是 Hexo 5 版本自身加入的一项新功能，所以在所有 Hexo 主题都是支持这个功能的
- 注入器可以将 HTML 片段注入生成页面的 `head` 和 `body` 节点中
- 编写注入代码，需要在博客的根目录下创建 `scripts` 文件夹，然后在里面任意命名创建一个 js 文件即可。



#### 1.2.1 - 实践示例 + 讲解

（1）例如创建一个 `/blog/scripts/example.js`，内容为：

```js
hexo.extend.injector.register('body_end', '<script src="/jquery.js"></script>', 'default');
```

上述代码会在生成的页面 `body` 中注入加载 `jquery.js` 的代码

（2）参数

- `register` 函数可接受三个参数，第一个参数是代码片段注入的位置，接受以下值：

|    参数    |   含义   |
| :--------: | :------: |
| head_begin | head开头 |
|  head_end  | head结尾 |
| body_begin | body开头 |
|  body_end  | body结尾 |

- 第二个参数是注入的片段，可以是字符串，也可以是一个返回值为字符串的函数。
- 第三个参数是注入的页面类型，接受以下值：

|   参数   |                         含义                         |
| :------: | :--------------------------------------------------: |
| default  |               注入到每个页面（默认值）               |
|   home   |     只注入到主页（`is_home()` 为 `true` 的页面）     |
|   post   |   只注入到文章页面（`is_post()` 为 `true` 的页面）   |
|   page   |   只注入到独立页面（`is_page()` 为 `true` 的页面）   |
| archive  | 只注入到归档页面（`is_archive()` 为 `true` 的页面）  |
| category | 只注入到分类页面（`is_category()` 为 `true` 的页面） |
|   tag    |   只注入到标签页面（`is_tag()` 为 `true` 的页面）    |

> 或是其他自定义 layout 名称，例如在Fluid 主题中 `about` 对应关于页、`links` 对应友联页



### 1.3 - fluid代码注入

- Fluid 主题也提供了一套注入代码功能，相较于 Hexo 注入功能更细致更丰富，并且支持注入 `ejs` 代码。
- 如果你想充分修改主题，又不想直接修改源码影响日后更新，本主题提供了代码注入功能，可以将代码无侵入式加入到主题里。
- 你可以直接注入 HTML 片段，不过建议你了解一下 [EJS 模板引擎](https://ejs.bootcss.com/)，这样你就可以像主题里的 `ejs` 文件一样编写自己的组件再注入进去。
- 进入博客目录下 `scripts` 文件夹（如不存在则创建），在里面创建任意名称的 js 文件，在文件中写入如下内容：

```javascript
hexo.extend.filter.register('theme_inject', function(injects) {
  injects.header.file('default', 'source/_inject/test1.ejs', { key: 'value' }, -1);
  injects.footer.raw('default', '<script async src="https://xxxxxx" crossorigin="anonymous"></script>');
});
```

- `header` 和 `footer` 是注入点的名称，表示代码注入到页面的什么位置；
- `file` 方法表示注入的是文件，第一个参数下面介绍，第二个参数则是文件的路径，第三个参数是传入文件的参数（可省略），第四个参数是顺序（可省略）；
- `raw` 方法表示注入的是原生代码，第一个参数下面介绍，第二个参数则是一句原生的 HTML 语句；
- `default` 表示注入的键名，可以使用任意键名，同一个注入点下的相同键名会使注入的内容覆盖，而不同键名则会让内容依次排列（默认按执行先后顺序，可通过 `file` 第四个参数指定），这里 default 为主题默认键名，通常会替换掉主题默认的组件；

- 主题目前提供的注入点如下：

|    注入点名称     |                  注入范围                  | 存在 `default` 键 |
| :---------------: | :----------------------------------------: | :---------------: |
|       head        |            `head` 标签中的结尾             |        无         |
|      header       |          `header` 标签中所有内容           |        有         |
|     bodyBegin     |            `body` 标签中的开始             |        无         |
|      bodyEnd      |            `body` 标签中的结尾             |        无         |
|      footer       |          `footer` 标签中所有内容           |        有         |
|    postMetaTop    |    文章页 `header` 标签中 meta 部分内容    |        有         |
|  postMetaBottom   |          文章页底部`meta`部分内容          |        有         |
| postMarkdownBegin | `<div class="markdown-body">` 标签中的开始 |        无         |
|  postMarkdownEnd  | `<div class="markdown-body">` 标签中的结尾 |        无         |
|     postLeft      |               文章页左侧边栏               |        有         |
|     postRight     |               文章页右侧边栏               |        有         |
|   postCopyright   |               文章页版权信息               |        有         |
|     postRight     |               文章页右侧边栏               |        无         |
|   postComments    |                 文章页评论                 |        有         |
|   pageComments    |                自定义页评论                |        有         |
|   linksComments   |                 友链页评论                 |        有         |



## 2 - 视频背景注入实现

### 2.1 - 流程

- 创建注入配置文件
- 创建注入代码文件
- 创建背景图片+背景视频url json文件
- 修改主题配置文件

### 2.2 - 实现

（1）创建配置注入路径及文件`blog/scripts/page.js`，内容为

```javascript
hexo.extend.filter.register('theme_inject', function(injects) {
  //injects.header.file('default', 'source/_inject/test1.ejs', { key: 'value' }, -1);
  injects.bodyBegin.file('default', "source/_inject/bodyBegin.ejs");
  injects.header.file('video-banner', 'source/_inject/header.ejs', { key: 'value' }, -1);
});
```

（2）创建代码注入路径及文件

- 路径为
    - `blog/source/_insert/header.ejs`
    - `blog/source/_insert/bodyBegin.ejs`

- 内容为：
    - `header.ejs`

```ejs
<%
var banner_video = theme.index.banner_video
var banner_img = page.banner_img || theme.index.banner_img
var banner_img_height = page.banner_img_height || theme.index.banner_img_height
var banner_mask_alpha = page.banner_mask_alpha || theme.index.banner_mask_alpha
%>
	<script type="text/javascript" src="/vvd_js/jquery.js"></script>

	<div class="banner" id='banner' >

		<div class="full-bg-img" >

			<% if(banner_video){ %>
				<script>
					var ua = navigator.userAgent;
					var ipad = ua.match(/(iPad).*OS\s([\d_]+)/),
						isIphone = !ipad && ua.match(/(iPhone\sOS)\s([\d_]+)/),
						isAndroid = ua.match(/(Android)\s+([\d.]+)/),
						isMobile = isIphone || isAndroid;

					function set_video_attr(id){

						var height = document.body.clientHeight
						var width = document.body.clientWidth
						var video_item = document.getElementById(id);

						if (height / width < 0.56){
							video_item.setAttribute('width', '100%');
							video_item.setAttribute('height', 'auto');
						} else {
							video_item.setAttribute('height', '100%');
							video_item.setAttribute('width', 'auto');
						}
					}

					$.getJSON('/vvd_js/video_url.json', function(data){
						if (true){
							var video_list_length = data.length
							var seed = Math.random()
							index = Math.floor(seed * video_list_length)
							
							video_url = data[index][0]
							pre_show_image_url = data[index][1]
							
							banner_obj = document.getElementById("banner")
							banner_obj.style.cssText = "background: url('" + pre_show_image_url + "') no-repeat; background-size: cover;"

							vvd_banner_obj = document.getElementById("vvd_banner_img")

							vvd_banner_content = "<img id='banner_img_item' src='" + pre_show_image_url + "' style='height: 100%; position: fixed; z-index: -999'>"
							vvd_banner_obj.innerHTML = vvd_banner_content
							set_video_attr('banner_img_item')

							if (!isMobile) {
								video_html_res = "<video id='video_item' style='position: fixed; z-index: -888;'  muted='muted' src=" + video_url + " autoplay='autoplay' loop='loop'></video>"
								document.getElementById("banner_video_insert").innerHTML = video_html_res;
								set_video_attr('video_item')
							}
						}
					});

					if (!isMobile){
						window.onresize = function(){
							set_video_attr('video_item')
							}
						}
				</script>
			<% } %>
			</div>
		</div>
    </div>
```

- `bodyBegin.ejs`

```ejs
<div>
    <!--其中rgba中的a，指的是mask文件的alpha，透明度-->
	<div class='real_mask' style="
		background-color: rgba(0,0,0,0.3);
		width: 100%;
		height: 100%;
		position: fixed;
		z-index: -777;
	"></div>
	<div id="banner_video_insert">
	</div>	
	<div id='vvd_banner_img'>
	</div>
</div>
<div id="banner"></div>
```

（3）创建背景图片+背景视频url json文件

在`blog/source/vvd_js/`创建背景图片、视频地址json文件`video_url.json`

内容格式为：

```json
[
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220318.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220318.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220356.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220356.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220415.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220415.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220556.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220556.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220626.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220626.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220640.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220640.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220742.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210808220742.jpg"],
	["https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210813194902.mp4", "https://101.43.39.125/HexoFiles/vvd-dell-2021-win-10/20210813194902.jpg"]
]
```

同时在该文件夹下放置下载文件`jquery.js`

（4）修改主题配置文件`_config.fluid.yml`

- 覆盖默认 banner 图为纯透明的 png 图像，其中banner图片指的是背景图片。将banner换成透明的，方便将自己的动态视频嵌入。
    - 将所有的 `banner_img` 替换为 `https://101.43.39.125/HexoFiles/new/bg-trans.png`
- 添加 `index/banner_video` ，设置为 true

```yaml
#---------------------------
# 首页
# Home Page
#---------------------------
index:
  # 首页 Banner 头图，可以是相对路径或绝对路径，以下相同
  # Path of Banner image, can be a relative path or an absolute path, the same on other pages
  banner_img: https://101.43.39.125/HexoFiles/new/bg-trans.png
  
  # 首页 Banner 使用随机视频
  # true 开启  false 关闭
  banner_video: true

```

- 将所有 `banner_mask_alpha` 设置为 ，这个参数将透明度设置为0











