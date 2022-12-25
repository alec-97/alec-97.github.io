hexo.extend.filter.register('theme_inject', function(injects) {
    // 示例代码
    // injects.header.file('default', 'source/_inject/test1.ejs', { key: 'value' }, -1);
    // injects.footer.raw('default', '<script async src="https://xxxxxx" crossorigin="anonymous"></script>');


    injects.bodyBegin.file('default', "source/_inject/bodyBegin.ejs");
    injects.header.file('video-banner', 'source/_inject/header.ejs', { key: 'value' }, -1);
    // 添加页面焦点监控文字
    injects.bodyBegin.file('monitortext', "source/_inject/monitortext.ejs");
  });