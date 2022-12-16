---
title: Java使用FileWriter类向文件写入内容
date: '2022年12月16日18:32:29'
tags:
  - JavaIO
categories:
  - Java技术栈
  - Java基础
  - 笔记
  - JavaIO
  - 0 - 知识点收集
---

```java
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class Main {
    //使用FileWriter向文本文件中写信息
    public static void main(String[] args) {
        String str = "Hello World";
        //1.创建流
        Writer fw = null;
        try {
            /*创建txt文件*/
            File file = new File("D:\\hello.txt");
            if (!file.exists()) {
                file.createNewFile();
            }
            fw = new FileWriter("D:\\hello.txt");//1
            //2.写入信息
            fw.write(str);
            // 3.刷新缓冲区，即写入内容
            fw.flush();
            if (fw != null) {
                // 4.关闭流,关闭缓冲流时，也会刷新一次缓冲区
                fw.close();

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

```

结果：

![image-20221114144244343](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202211141442388.png)

