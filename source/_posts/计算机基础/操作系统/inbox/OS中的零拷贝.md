---
title: OS中的零拷贝
date: '2022年11月21日14:39:48'
tags:
  - 操作系统基础知识
categories:
  - 计算机基础
  - 操作系统
  - inbox
abbrlink: 441628142
---

==传统的I/O的方式：==

数据的读取和写入需要从内核空间和用户空间之间来回的复制，而内核空间的数据则是通过操作系统的I/O从磁盘读取。这个过程需要发生多次的上下文切换和数据拷贝。<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202211211449759.png" alt="image-20221121144602045" style="zoom:50%;" /> 

---

==传统I/O方式的弊端：==

发生多次的上下文切换和内存拷贝，导致IO性能低。

4次上下文切换和4次数据拷贝

---

==解决方案：零拷贝技术==

零拷贝主要有两种实现方案，分别是：

-   mmap + write
-   sendfile

---

==零拷贝的方式1：mmap + write：==

mmap()系统调用函数，会直接将内核空间中数据的映射到用户空间，二者通过相同的内存地址访问同一份数据，因此也就不需要数据拷贝。（类似于共享内存）

4次上下文切换和3次数据拷贝

<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202211211455511.png" alt="image-20221121145454211" style="zoom:50%;" /> 

---

==零拷贝的方式2：sendfile：==

Linux 内核版本 2.1 中，提供了一个专门发送文件的系统调用，为 sendfile() ，这个系统调用可以直接将从硬盘 IO 进来的数据在内核空间直接发送到 socket 缓冲区，省去了先拷贝到用户空间这一步。

因此这一个命令替代了原来的 read() 和 write() 这两个命令。

因此就只有 2 次上下⽂切换，和 3 次数据拷⻉。

<img src="https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202211211502097.png" alt="image-20221121150146299" style="zoom:50%;" /> 

---

==该技术的应用：==

很多开源项目如Kafka、RocketMQ都采用了零拷贝技术来提升IO效率。



