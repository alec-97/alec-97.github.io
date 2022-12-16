---
title: sleep、yield、wait、join的区别
date: 2022年11月22日17:05:15
categories: [Java技术栈, Java多线程]
---

>   参考文章：
>
>   [sleep、yield、wait、join的区别(阿里) - 博客园 - 柴飞飞（√）]()

## 零碎点

只有runnable到running时才会`占用cpu时间片`，其他都会`出让cpu时间片`。



---

**线程状态转换图**

![image-20221122172926319](https://cdn.jsdelivr.net/gh/Alec-97/alec-s-images-cloud/img/202211221729611.png) 







## sleep 和 wait 的辨析

### 区别

线程的资源有不少，但应该包含`CPU资源`和`锁资源`这两类。

-   sleep(long mills)：让出CPU资源，但是不会释放锁资源
-   wait()：让出CPU资源和锁资源

---

wait用于锁机制，sleep不是，这就是为啥sleep不释放锁，wait释放锁的原因，sleep是线程的方法，跟锁没半毛钱关系，wait，notify,notifyall 都是Object对象的方法，是一起使用的，用于锁机制







## sleep 和 yield 的辨析

### 相同点

Thread.sleep(long) 和 Thread.yield() 都是 Thread 类的静态方法，在调用的时候都是 Thread.sleep(long) / Thread.yield() 的方式进行调用

而 join() 是由线程对象来调用





## wait





Object类的方法(notify()、notifyAll()  也是Object对象)，必须放在循环体和同步代码块中，执行该方法的线程会释放CPU资源和锁资源。然后该线程进入线程等待池中等待被再次唤醒(notify随机唤醒，notifyAll全部唤醒，线程结束自动唤醒)。线程在等待池中被唤醒之后，进入锁池，重新竞争获取同步锁。

---

wait() 和 notify()、notifyAll() 这三个方法都是 java.lang.Object 的方法。

---

都必须在 Synchronized 语句块内调用：它们都是用于协调多个线程对共享数据的存取，所以必须在Synchronized语句块内使用这三个方法。

---

前面说过Synchronized这个关键字用于保护共享数据，阻止其他线程对共享数据的存取。但是这样程序的流程就很不灵活了，如何才能在当前线程还没退出Synchronized数据块时让其他线程也有机会访问共享数据呢？此时就用这三个方法来灵活控制。 

-   （1）wait()方法使当前线程暂停执行并释放对象锁标志，让其他线程可以进入Synchronized数据块，当前线程被放入对象等待池中。
-   （2）当调用 notify()方法后，将从对象的等待池中移走一个任意的线程并放到锁标志等待池中，只有锁标志等待池中的线程能够获取锁标志；如果锁标志等待池中没有线程，则notify()不起作用。 
-   （3）notifyAll()则从对象等待池中移走所有等待那个对象的线程并放到锁标志等待池中。 

---

在java中，Thread类线程执行完run()方法后，一定会自动执行notifyAll()方法





## sleep

Thread类的方法，必须带一个时间参数。会让当前线程休眠进入阻塞状态并释放CPU资源，但是不会释放锁资源。



提供其他线程运行的机会且不考虑优先级



如果有同步锁则sleep不会释放锁即其他线程无法获得同步锁



可通过调用interrupt()方法来唤醒休眠线程。



阿里面试题 Sleep释放CPU，wait 也会释放cpu，因为cpu资源太宝贵了，只有在线程running的时候，才会获取cpu片段。



## yield

让出CPU调度，Thread类的方法，类似sleep只是不能由用户指定暂停多长时间 



并且yield()方法只能让同优先级的线程有执行的机会，优先级不同的线程，无法获得运行机会。



yield()只是使当前线程重新回到可执行状态，所以执行yield()的线程有可能在进入到可执行状态后马上又被执行。



调用yield方法只是一个建议，告诉线程调度器我的工作已经做的差不多了，可以让别的相同优先级的线程使用CPU了，没有任何机制保证采纳。



## join

一种特殊的wait，当前运行线程 A 调用另一个线程 B 的 join 方法，然后当前线程 A 进入阻塞状态直到另一个线程 B 运行结束，然后 A 再继续运行。 注意该方法也需要捕捉异常。



join()方法就是通过wait()方法实现的。



形象的讲，就是在线程 A 运行的过程中，执行了 `B.join()`，那么 A 就知道 B 加入进来了，那么 A 就让 B 先执行，B 执行完之后，A 再继续执行。相当于 A 在运行过程中，让 B 插了个队，



代码示例：

```java
class Solution {
    public static void main(String[] args) {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("子线程执行");
            }
        };
        Thread thread1 = new Thread(runnable);
        Thread thread2 = new Thread(runnable);
        thread1.start();
        thread2.start();
        try {
            //主线程开始等待子线程thread1，thread2
            thread1.join();
            thread2.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        //等待两个线程都执行完（不活动）了，才执行下行打印
        System.out.println("执行完毕");
    }

}
/*
子线程执行
子线程执行
执行完毕
*/
```

该示例中，就是主线程让 thread1 和 thread2 这两个线程插队。

