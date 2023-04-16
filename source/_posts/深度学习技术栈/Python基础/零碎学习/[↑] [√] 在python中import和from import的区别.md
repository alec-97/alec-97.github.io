---
title: 在python中import和from import的区别
tags:
  - python
  - 随手记
categories:
  - 深度学习技术栈
  - Python基础
  - 零碎学习
abbrlink: 1137730746
date: 2023-04-02 23:37:20
index_img:
---

在Python中，`import`和`from import`语句用于导入模块或模块中的特定函数或变量。它们之间的区别如下：

## import module

`import`语句用于导入整个模块，然后您可以使用模块中的所有函数和变量。例如，以下代码将导入整个`math`模块：

```python
import math

```

然后，您可以使用`math`模块中的所有函数和变量，例如：

```
x = math.sqrt(16)
print(x)
```

输出:

```
4.0
```

## from module import function



`from import`语句用于导入模块中的特定函数或变量。例如，以下代码将从`math`模块中导入`sqrt()`函数：

```
from math import sqrt
```

然后，您可以直接使用`sqrt()`函数，而不需要使用模块名称。

```python
x = sqrt(16)
print(x)
```

输出：

```python
4.0
```

如果您需要导入多个函数或变量，则可以使用逗号分隔它们，例如：

```python
from math import sqrt, pi
```

然后，您可以直接使用`sqrt()`和`pi`变量，例如：

```python
x = sqrt(16)
y = 2 * pi
print(x, y)

```

输出：

```python
4.0 6.283185307179586
```

请注意，`from import`语句可能会引起名称冲突，因为导入的函数或变量与您的代码中定义的名称相同。因此，最好只导入需要的函数或变量，并避免使用`*`通配符导入整个模块。











