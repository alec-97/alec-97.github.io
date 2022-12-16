---
title: SQL中DQL，DML，DDL，DCL，TCL的区别
date: '2022年11月22日14:37:05'
tags: []
categories:
  - 数据库
  - mysql
  - 笔记
  - SQL语法
---

## 笔记

SQL语言根据操作性质一共分为5大类

DQL：select（记录的操作）

DML：insert、delete、update（记录的操作）

DDL：create、alter、drop、truncate（表的操作）

DCL：grant、revoke（权限的操作）

TCL：commint、rollback、savepoint、set transaction（事务的操作）

### 1 - 数据查询语言，DQL（Query）

#### 1.1 - 作用

从数据库/表中查找字段的值

#### 1.2 - 主要命令 - select（查）

>   select 语法：

```sql
SELECT         select_list

[ INTO             new_table ]

FROM             table_source

[ WHERE        search_condition ]

[ GROUPBY   group_by_expression ]

[ HAVING        search_condition ]

[ ORDERBY    order_expression [ ASC | DESC ] ]
```

---

### 2 - 数据操纵语言，DML（manipulation）

#### 2.1 - 作用

对数据库的数据进行相关操作（对表中的记录进行操作）

#### 2.2 - 主要命令 - insert、delete、update（增删改）

```sql
#insert语法
INSERT      INTO      表名（列1，列2，...）     VALUES    （值1，值1，...）
```

```sql
#delete语法
 DELETE    FROM    表名      WHERE       列名 = 值
```

```sql
#update语法
 UPDATE    表名     SET     列名 = 新值       WHERE       列名称 = 某值
```

---

### 3 - 数据定义语言，DDL（definition）

#### 3.1 - 作用

主要是对表进行操作

#### 3.2 - 主要命令 - create、alter、drop、truncate（建立表、修改表（增加列、更改列、删除列）、删除表）

```sql
#create语法
CREATE       table         表名
```

```sql
#alter语法

ALTER       table       表名

ADD           (test_id    number)           --增加列

#----------------------------------------------------------------------------

ALTER       table       表名

MODIFY     (test_id    number)          --更改列

#----------------------------------------------------------------------------

ALTER       table       表名

DELETE     (test_id  )                         --删除列
```

```sql
#drop语法
DROP             table         表名
```

```sql
#truncate语法
TRUNCATE     table        表名
```

---

### 4 - 数据控制语言，DCL（Control）

#### 4.1 - 作用

DCL用来设置或更改数据库用户或角色权限

#### 4.2 - 主要命令 - grant、revoke（授权和撤销）

注意： 在默认状态下，只有sysadmin,dbcreator,db_owner或db_securityadmin等人员才有权力执行DCL

---

### 5 - 事务控制语言，TCL（Transaction Control）

#### 5.1 - 作用

用于数据库事务的相关操作

#### 5.2 - 主要命令 - commit、rollback、savepoint、set transaction（提交、回滚、设置保存点、设置事务选项）

## 例题

下列四组SQL命令，全部属于数据定义语句的命令是（A）。

```
CREATE，DROP，ALTER
```

```
CREATE，DROP，UPDATE
```

```
CREATE，DROP，GRANT
```

```
CREATE，DROP，SELECT
```

下列哪些属于DQL语句的命令(BCD)

```
A、INSERT
```

```
B、WHERE
```

```
C、FROM
```

```
D、SELECT
```

