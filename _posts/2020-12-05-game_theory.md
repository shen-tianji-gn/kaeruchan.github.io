---
title: 浅谈博弈论（Game theory）
author: Kaeruchan
date: 2020-12-07 09:44:00 +0900
categories: [Security, Game theory]
tags: [Game theory]

---

# 前言

博弈论英文叫"game theory"，目前属于经济学一个分支。
收到世人瞩目的原因是因为这个单独的学科，适用于任何
明显有竞争对抗成分在里面的场景。
我第一次接触到这块也是因为参与了安全性问题的探讨而
学习到这部分知识。

这个理论专门研究多个个体之间对抗行为（竞争行为）。
所以某些叫法也有叫"对策论 or 赛局理论"。

---

# 博弈论的起源

在开始，我们先聊一下"博弈论"的发展史。

要开始这个话题，约翰·冯·诺伊曼（John von Neumann）
是个怎么样都绕不过去的人物。

！[约翰·冯·诺伊曼](https://imgur.com/nrwodGm.jpg)

此人是个**超级跨界牛人**，
就算用了那么夸张的称呼，
依然不足以体现这个人的牛逼之处——
他同时在"数学，物理学，经济学，计算机"等
多个领域作出了划时代的贡献，并且留下了一大堆以他命名的东西。
比如：程序员都应该听说过的"冯诺伊曼体系"，
数学领域的"冯诺伊曼代数、冯诺伊曼便历定理…"，
物理领域的"冯诺伊曼量子测量，冯诺伊曼熵，冯诺伊曼方程…"。
另外还有很多东西，
虽然不是以他的名字命名，也是他先发现的，
比如：量子力学的公理化表述，
希尔伯特第五问题，
连续几何（其空间维数不是整数），
蒙特卡洛仿真方法，
归并排序算法…

1944年，他与奥斯卡·摩根斯坦（Oskar Morgenstern）
合作发表了《博弈论与经济行为》
（英语"Theory of Games and Economic Behavior"），
一举奠定博弈论体系的基础，
所以他也被称作"博弈论之父"。

这个《博弈论与经济行为》一开始是以论文形式写成，
长达1200页，基本上是冯·诺曼依一个人的手笔。
有些朋友会误会——那为什么摩根斯坦能当第二作者啊？
这里面大致有2个原因：
其一，摩根斯坦本人非常看好"博弈领域的研究"，
他认为：该领域的研究可以为一切经济学理论建立正确的基础。
当他解释了大牛知乎，就一直劝说大牛写该领域的论文；
其二，当大牛完成上千页的论文之后，
摩根斯坦为这篇论文补了一个非常有煽动性的"绪论"，
是的这篇论文一发表就在数学界和经济学界产生轰动效果。
所以把摩根斯坦列为第二作者，也算说得过去。

另外，这本《博弈论与经济行为》的某些思想，
源自冯·诺伊曼在1928年发表的论文
《On the Theory of Parlor Games》。
因此有些学者认为1928年才是真正意义上的博弈论诞生之年。

---

# 博弈的类型

"博弈的类型"是博弈论的基本概念，先来聊这个。

---

## 合作博弈（cooperative game） VS 非合作博弈（non-cooperative game）

无论是"合作博弈"or"非合作博弈"，在博弈过程中都可能会出现"合作的"现象。差别在于——<br>
对于"合作博弈"，存在某种【外部约束力】，使得"背叛"的行为会受到这种外部约束力的惩罚。<br>
对于"非合作博弈"，【没有】上述这种"外部约束力"，对"背叛"的惩罚只能依靠博弈过程的其他参与者。
