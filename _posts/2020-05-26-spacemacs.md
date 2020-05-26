---
title: Emacs个人使用指北
author: Kaeruchan
date: 2020-05-26 14:30:00 +0900
categories: [Blogging, IT, Emacs]
tags: [Spacemacs]
---


## 前言
----

从初代目Vim到Emacs到现在过去已经三年多了，
我的主力开发环境仍然是iTerm2+Tmux+Emacs。
因为作为科研人员，一直在使用Emacs下的一个强力的扩展包
[AUCTEX](http://gnu.org/software/auctex/)。
虽然途中试过VSCode和Atom，但是效果总是一言难尽。
但是Emacs那复杂的配包和自定义总是让人望而却步。

在Google里找了很多资料，还找到几份不错的Emacs配置，
在加上自己的少许修改，然后其实就可以用来了，
但是包的更新始终是一个问题。

后来碰到了Spacemacs，
其实它并不是一个Emacs的单独发行版本，
它也是一个基于Emacs开发的一个扩展包。
按照官方介绍，Spacemacs是一个由社区驱动的Emacs衍生版本，
正如官网上的这句话：
`The best editor is neither Emacs nor Vim, it's Emacs and Vim!`

## 配置Spacemacs
----

稍早也提过，Spacemacs并不是一个Emacs的独立开发版本，
只是一个扩展包，所以Spacemacs的安装也比较简单。
这里只是简单叙述一下Mac OS版本的安装方法：
- 安装一个Emacs。
- 复制一个官方的Spacemacs的配置文件。

### 安装Emacs

首先基于Mac OS的开发，是需要Xcode的支持的。
首先先从App Store上搜索Xcode并进行下载。（4.43GB十分良心）

下载完毕之后再启动Xcode，等待配置。

等这些全部完成之后，启动iTerm2后输入：
```bash
$ sudo xcode-select --install
```


接下来安装Homebrew，打开iTerms2后输入：

```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```
之后再检查Homebrew的安装是否正确即可：

```bash
$ brew doctor
```

之后便可以选择安装Emacs：

```bash
$ brew cask install emacs
```
等运行结束即可。

### 安装Spacemacs

相对于之前的配置，Spacemacs的配置就比较简单。
打开iTerms2输入：
```bash
$ git clone https://github.com/syl20bnr/spacemacs ~/.emacs.d
```
这样的话，再打开Emacs，那就是Spacemacs的模样啦！

### 自定义Spacemacs

这部分说起来比较复杂，可以采用我这部分的懒人包。
打开iTerm2后，直接复制粘贴下面这段代码即可：

```bash
$ git clone https://github.com/evacchi/tabbar-layer ~/.emacs.d/private/tabbar # copy tabber to local
$ rm -rf ~/.spacemacs # delete origin spacemacs file
$ git clone https://github.com/kaeruchan/dotspacemacs ~/dotspacemacs # copy file to home
$ mv ~/dotspacemacs/.spacemacs ~/.spacemacs # move file
$ rm -rf ~/dotspacemacs #remove directory

```

然后重启Emacs即可。

## 小结

作为一个长期使用Emacs的用户，的确这个软件还是需要更多的发掘的。
不过作为一个科研用户，
C-layer，
MATLAB-layer，
Python-layer
和TeX的支持
（尤其是AUCTEX）
已经让我觉得可以长期使用。
总之相比于其他的各类编辑器来，值得尝试。
