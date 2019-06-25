## gitbook 使用（for Mac）

## 一. install gitbook

1.安装node （先安装brew）

```
brew install node
```

 2.利用node 安装gitbook

```bash
npm install gitbook-cli -g
```

3.新建GitBook项目

```bash
mkdir MyBook #新建目录
cd MyBook    #进入目录
gitbook init #初始化目录
```

4.编辑电子书内容

首先，GitBook使用`SUMMARY.md`文件组织整个内容的目录，如：

```markdown
# Summary

* [简介](README.md)
* [常见问题](Faq.md)
```

5.预览电子书

```bash
gitbook serve
```

`gitbook serve` 命令实际上会首先调用 `gitbook build`编译书籍，完成以后会打开一个 web 服务器，监听本地 4000 端口，在浏览器中输入`http://localhost:4000`，即可打开电子书。

6.发布电子书

当电子书内容制作好后，可以使用如下命令，生成html版本的电子书：

```bash
gitbook build
```

该命令会在当前文件夹中生成`_book`文件夹，用户可以将这个文件夹内容托管到网上，从而实现内容的发布。

### 二.发布到git pages

> pPages 是github类网站提供的免费的静态网页托管服务，既然GitBook能生成基于HTTML的静态电子书籍，那自然而然，我们就会有将GitBook静态页面发布到pages服务的需求。
>
> 除了能够将书籍发布到GitBook.com外，用户还可以将电子书发布到提供Pages服务的GitHub类站点上，如国内的oschina、coding.net等等。这样做有个好处，就是访问速度比gitbook.com快很多。
>
> 这个需求情景实际上是git的应用，即将书籍的源码存放到仓库的master分支，而将书籍的html版本存放到仓库的pages分支，

1.首先在github上 new repository，名字不重要。

2.在gitbook 的目录：

```bash
git init
```

使用文本编辑器，创建名为`.gitignore`的文件，通过`.gitignore`文件，本地仓库将忽略临时文件和`_book`文件夹，达到只保存书籍源码的目的。内容如下：

```bash
*~
_book
.DS_Store
```

现在可以将本地书籍源码添加到本地git仓库中了：

```bash
git add .
```

添加更新说明：

```bash
git commit -m '更新说明文字'
```

建立本地仓库与远端仓库的对应关系：

```bash
git remote add origin https://远程仓库地址.git
```

推送：

```bash
git push -u origin master
```

至此，就完成了将gitbook源码推送到远程仓库的任务，之后书籍内容修改后，执行如下操作即可：

```bash
git add .
git commit -m '更新说明文字'
git push -u origin master
```

3.使用pages服务展示gitbook书籍（<u>**重要**</u>）

> p接下来，需要在原有本地仓库新建一个分支，在这个分支中，只保留_book文件夹中的内容，然后将这些内容推送到远程仓库的pages分支，启用pages服务，最终达到免费发布电子书的目的。

新建分支,新分支一定要是gh-pages这个名字

```bash
git checkout --orphan gh-pages
```

删除不需要的文件，切换到pages分支后，我们需要将_books目录之外的文件都清理掉：

```bash
git rm --cached -r .
git clean -df
rm -rf *~
```

使用文本编辑器，创建名为.gitignore的文件，内容如下：

```bash
*~
_book
.DS_Store
```

复制_book文件夹到分支根目录：

```bash
cp -r _book/* .
```

后面就是推送到这个分支了：

```bash
git add .
git push -u origin gh-pages

```

### 三.写了一个shell 脚本，mac 终端直接执行：

```shell
git checkout master

git add .

git commit -m $1

git push -u origin master

git checkout gh-pages

cp -r _book/* .

git add .

git commit -m $1

git push -u origin gh-pages

git checkout master
```

