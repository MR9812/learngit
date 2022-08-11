#git
Git is free software.
it is a test.
#cat读取

#初始化仓库
git init

#提交内容
git add file
git commit -m "zhushi"

#查看状态
git status

#查看所做的修改
git diff fil

#版本回退
git log
git log --pretty=oneline
git reset --hard HEAD^ ^^ ^100
git reset --hard 1094a
git reflog

#learngit工作区，.git(ls -ah)版本区，
#git add 将修改放到暂存区，git commit 提交到分支
#提交后对工作区没有任何修改，git status干净的
#红色在工作区，绿色在暂存区
#只有在暂存区的才会被commit;

#撤回修改
#git checkout -- readme.txt

#删除文件
git rm test.txt           #彻底删除
git checkout -- test.txt  #误删，恢复

#连接远程库
git remote add origin git@github.com:MR9812/learngit.git
#git remote add origin git@server-name:path/repo-name.git
#origin 为远程库的名字，
git push -u origin master #第一次连接，推送master分支的全部内容
git push origin master #之后推送最新修改

#从远程库中clone
git clone git@ser_name/repo-name.git

#创建分支并操作
git branch           #查看当前分枝
git branch mr98      #创建新分支
git checkout mr98    #切换分支
git switch mr98      #也是切换分支，区分checkout的删除
git checkout -b mr98 #创建分枝并切换
get merge mr98       #合并分支到master
git branch -d mr98   #删除分支

#解决冲突
#当master和mr98都做出了修改，master不是mr98所基于的master，冲突。

#分支管理，一般不在master干活，会在另一个分支干活，master只负责上线
git merge --no--ff -m "zhushi" mr98 #合并分支时会留下注释，方便管理
git log #可以查看分支历史

#bug分支
git stash  #工作现场保存，修复bug，
git stash pop #回到工作现场

#删除未合并分支
git branch -D mr98

#多人协作
git push origin mr98 #推送到远程库对应的远程分支

#rebase
git rebase  #变为直线
