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
