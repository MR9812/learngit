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

#
