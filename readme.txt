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







git clone 
git branch
git checkout mr98
git branch
git push origin mr98 #推送到远程库对应的远程分支

git add shop_quality/shop_ban/ban_sale   #提交到暂存区
git commit -m "zhushi"    #提交到分支
git push origin mr98 #推送分支

get merge mr98       #合并分支到master
git branch -d mr98   #删除分支





pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

nvcc —version
!python --version
import torch
print(torch.__version__)

cuda 11.7 镜像
pip3 install torch torchvision torchaudio
pip install torch_geometric
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html

https://anonymous.4open.science/r/Joint-cluster-loss-01C7/single_label/run.sh


pip list > o
vi o
awk 'NR==FNR{a[$1]} NR!=FNR&&($1 in a)' all_recall_shop_id.txt all_recall_shop_id.txt > merge
!pip3 list | grep euler
MiaoRui19981202

Python 3.9
Cuda 11.8
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip3 install tqdm
pip3 install texttable
pip3 install ogb
pip3 install down
pip3 install munch
pip3 install rdkit
pip3 install -U openmim
mim install mmcv
https://github.com/open-mmlab/mmsegmentation/issues/1327

Python3执行命令
