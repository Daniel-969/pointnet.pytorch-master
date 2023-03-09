# 获取download.sh所在文件夹的绝对路径
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# 进入download.sh所在文件夹的上一层
cd $SCRIPTPATH/..
# 下载数据集压缩包、解压压缩包、删除压缩包
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
# 重新进入当前文件夹
cd -
