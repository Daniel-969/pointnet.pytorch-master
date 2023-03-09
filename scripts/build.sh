# 获取build.sh所在文件夹的绝对路径
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
echo $SCRIPTPATH

# 对../utils/render_balls_so.cpp进行编译，render_balls_so.cpp文件是用于可视化的C++代码
# -o参数用来指定生成程序的名字
# -shared参数表示编译动态库
# -O2用于优化编译文件
# -D_GLIBCXX_USE_CXX11_ABI用于区分有旧版(c++03规范)的libstdc++.so，
# 和新版(c++11规范)的libstdc++.so两个库，
# -D_GLIBCXX_USE_CXX11_ABI=0 链接旧版库，-D_GLIBCXX_USE_CXX11_ABI=1 链接新版库
g++ -std=c++11 $SCRIPTPATH/../utils/render_balls_so.cpp -o $SCRIPTPATH/../utils/render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
