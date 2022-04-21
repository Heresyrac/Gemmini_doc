

Gemmini
====================================

本文档将会介绍将深度学习模型移植到gemmini上，并使用spike模拟器运行模型完成推理的一般步骤。同时会介绍了一些操作过程中可能遇到的弯路以及规避方式。
流程主要涉及到三个项目，[chipyard](https://chipyard.readthedocs.io/en/stable/)，[gemmini](https://github.com/ucb-bar/gemmini), [onnxruntime_riscv](https://github.com/ucb-bar/onnxruntime-riscv)


从某框架（如Pytorch，TensorFlow等）移植模型大致需要的步骤包括：


1.完成chipyard，gemmini，onnxruntime-riscv以及相关工具链的安装

2.导出onnx模型：
        将待移植的模型利用从onnx支持的框架提供的转换工具，得到onnx模型 

3.量化onnx模型：
        用onnxruntime-riscv中提供的工具，将导出的onnx模型的数据类型从fp32转化为适合gemmini运行的int8，得到一个已量化的onnx模型 

4.编写一个与模型对应的Runner：
        onnxruntime-riscv已经提供了针对数种模型可用的runner（如imagenet，bert_mask等），但若需要移植的模型属于其他类型，则需要自行编写runner。运行器需要做的主要工作包括：创建ORT_session，并将各个参数（如模型路径，优化的等级，dataflow mode）传递给session；实现一个将input整形为onnx模型可以接收的input_tensor的算法；调用session.run；输出推理结果。

5.运行编译好的Runner：
        待推理的输入数据，已量化的onnx模型将会作为runner程序的输入，推理结果由运行器给出。模型与gemmini的交互，各层运算操作将会在执行session.run后自动执行，不需人工干预。

预计移植一个模型的主要工作量在于：

1. 编写runner部分，该部分ONNX提供的C++教程内容极少，不成体系。估计只能通过仿写onnxruntime—riscv中提供的运行器源码完成，编写运行器的工作会存在困难。

2. onnxruntime—riscv的自动执行并不足够可靠，非常容易遇到运行失败的情况，即使使用作者在项目中已发布的模型、已发布的运行器也会遇到。可以预计如果随意编写一个另外的runner执行其他任务，不能期待onnxruntime—riscv可以完美地自动执行，大概率会需要较大的调试成本。

3. 量化器，据onnxruntime-riscv作者在[文档](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/docs/bumping.md)中所说，由于onnx本身提供的量化器不能很好的支持int8类型，作者作了大量修改才能使其成功运行，但这个量化器并未经过大量测试，在大型的复杂网络上存在出错的可能。因此导出onnx模型后的量化工作也可能出现问题，并不能保证可以运行脚本后一键导出，这部分也可能需要进行大量调试或是修改量化程序。

4. 导出onnx模型，该部分也并不能保证任何模型都能转换成功，可能会遇到一定的问题，但这部分的工具是由深度学习框架官方给出的，文档较为完善，有经受过更多的测试，预计也会有一定的工作量。





运行环境
==========
Ubuntu 18.04


网络设置
==========

由于chipyard，gemmini，onnxruntime-riscv以及相关工具链的安装编译中需要使用的脚本，存在着如果中途因为网络问题下载失败而退出，后续即使重新运行也会导致无法正确安装的问题。

同时，经过测试发现脚本中的存在一些站点，存在依靠国内网络会出现必定下载失败（包括google-source站点），又或是通过HTTPS协议clone必定失败的情况。

因此建议在进行一切工作之前，首先需要：

1.设置好某种翻墙工具，如vpn，V2RAY代理均可。若使用代理需设置git的下载均通过代理完成。

2.启用SSH协议进行clone，介绍详见 [这里](https://docs.github.com/cn/authentication/connecting-to-github-with-ssh).

3.设置git用SSH协议代替HTTPS协议进行下载。

```shell
git config --global url.ssh://git@github.com/.insteadOf https://github.com/
```

4.若使用代理，在git设置中设定通过代理进行ssh连接，以避免ssh连接不稳定引发的安装失败。

```shell
#打开git config文件
vim ~/.gitconfig

#若代理协议为socks5，在文件内写入：
[core]
        gitProxy = /opt/bin/socks5proxywrapper
```

若协议为socks5，则创建socks5proxywrapper文件：

```shell
vim /opt/bin/socks5proxywrapper

#若代理ip为127.0.0.1 端口为10808，协议为socks5，则在文件内写入：

#!/bin/sh
/usr/bin/ncat --proxy 127.0.0.1:10808 --proxy-type socks5 "$@"

```

进行上述网络设置以尽量保证在运行chipyard，gemmini，onnxruntime的诸个安装脚本时能够一次通过。若中途某一步骤中出现了因网络原因而执行失败的情况，尽量选择删除当前文件夹内的内容，并从头再来。否则即使后续显示安装成功，实际使用时也可能出现各种未知错误。


安装chipyard
====================================

Chipyard dependencies
---------------------------

本部分用于安装chipyard所需的依赖项，其中所需的操作包括：

CentOS-based 平台的用户根据如下命令下载依赖项：
```shell
#!/bin/bash

set -ex

sudo yum groupinstall -y "Development tools"
sudo yum install -y gmp-devel mpfr-devel libmpc-devel zlib-devel vim git java java-devel

# Install SBT https://www.scala-sbt.org/release/docs/Installing-sbt-on-Linux.html#Red+Hat+Enterprise+Linux+and+other+RPM-based+distributions
# sudo rm -f /etc/yum.repos.d/bintray-rpm.repo
# Use rm above if sbt installed from bintray before.
curl -L https://www.scala-sbt.org/sbt-rpm.repo > sbt-rpm.repo
sudo mv sbt-rpm.repo /etc/yum.repos.d/

sudo yum install -y sbt texinfo gengetopt
sudo yum install -y expat-devel libusb1-devel ncurses-devel cmake "perl(ExtUtils::MakeMaker)"
# deps for poky
sudo yum install -y python38 patch diffstat texi2html texinfo subversion chrpath git wget
# deps for qemu
sudo yum install -y gtk3-devel
# deps for firemarshal
sudo yum install -y python38-pip python38-devel rsync libguestfs-tools makeinfo expat ctags
# Install GNU make 4.x (needed to cross-compile glibc 2.28+)
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-8-make
# install DTC
sudo yum install -y dtc
sudo yum install -y python

# install verilator
git clone http://git.veripool.org/git/verilator
cd verilator
git checkout v4.034
autoconf && ./configure && make -j$(nproc) && sudo make install
```

Ubuntu/Debian-based 平台的用户根据如下命令下载依赖项：

```shell
#!/bin/bash

set -ex

sudo apt-get install -y build-essential bison flex software-properties-common curl
sudo apt-get install -y libgmp-dev libmpfr-dev libmpc-dev zlib1g-dev vim default-jdk default-jre
# install sbt: https://www.scala-sbt.org/release/docs/Installing-sbt-on-Linux.html#Ubuntu+and+other+Debian-based+distributions
echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt-get update
sudo apt-get install -y sbt
sudo apt-get install -y texinfo gengetopt
sudo apt-get install -y libexpat1-dev libusb-dev libncurses5-dev cmake
# deps for poky
sudo apt-get install -y python3.8 patch diffstat texi2html texinfo subversion chrpath wget
# deps for qemu
sudo apt-get install -y libgtk-3-dev gettext
# deps for firemarshal
sudo apt-get install -y python3-pip python3.8-dev rsync libguestfs-tools expat ctags
# install DTC
sudo apt-get install -y device-tree-compiler
sudo apt-get install -y python
# install git >= 2.17
sudo add-apt-repository ppa:git-core/ppa -y
sudo apt-get update
sudo apt-get install git -y

# install verilator
sudo apt-get install -y autoconf
git clone http://git.veripool.org/git/verilator
cd verilator
git checkout v4.034
autoconf && ./configure && make -j$(nproc) && sudo make install
```


chipyard安装的详细信息见 [此处](https://chipyard.readthedocs.io/en/stable/Chipyard-Basics/Initial-Repo-Setup.html),此处只需要根据1.4.1节中的步骤安装依赖即可。

*若使用本页中提供的镜像，其中自带的chipyard可能存在版本过旧等数个问题，会导致后续安装出错，删除镜像中的chipyard文件后并在后续步骤自行clone可避免后续麻烦。

安装chipyard，gemmini
------------------------------

遵照[此处](https://github.com/ucb-bar/gemmini) Quick Start的指示的命令，运行到Run Simulators为止。

其中包括：

1.```Installing Chipyard and Spike``` 本部分包括安装chipyard，编译chipyard工具链，设置chipyard的环境变量，安装gemmini，安装spike。

2.```Setting Up Gemmini``` 本部分用于设置gemmini的文件、符号链接和子目录。

3.```Building Gemmini Software``` 本部分用于编译 Gemmini程序，以及一些测试程序（如ResNet50），完成之后将会生成上述程序的二进制文件。

4.```Building Gemmini Hardware and Cycle-Accurate Simulators``` 本部分用于通过Verilator创建时钟准确的模拟器，同时也会生成Soc的verilog文件。

5.```Building Gemmini Functional Simulators``` 本部分用于通过spike编译功能模拟器

6.```Run Simulators``` 本部分用于运行测试程序

若Run Simulators内的命令运行成功，则说明本部分安装成功。

*如前文所述，本部分安装步骤需要运行大量脚本，其运行时间可能较长，如果运行中途出现网络问题导致脚本运行失败，则最好直接删除chipyard文件，解决网络问题后重新安装以避免后续可能出现的各种问题。

*注意Installing Chipyard and Spike部分中的，source env.sh命令。这个命令的功能是为chipyard设置一系列环境变量，而这个命令在每次进入当前环境时都会失效并需要重新输入，为保证后续步骤也能正常运行，建议将```source ~/chipyard/env.sh``` 写入.bashrc中，以保证每当进入当前环境时，env.sh 总是被执行过。


安装onnxruntime—riscv
------------------------------

在完成安装gemmini后，[gemmini](https://github.com/ucb-bar/gemmini#software)中software一节介绍了运行onnx模型的方法，且在```chipyard/gemmini/software/onnxruntim-riscv```的位置，gemmini已经安装了onnxruntime-riscv。但需要注意的是，此处gemmini的文档的链接指向的是一个非常陈旧的onnxruntime的分支，且gemmini中已经包含的onnxruntim-riscv也是一个较旧的版本。

因此不需要执行gemmini中software介绍的步骤，而可以直接阅读[此处](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/docs/BUILD.md)文档中的步骤进行操作。若要测试在spike上运行模型，则需要根据如下步骤操作。此文档存在部分描述不清和细节缺失的问题，详细描述如下：

1. ```Setting up your Toolchain``` 本步骤对应的是安装chipyard时运行的```./scripts/build-toolchains.sh esp-tools```命令。若已经根据上述步骤成功安装chipyard 的Toolchain，则本部分不需要额外操作。

2. ```Building this repo```

*在运行该部分前需要运行```cmake --version```确认当前环境的cmake版本是否大于3.12.0，若不满足，后续的步骤会报错。此处要求的cmake由于版本较新，可能并不支持自动安装，而需要手动拉取源码并进行编译。[此处](https://blog.csdn.net/Boys_Wu/article/details/104940575)为教程。

若已经安装chipyard的toolchain，且在当前环境中运行过env.sh的话，本步骤要求的"have riscv g++ in your PATH"条件就已经满足。
本部分分为如下几步：

        2.1.  下载安装onnxruntime

          安装前首先删除位于 ```chipyard/generators/gemmini/software```中的onnxruntime-riscv文件夹

        ```shell
        cd chipyard/generators/gemmini/software
        rm -rf onnxruntime-riscv
        git clone  git@github.com:ucb-bar/onnxruntime-riscv.git
        git submodule update --init --recursive
        #安装新版本的onnxruntime-riscv
        ```

        2.2.  选择启用的Gemmini数据类型

        在编译onnxruntime前，需要选择启用的Gemmini数据类型，有fp32与int8可选。选择方式是通过修改```chipyard/generators/gemmini/software/onnxruntime-riscv/cmake/CMakeLists.txt ```文件，找到其中的

        ```shell
        option(onnxruntime_SYSTOLIC_FP32 "If Systolic is enabled, whether to use for fp32 ops" ON)
        option(onnxruntime_SYSTOLIC_INT8 "If Systolic is enabled, whether to use for int8 ops" OFF) 
        ```

        并进行修改。

        文档中提到的需要确认```systolic_params.h```，```gemmini_params.h```文件是否一致，但在测试中，当前版本在此处并不需要特别进行什么操作。

        如果确实需要确认则这两个文件的具体地址分别位于```gemmini/software/gemmini-rocc-tests/include/gemmini_params.h```与```onnxruntime-riscv/onnxruntime/core/mlas/lib/systolic/systolic_params_int8.h```


        *此处需要注意，fp32与int8同时只能选择启用一项，两者不可兼容。

        *其中，如果模型是用于训练，则必须启用fp32.而如果模型是已量化的模型，已将数据转化为int8类型以利用Systolic进行推理的话，则需要启用int8。若启用的类型与模型不符，在运行模型的时候会报“bad syscall” 错误。

        *当需要重新编译onnxruntime的时候，必须首先删除```onnxruntime-riscv/build```文件夹，否则运行```build.sh```脚本的时候实际上并不会应用新的设置。这种情况下后续的runner也应当重新编译以应用更改。

        2.3. 编译ORT

        在```onnxruntime-riscv```目录下运行```./build.sh --config=Release --parallel```, 以编译ORT(Onnxruntime)。编译的输出可以在```build```文件夹下找到。

        其中--config=Release代表以release（-O3）模式编译，--parallel代表并行进行编译。如果希望ORT支持模型训练，还需要加上--enable_training选项。

        2.4. 编译runner

        Model runner为用于加载ONNX模型进行推理的程序，上一步编译的ORT实际上就是在runner中被调用以完成推理的。对于某种特定的ONNX模型，需要特定的runner以支持其运行。在onnxruntime-riscv/systolic_runner中提供了特定的几个runner(或trainer)以支持模型进行推理或训练。

        此处以编译imagenet_runner为例：

        ```shell
        cd onnxruntime-riscv/systolic_runner/imagenet_runner
        ./build.sh --config=Release --parallel
        #此处用于编译runner，输入的命令选项应当与编译ORT时一致。编译的输出为对应目录下的ORT_TEST 文件
        ```
        *当修改配置重新编译ORT时，需要将此处的ORT_TEST 一并删除且重新编译。





3. ```Running via Spike```

*关于[onnxruntime-riscv](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/docs/BUILD.md)文档中此部分的操作，需要注意的是这部分操作不再是在```onnxruntime-riscv```目录下执行，而是在```chipyard//toolchains/esp-tools```目录下完成的，这点在执行前需要注意。

这部分的功能是为了使用spike执行runner并加载onnx模型进行推理的一些前置工作，若并不打算通过spike执行则可以跳过该部分。

包括如下几个步骤：

3.1. patch riscv-pk to no-op the futex and tid syscalls

本部分需要遵照如下的diff修改```chipyard/toolchains/esp-tools/riscv-pk/pk```中的```syscall.h```与```syscall.c```
```shell
--- a/pk/syscall.h
+++ b/pk/syscall.h
@@ -52,6 +52,9 @@
 #define SYS_clock_gettime 113
 #define SYS_set_tid_address 96
 #define SYS_set_robust_list 99
+#define SYS_futex 98
+#define SYS_gettid 178

 #define OLD_SYSCALL_THRESHOLD 1024

 #define SYS_open 1024
```

```shell
diff --git a/pk/syscall.c b/pk/syscall.c
@@ -434,6 +434,7 @@ long do_syscall(long a0, long a1, long a2, long a3, long a4, long a5, unsigned l
     [SYS_brk] = sys_brk,
     [SYS_uname] = sys_uname,
     [SYS_getpid] = sys_getpid,
+    [SYS_gettid] = sys_getpid,
     [SYS_getuid] = sys_getuid,
     [SYS_geteuid] = sys_getuid,
     [SYS_getgid] = sys_getuid,

@@ -462,6 +463,7 @@ long do_syscall(long a0, long a1, long a2, long a3, long a4, long a5, unsigned l
     [SYS_chdir] = sys_chdir,
     [SYS_set_tid_address] = sys_stub_nosys,
     [SYS_set_robust_list] = sys_stub_nosys,
+    [SYS_futex] = sys_stub_success,
   };
```

实际文件中的内容可能和此处的描述略有不同，但并不影响运行。

文档中提到的"double check that the proxy kernel is patched to enable RoCC extensions"的步骤，经过测试发现实际文件内容已经和文档里描述的完全不同，经过测试也发现该部分事实上可以忽略。

3.2. rebuild pk

该部分的操作需要在```chipyard/toolchains/esp-tools/riscv-pk/pk```目录下完成

```shell
$ mkdir build
$ cd build
$ ../configure --prefix=$RISCV --host=riscv64-unknown-elf
$ make
$ make install
```

最后需要在```chipyard/toolchains/esp-tools/riscv-isa-sim```下执行```git pull origin master```用于让 Spike 使用最新的 Gemmini ISA。

完成该步骤后，若已经准备好现成的已量化的ONNX模型，以及与之对应的model runner，则可以直接执行以完成推理，具体执行所需的命令/参数与相应的runner的实现方式相关。


导出ONNX模型
=============================
ONNX 支持多个主流的深度学习框架，如Pytorch，TensorFlow等等，也支持一些传统的机器学习框架如Sciki-learn, LightGBM, XGBoost, LibSVM。具体的将这些模型转化为ONNX模型的方法可以在[此处](https://onnxruntime.ai/docs/tutorials/accelerate-pytorch/)找到。需要注意的是，若想将这些模型转化为onnx模型，则可能对编写模型时使用的功能有一定限制，否则转化工具可能无法转化成功，具体的限制详见各个框架的转换文档，该部分的文档描述较为详细。

对于ONNX格式的模型，可以在[netron.app](https://netron.app/)上查看其结构。

而在[ONNX model zoo](https://github.com/onnx/models)上，则可以直接下载各类ONNX模型和已量化的ONNX模型.

量化ONNX模型
=============================
onnxruntime-riscv提供了将原始的onnx模型进行量化的工具，量化为int8类型的模型更适用于在gemmini的Systolic上运行推理。关于量化工具的介绍详细见[此处](https://github.com/ucb-bar/onnxruntime-riscv/tree/2021-12-23/systolic_runner/quantization)与[此处](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/docs/quantizer.md)。

关于将一个原始ONNX模型量化的操作步骤则可见[此处](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/imagenet_runner/README.md#e2e-resnet-example),其步骤如下:
1. 
```shell
python3 optimize.py --input=models/resnet50/model.onnx  --output=models/resnet50/model_opt.onnx
#optimize.py用于优化原模型
```

2. 
```shell
python3 calibrate.py --model_path $MODEL/model_opt.onnx   --dataset_path $MODEL --output_model_path $MODEL/model_opt_quantized.onnx  --static=True --data_preprocess=mxnet --mode=int8
#calibrate.py用于量化已优化过的模型，其中data_preprocess=mxnet用于选择数据处理的方式, 此处选择的方式应当与运行时的-p 参数一致。
```

运行/编写runner
=================================

运行已有的imagenet runner
-------------------------------

若以通过spike执行imagenet模型为例，[此处](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/imagenet_runner/README.md)为与imagenet runner对应的文档。若已量化的ONNX模型已经准备好，则可以在```onnxruntime-riscv/systolic_runner/imagenet_runner```目录下执行```spike --extension=gemmini pk ort_test -m googlenet.onnx  -i images/cat.jpg  -p caffe2 -x 1 -O 0```,以执行runner并完成推理。

*其中pk代表使用代理内核；ort_test为imagenet runner的可执行文件； -m googlenet.onnx为加载的模型路径；-i images/cat.jpg 为用于推理的输入（也可以将输入的图片路径写在.txt文件中，以批量输入并推理）；-p caffe2 用于选择processing style，该参数与使用的模型的量化方式相关; -x 1 代表选择执行方式（dataflows），其中0代表选择用cpu执行，1代表output-stationary mode（OS），2代表weight-stationary mode(WS);-O 0代表选择优化等级，0代表禁用优化，99代表启用一切可能的优化

*在使用imagenet runner运行推理时，需要注意：

1.在提高优化等级后，模型内的计算流程可能会发生改变（例如：原本需要调用MatMul执行计算的layer，可能会改为调用Conv)

2.不同dataflow下支持的操作是不同的，例如如果模型中存在conv或add操作，则在OS mode下执行将会报错（"Unsupport datatype"），改为WS mode则可以成功.

3.作者表示目前的onnxruntime-riscv不能支持同时使用OS与WS，因此会导致上述问题，这会在后续版本修复。

若使用项目中已经给出的Imagenet runner与已量化的onnx模型运行推理成功，则说明这部分的安装无误。

onnxruntime-riscv 提供的可以适用于各类模型的runner可以在[此处](https://github.com/ucb-bar/onnxruntime-riscv/tree/2021-12-23/systolic_runner)找到，而如果待移植的模型不属于其中，则需要考虑自行编写。

自行编写model runner并运行
----------------------

关于使自行编写model runner，[ONNX官方网站](https://onnxruntime.ai/docs/)上提供的C/C++教程极少，仅有ORT的API doc与示例程序且缺乏解释与注释。如果需要参考，可以查看```onnxruntime-riscv/systolic_runner```提供的诸个runner的源码，[onnxruntime-riscv](https://github.com/ucb-bar/onnxruntime-riscv/tree/2021-12-23/systolic_runner/docs)中也提供了数个对开发者理解onnx有帮助的文档，但同样内容不多且不成体系，仅仅从作者个人角度粗略介绍了一些开发中需要注意的问题。而ONNX提供给其他语言（如Python）的教程，则比起C/C++更为详细一些，也可以作为参考。

如果基于[imagenet runner的部分源码](https://github.com/ucb-bar/onnxruntime-riscv/blob/2021-12-23/systolic_runner/imagenet_runner/src/runner.cpp)分析ONNX API的使用，运行器需要做的主要工作包括：

1.创建ORT_session，并将各个参数（如模型路径，优化的等级，dataflow mode）传递给session；

2.实现一个将input整形为onnx模型可以接收的input_tensor的算法；

3.调用session.run，运行推理；

4.输出推理结果。

```shell
Ort::Env env(static_cast<OrtLoggingLevel>(cmd["debug"].as<int>()), "test");
//每个进程对应一个env，env用于维护线程池与存放状态信息

Ort::SessionOptions session_options;
//session_options用于存放一些ORT参数，并传递给ORT:session

OrtSessionOptionsAppendExecutionProvider_Systolic(session_options, /*use_arena=*/ 1, /*accelerator_mode=*/ (char) cmd["execution"].as<int>())//设置执行模式（dataflow）
session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(cmd["optimization"].as<int>()));//设置优化等级
...

Ort::Session session(env, model_path, session_options);//根据决定好的参数，创建session
...

Ort::AllocatorWithDefaultOptions allocator;//创建Mem allocator
...

char* input_name = session.GetInputName(i, allocator);
size_t num_input_nodes = session.GetInputCount();
Ort::TypeInfo type_info = session.GetInputTypeInfo(i);//从加载的模型中获取input的一些相关信息，如dims，nodenames等
...

unsigned char *data = stbi_load(path.c_str(), &dimX, &dimY, &numChannels, 0);
unsigned char *orig_data = data;
std::vector<float> input_tensor_values(input_tensor_size);
...

for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {
      unsigned char r = *(data++);
      unsigned char g = *(data++);
      unsigned char b = *(data++);

      if (preprocess == "caffe2") {
        input_tensor_values[(0*224 + i)*224 + j] = b - 103.939;
        input_tensor_values[(1*224 + i)*224 + j] = g - 116.779;
        input_tensor_values[(2*224 + i)*224 + j] = r - 123.68;  
      } 
      else if (preprocess == "caffe") {
        input_tensor_values[(0*224 + i)*224 + j] = (b - 103.94)*0.017;
        input_tensor_values[(1*224 + i)*224 + j] = (g - 116.78)*0.017;
        input_tensor_values[(2*224 + i)*224 + j] = (r - 123.68)*0.017;  
      } else if (preprocess == "mxnet") {
        input_tensor_values[(0*224 + i)*224 + j] = (b/255.0 - 0.406)/0.225;
        input_tensor_values[(1*224 + i)*224 + j] = (g/255.0 - 0.456)/0.224;
        input_tensor_values[(2*224 + i)*224 + j] = (r/255.0 - 0.485)/0.229;  
      } else {
        std::cout << "Unknown preprocess option: " << preprocess << std::endl;
        exit(1);
      }
    }
  }
 auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
 Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
//在本例中，该段算法用于将255*255*3的图片转化为imagenet模型可接受的input_tensor的格式。如果需要执行其他任务，可能需要自行编写类似的算法，其中caffe2，mxnet等参数和使用量化器时使用的量化参数有关。
...

auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
//调用gemmini，执行推理的各层的操作，直到给出推理结果的全部过程都在执行session.Run()的过程中完成。结果则返回为output_tensor
...

//余下部分为将ouput_tensors显示的代码，此处略
```

