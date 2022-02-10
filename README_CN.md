# BP实验说明——中文翻译版byJeremy

论文主页：[Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability](https://arxiv.org/abs/2005.00191)

[TOC]

## 实现提示

BP是CP的同源工作，十分推荐您也阅读CP论文。

本项目源码是基于[CP源码](https://github.com/zhuchen03/ConvexPolytopePosioning/)构建的。

原实验在PyTorch- v1.3.1/Cuda 10.1的软件环境下进行，同时配置了NVIDIA Titan RTX显卡。



##  前置要求

[数据集下载链接](https://drive.google.com/file/d/1wVRobdlwvD9-VL9mYKCu_onq8PbbyP0V/view?usp=sharing)

[受害模型与子模型下载链接](https://drive.google.com/file/d/1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko/view?usp=sharing)

你也可以使用脚本<train_cifar10_models.py>训练任何来源于[kuangliu](https://github.com/kuangliu/pytorch-cifar)的模型，一个实例为：

```shell
python train_cifar10_models.py --gpu 0 --net DPN92 --train-dp 0.3 --sidx 0 --eidx 4800
```



## 实验

本工程中，对单目标有5个不同类的实验。你可以执行<craft_poisons_transfer.py>脚本来呈现所有攻击。本实验中有大量的参数，这里列举一些比较重要的：

* --mode：convex(CP)或mean(BP)
* --net-repeat：BP对每个网络中毒物特征向量的计算次数。（1:CP）
* --target-index：选择一个作为目标的序列。在实验中我们选择0～49已获得更严格的数据。
* --poison-ites：生成毒物时的迭代次数，默认次数与论文中相同。
* --device and --gpu：决定你想使用的硬件与gpu数量。

为了评估针对一个独立的受害网络的单次攻击实例，你可以执行脚本<eval_poisons_transfer.py>。你需要保障你对 *--eval-poisons-root* 参数传递正确，这需要你的目录中包含对所有目标的特定模式的攻击结果，即：目标序列不应该是路径的一部分。

对于所有接下来的实验，我们提供了基本sh脚本。希望你使用GPU，但是你也可以轻易使用CPU应对。

现在，我们回到基本实验。

### 对建立受害者网络特征空间的训练集完全了解的实验

需要注意攻击者依旧不知道已经预训练好的数据集（除了加入的毒物）。可以调用<launch/attack-transfer-18.sh>和<launch/attack-end2end-12.sh>来进行实验。如果需要评估毒物在受害者模型中的表现，可以调用<launch/eval-transfer.sh>和<launch/eval-end2end.sh>。

如果想在两个迁移学习情境中执行target-index为17的CP，设定net-repeat为1，注意4000表示攻击有4000次迭代( --poison-ites 4000)。

``` shell
# Linear transfer learning
bash launch/attack-transfer-18.sh 0 convex 17 1
bash launch/eval-transfer.sh 0 17 attack-results/100-overlap/linear-transferlearning/convex/4000/

# End-to-end training, --poison-ites by default is set to 1500.
bash launch/attack-end2end-12.sh 0 convex 17 1
bash launch/eval-end2end.sh 0 17 attack-results/100-overlap/end2endtraining/convex/1500/
```

如果你运行了各种目标序列的各种攻击，并想获得论文中的那类图片，可以执行脚本<analysis/compare-attacks-100-overlap-linear-transferlearning.sh>和<analysis/compare-attacks-100-overlap-end2end-training.sh>。



### 对建立受害者网络特征空间的训练集了解一半的实验

我们可以调用<launch/attack-transferdifftraining-50.sh>和<launch/attack-end2end-difftraining-50.sh>来进行实验。如果需要评估毒物在受害者模型中的表现，可以调用<launch/eval-difftraining.sh>。

下述的脚本执行对第17个目标的1500次迭代BP-3x。

``` shell
bash launch/attack-transfer-difftraining-50.sh 0 mean 17 3
bash launch/eval-difftraining.sh 0 17 attack-results/50-overlap/linear-transferlearning/mean-3Repeat/1500

# End-to-end training
bash launch/attack-end2end-difftraining-50.sh 0 mean 17 3
bash launch/eval-difftraining.sh 0 17 attack-results/50-overlap/end2end-training/mean-3Repeat/1500
```

如果你运行了各种目标序列的各种攻击，并想获得论文中的那类图片，可以执行脚本<analysis/compare-attacks-50-overlap-linear-transferlearning.sh>和<analysis/compare-attacks-50-overlap-end2end-training.sh>。需要提醒，因为执行这些攻击的高昂代价（特别是CP），我们的结果中没有包含此类设定的端到端攻击。我们将在不久的将来有应对方案。



### 对建立受害者网络特征空间的训练集完全不了解的实验

我们可以调用<launch/attack-transferdifftraining-0.sh>和<launch/attack-end2end-difftraining-0.sh>来进行实验。如果需要评估毒物在受害者模型中的表现，可以调用<launch/eval-difftraining.sh>。

下述的脚步执行对第17个目标的1500次迭代BP。

``` shell
bash launch/attack-transfer-difftraining-0.sh 0 mean 17 1
bash launch/eval-difftraining.sh 0 17 attack-results/50-overlap/linear-transferlearning/mean/1500

# End-to-end training
bash launch/attack-end2end-difftraining-0.sh 0 mean 17 1
bash launch/eval-difftraining.sh 0 17 attack-results/50-overlap/end2endtraining/mean/1500
```

如果你运行了各种目标序列的各种攻击，并想获得论文中的那类图片，可以执行脚本<analysis/compare-attacks-0-overlap-linear-transferlearning.sh>和<analysis/compare-attacks-0-overlap-end2end-training.sh>。需要提醒，因为执行这些攻击的高昂代价（特别是CP），我们的结果中没有包含此类设定的端到端攻击。我们将在不久的将来有应对方案。



### 基于我们在BP背后冻结参数直觉的实验

调用<launch/run_attack_fixedcoeffs.py>进行实验，这个实验可以评估攻击者创建毒物对受害模型的性能表现。最后，如果需要进行更深的分析(以获得论文中的图像)，可以执行<analysis/compare_diff_fixedcoeffs.py>脚本。

下述脚本执行了9种不同的BP，这是一个全了解训练集的线性迁移学习例子：

``` shell
python launch/run_attack_fixedcoeffs.py
```



### 多目标模式

和单目标模式很相似，可以执行launch和analysis脚本来进行实验。

多目标的结果可以从这里[下载](https://drive.google.com/file/d/13p24HnylrDLPIv3EHv7VGvwJgbqqctSg/view?usp=sharing)



## 结果

为了让您生活更简单，节约您的金钱与时间，可以[在此下载](https://drive.google.com/file/d/1mbQs239HVxnOLHdh1I2lE1B3zsmaBSO3/view?usp=sharing)我们的实验细节。多目标结果在上一部分中有呈现。与代码结合，需要解压Multi-Traget-Mode文件夹。

你需要耐心一点，因为记录的格式不是很完美。:)



# 有任何问题，欢迎联系原作者