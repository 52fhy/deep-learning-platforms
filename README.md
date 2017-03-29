# deep-learning-platforms
deep-learning platforms,framework,data（深度学习平台、框架、资料）

## 深度学习平台

### 腾讯DI-X深度学习平台

网址：https://www.qcloud.com/product/dix

简介：DI-X（Data Intelligence X）是基于腾讯云强大计算能力的一站式深度学习平台。它通过可视化的拖拽布局，组合各种数据源、组件、算法、模型和评估模块，让算法工程师和数据科学家在其之上，方便地进行模型训练、评估及预测。目前支持 TensorFlow、Caffe、Torch 三大深度学习框架，并提供相应的常用深度学习算法和模型。DI-X 可以帮助您快速接入人工智能的快车道，释放数据潜力。

应用场景：图像识别、用户流失预测、DSP精准营销

### 阿里云机器学习 PAI

网址：https://data.aliyun.com/product/learn

简介：阿里云机器学习是基于阿里云分布式计算引擎的一款机器学习算法平台。用户通过拖拉拽的方式可视化的操作组件来进行试验，使得没有机器学习背景的工程师也可以轻易上手玩转数据挖掘。平台提供了丰富的组件，包括数据预处理、特征工程、算法组件、预测与评估。所有算法都经历了阿里云内部业务的大数据的锤炼。阿里云机器学习帮助您的业务从BI跨入AI，让越来越多的人享受人工智能带来的福利。

应用场景：金融风控、疾病预测、商品推荐、文本分析、贷款预测


### 百度深度学习（Baidu Deep Learning）

网址：https://cloud.baidu.com/product/bdl.html

简介：百度深度学习是一款面向海量数据的深度学习平台。平台基于PaddlePaddle/TensorFlow开源计算框架，支持GPU运算，依托百度云分布式技术，为深度学习技术的研发和应用提供可靠性高、扩展灵活的云端托管服务。

应用场景：人脸相似度比对、人脸搜索、语音识别

### Amazon Machine Learning

网址：https://aws.amazon.com/cn/machine-learning/?nc2=h_m1
Amazon AI：https://aws.amazon.com/cn/amazon-ai/

简介：Amazon Machine Learning 是一项面向各个水平阶层开发人员的服务，可以帮助他们利用机器学习技术。Amazon Machine Learning 提供可视化的工具和向导，指导您按部就班地创建机器学习模型，而无需学习复杂的机器学习算法和技术。当您的模型准备好以后，Amazon Machine Learning 只要使用简单的 API 即可让您的应用程序轻松获得预测能力，而无需实现自定义预测生成码或管理任何基础设施。

应用场景：欺诈侦测、内容个性化、营销广告、客户不良表现预测、自动化解决方案推荐

## 深度学习框架

### TensorFlow

开发商： Google
官网： https://www.tensorflow.org/
GitHub：https://github.com/tensorflow/tensorflow
中文站：http://www.tensorfly.cn/

简介：TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。

TensorFlow可被用于语音识别或图像识别等多项机器深度学习领域，对2011年开发的深度学习基础架构DistBelief进行了各方面的改进，它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow将完全开源，任何人都可以用。

### Caffe

开发者： 加州伯克利分校
官网： http://caffe.berkeleyvision.org/
GitHub：https://github.com/BVLC/caffe

资料：http://suanfazu.com/t/caffe/281

简介：Caffe605是一个清晰而高效的深度学习框架，其作者是博士毕业于UC Berkeley的[贾扬清](http://daggerfs.com/)，目前在Google工作。

Caffe是纯粹的C++/CUDA架构，支持命令行、Python和MATLAB接口；可以在CPU和GPU直接无缝切换。

### Torch

官网： http://torch.ch/

简介：Torch是一个广泛支持机器学习算法的科学计算框架。易于使用且高效，主要得益于一个简单的和快速的脚本语言LuaJIT，和底层的C / CUDA实现。

Torch目标是让你通过极其简单过程、最大的灵活性和速度建立自己的科学算法。Torch有一个在机器学习领域大型生态社区驱动库包，包括计算机视觉软件包，信号处理，并行处理，图像，视频，音频和网络等，基于Lua社区建立。

Torch 的核心是流行的神经网络，它使用简单的优化库，同时具有最大的灵活性，实现复杂的神经网络的拓扑结构。你可以建立神经网络和并行任意图，通过CPU和GPU等有效方式。

Torch 广泛使用在许多学校的实验室以及在谷歌/ deepmind，推特，NVIDIA，AMD，英特尔和许多其他公司。

### Theano

开发者： 蒙特利尔理工学院
官网：http://deeplearning.net/software/theano/
GitHub：https://github.com/Theano/Theano

简介：2008年诞生于蒙特利尔理工学院,Theano派生出了大量深度学习Python 软件包，最著名的包括Blocks和Keras。

### PaddlePaddle

开发商： 百度
官网：http://www.paddlepaddle.org/cn/index.html

简介：PaddlePaddle的前身是百度于2013年自主研发的深度学习平台Paddle（Parallel Distributed Deep Learning，并行分布式深度学习），且一直为百度内部工程师研发使用。

PaddlePaddle在深度学习框架方面，覆盖了搜索、图像识别、语音语义识别理解、情感分析、机器翻译、用户画像推荐等多领域的业务和技术。目前，PaddlePaddle已实现CPU/GPU单机和分布式模式，同时支持海量数据训练、数百台机器并行运算，以应对大规模的数据训练。此外，PaddlePaddle具备高质量GPU代码，提供了Neural Machine Translation、推荐、图像分类、情感分析、Semantic Role Labelling等5个Task，每个Task都可迅速上手，且大部分任务可直接套用。

### Brainstorm

官网：http://neuroimage.usc.edu/brainstorm/
GitHub：https://github.com/brainstorm-tools/brainstorm3

简介：来自瑞士人工智能实验室IDSIA 的一个非常发展前景很不错的深度学习软件包，Brainstorm 能够处理上百层的超级深度神经网络——所谓的公路网络Highway Networks。

### Chainer

官网：http://chainer.org/

简介：来自一个日本的深度学习创业公司Preferred Networks，2015年6月发布的一个Python框架。Chainer 的设计基于 define by run原则，也就是说，该网络在运行中动态定义，而不是在启动时定义。

### Deeplearning4j

官网：https://deeplearning4j.org/
GitHub：https://github.com/deeplearning4j/deeplearning4j

简介：Deeplearning4j 是”for Java”的深度学习框架，也是首个商用级别的深度学习开源库。Deeplearning4j由创业公司Skymind于2014 年6月发布，使用Deeplearning4j的不乏埃森哲、雪弗兰、博斯咨询和IBM等明星企业。

DeepLearning4j 是一个面向生产环境和商业应用的高成熟度深度学习开源库，可与Hadoop和Spark集成，即插即用，方便开发者在APP中快速集成深度学习功能，可应用于以下深度学习领域：

- 人脸/图像识别
- 语音搜索
- 语音转文字(Speech to text)
- 垃圾信息过滤(异常侦测)
- 电商欺诈侦测

### Marvin

官网：http://marvin.is/
GitHub：https://github.com/PrincetonVision/marvin
简介：是普林斯顿大学视觉工作组新推出的C++ 框架。该团队还提供了一个文件用于将Caffe模型转化成语Marvin兼容的模式。

### ConvNetJS

官网：http://www-cs-faculty.stanford.edu/people/karpathy/convnetjs/
GitHub：https://github.com/karpathy/convnetjs
简介：这是斯坦福大学博士生Andrej Karpathy开发浏览器插件，基于万能的JavaScript可以在你的游览器中训练神经网络。Karpathy还写了一个ConvNetJS 的入门教程，以及一个简洁的浏览器演示项目。

### MXNet
官网：http://mxnet.io/
GitHub：https://github.com/dmlc/mxnet/
简介：出自CXXNet、Minerva、Purine 等项目的开发者之手，主要用C++ 编写。MXNet 强调提高内存使用的效率，甚至能在智能手机上运行诸如图像识别等任务。

### Neon
官网：http://neon.nervanasys.com/index.html/
GitHub：https://github.com/NervanaSystems/neon
简介：由创业公司Nervana Systems 于2015年五月开源，在某些基准测试中，由Python和Sass 开发的Neon的测试成绩甚至要优于Caffeine、Torch 和谷歌的TensorFlow。


## 深度学习类库

### Keras

语言：Python
官网：https://keras.io/
GitHub：https://github.com/fchollet/keras

简介： 基于Theano 和 TensorFlow的深度学习库

## 深度学习范例
1、TensorFlow Examples
https://github.com/aymericdamien/TensorFlow-Examples

## 深度学习资料

1、自上而下的学习路线: 软件工程师的机器学习[译文]
https://github.com/ZuzooVn/machine-learning-for-software-engineers/blob/master/README-zh-CN.md

2、Deep Learning Papers Reading Roadmap
https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap

3、Awesome - Most Cited Deep Learning Papers[论文]
https://github.com/terryum/awesome-deep-learning-papers

4、data-science-ipython-notebooks[笔记]
https://github.com/donnemartin/data-science-ipython-notebooks

5、Oxford Deep NLP 2017 course[讲座]
https://github.com/oxford-cs-deepnlp-2017/lectures

6、ty4z2008/Qix: Machine Learning、Deep Learning、PostgreSQL、Distributed System、Node.Js、Golang
https://github.com/ty4z2008/Qix

