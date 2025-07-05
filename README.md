# PASR：基于 PyTorch 的自动语音识别框架

PASR（PyTorch Automatic Speech Recognition）一个灵活可扩展的端到端自动语音识别（ASR）框架，基于 PyTorch 实现，支持流式与非流式模式、多种模型结构、数据增强、实验管理及多平台部署。

## 本项目使用的环境：
 - Miniconda 3
 - Python 3.11
 - Pytorch 2.7.1
 - Windows 11

## 🌟 主要特性

- **多种模型结构**：Conformer、Efficient Conformer、DeepSpeech2、Squeezeformer、Transformer 等
- **多模式推理**：流式（实时）与非流式（离线）两种工作模式
- **数据增强**：支持多种数据增强方法，包含噪声增强、混响增强、语速增强、音量增强、重采样增强、位移增强、SpecAugmentor、SpecSubAugmentor等。
- **实验管理**：训练日志、TensorBoard 可视化、超参数自动记录
- **多种部署方式**：导出为 TorchScript、ONNX；支持 gRPC/REST 服务和 Web GUI
- **易扩展**：模块化设计，方便添加新模型、解码器或数据处理流程
- **预处理**：声学特征提取（Mel-Spectrogram）、归一化、增量提取

## 📦 环境搭建与安装

```bash
# 1. 克隆仓库并进入目录
git clone https://github.com/yourusername/PASR.git && cd PASR

# 2. 创建并激活 Conda 环境
conda create -n pasr python=3.11 -y
conda activate pasr

# 3. 安装依赖
pip install -r requirements.txt
```

## 1. 数据下载与预处理

```bash
支持多种公开语音数据集，如 AISHELL-1、LibriSpeech、THCHS-30、WenetSpeech 等：

# 以 THCHS-30 为例：
# 下载数据集
https://aistudio.baidu.com/datasetdetail/38012/0
# 将压缩包位置用filepath指定，运行代码自动跳过下载，直接解压文件文本生成数据列表
python thchs_30.py --filepath = "D:\\Download\\data_thchs30.tgz"
# 最后执行下面的数据集处理程序，这个程序是把我们的数据集生成JSON格式的训练和测试数据列表，分别是`test.jsonl、train.jsonl`。然后使用Sentencepiece建立词汇表模型，建立的词汇表模型默认存放在`dataset/vocab_model`目录下。最后计算均值和标准差用于归一化，默认使用全部的语音计算均值和标准差，并将结果保存在`mean_istd.json`中。
python create_data.py
```

## 2. 配置文件详解

```bash
所有配置集中在 `configs/` 目录下，以 YAML 格式存储。
这些文件共同定义了从数据处理、模型结构到训练和解码的每一个环节。

```yaml
# configs/conformer.yaml
# 1. 模型结构定义
encoder_conf:
  encoder_name: 'ConformerEncoder' # 使用的编码器类型
  encoder_args:
    output_size: 256               # 模型隐层维度 (d_model)
    attention_heads: 4             # 注意力头的数量
    num_blocks: 12                 # 编码器层数

decoder_conf:
  decoder_name: 'BiTransformerDecoder' # 使用的解码器类型
  decoder_args:
    num_blocks: 3                  # 解码器层数

model_conf:
  model: 'ConformerModel'          # 最终使用的模型组合
  model_args:
    streaming: True                # 是否为流式模型
    ctc_weight: 0.3                # CTC损失的权重

# 2. 数据与预处理定义
dataset_conf:
  train_manifest: 'dataset/train.jsonl' # 训练数据列表路径
  test_manifest: 'dataset/test.jsonl'  # 测试数据列表路径
  batch_sampler:
    batch_size: 16                 # 训练时的批量大小
  dataLoader:
    num_workers: 8                 # 数据加载的并行进程数

preprocess_conf:
  feature_method: 'fbank'          # 音频特征提取方法
  method_args:
    num_mel_bins: 80               # Mel频谱的维度

# 3. 训练策略定义
optimizer_conf:
  optimizer: 'Adam'                # 优化器选择
  optimizer_args:
    lr: 0.001                      # 初始学习率
  scheduler: 'WarmupLR'            # 学习率调度器
  scheduler_args:
    warmup_steps: 25000            # 学习率预热步数

train_conf:
  max_epoch: 200                   # 最大训练轮数
  accum_grad: 4                    # 梯度累积步数，用于变相扩大batch_size
  log_interval: 100                # 每隔多少步打印一次日志
```

## 3. 特征提取与缓存

```bash
#可预先提取 Mel-Spectrogram 特征并缓存，以加速训练，若未进行，在模型训练该过程中会自动执行
python extract_features.py 
```

## 4. 模型训练

```bash
python train.py
# 单机多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

**训练过程中可视化**：
#在训练过程中，程序会使用VisualDL记录训练结果，可以通过在根目录执行以下命令启动VisualDL
visualdl --logdir=log --host=0.0.0.0
#然后再浏览器上访问`http://localhost:8040`可以查看结果显示
```
## 5. 模型评估

```bash
python eval.py --resume_model=models/ConformerModel_fbank/best_model/
```

## 6. 模型导出

```bash
python export_model.py
```

## 7. 添加标点符号模型

```bash
#下载标点符号的模型放在`models/`目录下。
https://pan.baidu.com/s/1GgPU753eJd4pZ1LxDjeyow?pwd=7wof
```

## 8. 本地预测

```bash
#非流式
python infer_path.py --audio_path=./dataset/test.wav
#流式
python infer_path.py --audio_path=./dataset/test.wav --real_time_demo True
#添加标点符号
python infer_path.py --audio_path=./dataset/test.wav --use_punc=True
```

## 9. Web部署模型

```bash
#在服务器执行下面命令，创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，如果在本地运行的话，在浏览器上访问`http://localhost:5000`。打开页面之后可以选择上传长音或者短语音音频文件，也可以在页面上直接录音，录音完成之后点击上传，播放功能只支持录音的音频。
pip install aiofiles -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
python infer_server.py
```

## 10. GUI部署模型

```bash
#通过打开页面，在页面上选择长语音或者短语音进行识别，也支持录音识别实时识别，带播放音频功能。该程序可以在本地识别，也可以通过指定服务器调用服务器的API进行识别。
pip install pyaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install websockets -i https://pypi.tuna.tsinghua.edu.cn/simple
python infer_gui.py
```


## 🗂️ 项目目录结构

```plain
PASR/
├── configs/            # YAML 配置文件目录
├── download_data/      # 数据集下载脚本
├── create_data.py      # 数据预处理
├── extract_features.py # 特征提取脚本
├── train.py            # 训练主程序
├── infer_path.py       # 离线推理脚本
├── infer_server.py     # Web界面部署
├── infer_gui.py        # GUI界面部署
├── export_model.py     # 模型导出
├── pasr/               # 核心模块：数据加载、模型定义、解码器、工具函数
├── requirements.txt    # Python 依赖列表
├── checkpoints/        # 训练中保存模型及日志
├── dataset/            # 原始及处理后数据
└── models/             # 导出后的部署模型
```

---

## 📈 评估指标

- **CER**（Character Error Rate）: 字符错误率

执行 `eval.py` 后，可在标准输出中查看详细结果。

---

## 参考资料
 - https://github.com/yeyupiaoling/PPASR