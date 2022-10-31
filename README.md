### 注意事项
1. 显存不足 16 G 不要训练 CTPN 网络(运行 ctpn 包下的 `train.py`)
2. 显存 4G 即可训练 CRNN 网络(可能不需要 4G, 但未测试)
3. 可通过 data_generator 更换 CRNN 网络训练, 测试数据。如果需要更改数据生成文件夹, 尽量通过参数, 避免修改源文件

### 基本使用
**python 环境下的使用**
```python
python run.py --input <测试图片所在文件夹> --output <生成文件存放文件夹>
```

**IDE 环境下的使用**

直接运行 `run` 文件即可

### 环境配置
仅提供 Anaconda 的环境配置\
先不要尝试换源, 实在下载慢再说
1. 前往 [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 下载最新版本的 anaconda
2. 打开 Anaconda Prompt
3. 输入 `conda create -n <your_env_name> python=3.7` 创建虚拟环境
4. 输入 `activate <your_env_name>` 进入虚拟环境(此时命令行前的环境名应为你创建的虚拟环境)
5. 输入 `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` 安装 pytorch
6. 更换 pycharm 的 python 解释器
   1. 选择 `文件 > 设置 > 项目 > python 解释器
   2. 点击 python 解释器 > 全部显示 > + 号 > conda 环境 > 现有环境
   3. 一直确认即可
   4. 等待 pycharm 进行包扫描(每次安装新的包都需要扫描)
7. 对于其他的依赖的安装(Anaconda Prompt)
**_不要使用其他包管理器, 使用 conda 进行安装_**
   1. 使用 `conda search <package>` 搜索依赖
   2. 使用 `conda install <package>=<version>` 安装依赖
   3. 尽量安装新版本的依赖(最好不要让 conda 更改原有依赖)
