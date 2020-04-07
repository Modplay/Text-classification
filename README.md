# Text-classification
A classification of Text which is implemented by pytorch
## checkpoints
存储训练好的模型，vectorizer
## data
对数据进行处理，加载
## model
存放模型
## utils
工具函数
## config
配置信息
## main
主函数，主要是训练，验证模型
## test
测试模型的准确率
## Requirement
        python : 3.5+
        pytorch : 1.4.0
        cuda : 9.0 (support GPU, you can choose)
## Usage
- first step

python main.py
- second step

python test.py

## Result
| Data/Model(acc)   | AG_News  | SUBJ  | MR    | CR    | MPQA  |
| ------            | ----- | ----- | ----- | ----- | ----- |
| Pooling          | - | - | - | - | - |
| text_cnn               | 82.28 | - | - | - | - |
| Char_CNN          | - | - | - | - | - |
| Multi_Channel_CNN | - | - | - | - | - |
| Multi_Layer_CNN   | - | - | - | - | - |
| LSTM              | - | - | - | - | - |
| LSTM_CNN          | - | - | - | - | - |
| GRU               | 80.1 | - | - | - | - |
| TreeLSTM         | - | - | - | - | - |
| biTreeLSTM        | - | - | - | - | - |
| TreeLSTM_rel     | - | - | - | - | - |
| biTreeLSTM_rel    | - | - | - | - | - |
| CNN_TreeLSTM      | - | - | - | - | - |
| LSTM_TreeLSTM     | - | - | - | - | - |

### In addition:

#### Emphasize
 - `pre_trained_embed` which is using `glove.6B.100d.txt`.

