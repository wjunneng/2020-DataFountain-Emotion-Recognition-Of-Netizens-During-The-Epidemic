# 2020-DataFountain-Emotion-Recognition-Of-Netizens-During-The-Epidemic
2020 DataFountaion 疫情期间网民情绪识别

### 注意

>>> 
> cmd: file nCov_10k_test.csv
>
> 出现：nCov_10k_test.csv: Non-ISO extended-ASCII text, with very long lines, with CRLF, NEL line terminators
> 
>在window系统下,使用记事本打开,点击另存为utf-8格式.
>>>

# Trick

    1. 多类别不均衡的话, 这时候直接使用神经网络优化交叉熵损失得到的结果, f1显然不是全局最优的, 二分类下可以用阈值搜索, 
    如果是多分类怎么做一个阈值搜索呢?传统的多分类我们预测结果使用argmax(logits)这时候可以形式化的表达为求argmax(w*logits)使得f1均值最大.
    其中w就是要求得的再放缩权重.
    

### 参考

>>>
> https://github.com/guoday/CCF-BDCI-Sentiment-Analysis-Baseline
>>>