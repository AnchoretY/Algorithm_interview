# BERT

推荐阅读：

- [BERT原理与NSL和MLM](https://anchorety.github.io/Algorithm_interview/dl/bert.html)
- 

1. **简述BERT模型的基本结构**

   **BERT的整体结构：Embedding+Transformer Encoder**

   **输入**：每次输入为两句话，通过[SEP]分隔符隔开

   结构：

   - **Embedding**

     组成：word embedding(无预训练) + position embedding + segment embedding

     ![](https://github.com/AnchoretY/images/blob/master/blog/BERT_input_representation.png?raw=true)

   - **Transformer Encoder**

     Transformer Encoder由**N个Transformer block组成**,每个block可以分为输入和输出两部分,每一部分都按照sublayer的标准结构组成：

     <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ms6w1661ntl.png" alt="image" style="zoom:50%;" />

     - **输入子层**

       输入子层中核心为**Muti-Head Attention**，在其上加入了Norm、残差连接、dropout机制。

     - **输出子层**

       输出子层中sulayer为**Feed Forward**，在其上加入了Norm、残差连接、dropout机制。

       Feed Forward结构：

       <img src="/Users/yhk-home/Library/Application Support/typora-user-images/image-20220413154623295.png" alt="image-20220413154623295" style="zoom:60%;" />

       - **Sublayer结构**

         ​	输入子层与输出子层都遵循下面的结构，其中Muti-Head Attention、Feed Forward即为其中的核心子层。

         <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ad3lkvbrzj.png" alt="image" style="zoom:67%;" />

         ​	

2. **BERT如何进行训练？Loss如何构成？**

   **BERT的训练任务：Mask Language Model(MLM)+Next Sentence Predict(NSP)**

   - **Mask Language Model(MLM)**

     **目标**：根据上下文预测当前词(只预测mask的词)

     **样本生成**：随机选择15%的词进行随机token mask，这15%中10%不替换，10%采用随机词进行替换，80%采用[MASK]标记进行替换

     > 在token mask中加入了随机化操作目的是为了减小mask对模型的影响，mask只会在训练时出现，在预测时并不会出现mask

   - **Next Sentence Predict(NSP)**

     目标：预测第二条句子是否为第一条句子的后续句子（以[SEP]标记分割）

     **样本生成**：从语料库中相邻的句子对中，随机选择50%的样本将第二条句子进行随机替换，替换后的样本为负样本，未进行替换的50%样本为正样本

   **在BERT中的Embedding+Transformer Encoder主结构后，将输出向量用两个全连接网络进行MLM和NSP任务的预测**

   <img src="/Users/yhk-home/Library/Application Support/typora-user-images/image-20220413153015966.png" alt="image-20220413153015966" style="zoom:67%;" />

   **Loss构成：MLM Loss+NSP Loss**

3. **什么是Muti-Head Attention？有什么作用？**

   

4. **BERT为代表的的语言模型和word2vec为代表的词向量模型在进行文本表征任务时有什么区别？**

**最大的区别：词向量模型无法处理同义词的问题**。

词向量模型当前词表示过于依赖上下文N个词的共现关系，而这种关系只是短程依赖，很多时候不能完整的表达出词语的语义信息。

> 例如：两句话中特朗普和川普，表示相同的语义，但依照词向量的建模方法，他们拥有不同的上下文，就会有不同的词向量
>
> - 美国总统**特朗普**决定再墨西哥边境修建隔离墙
>
> - 听说美国那个川普要在墨西哥边境修建隔离墙

3. **BERT与transformer有哪些异同点？**

   相同点：

   - Bert的网络结构就是基于transformer的，在网络结构上基本一致

   不同点：

   - 输入编码不同
     - transformer的编码采用Token Embedding+Positional Embedding，Bert编码采用Token Embedding+Positional Embeddin+**Segment Embedding（用于区分不同序列）**。
     - transformer的Positional Embedding采用**固定的三角函数转换公式**进行编码，Bert的Positional embedding则采用词嵌入的方式为每个位置初始化一个向量，然后**随着网络一起训练**
   - **训练任务不同**
     - transformer
     - Bert在训练过程中引入两个训练任务：
       - MLM：
       - NSP：
         1. BERT训练过程中引入了哪两种任务？分别有什么作用？

4. **BERT中的token mask与Cbow有什么异同点？**

   **相同点**：两种方式都采用了使用一个词周围词去预测其自身的模式。

    **不同点**：1.mask ml是应用在多层的bert中，用来防止 transformer 的全局双向 self-attention所造成的信息泄露的问题；而Cbow时使用在单层的word2vec中，虽然也是双向，但并不存在该问题

    2.cbow会将语料库中的每个词都预测一遍，而mask ml只会预测其中的15%的被mask掉的词

