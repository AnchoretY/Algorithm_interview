# RNN

#### LSTM结构以及内部参数个数计算

> LSTM主要通过引入门机制来控制遗忘，防止梯度下降的出现（因为在门中记忆是线性叠加的），其核心是遗忘门、输入门、输出门。
>
> - **遗忘门**
>
> 作用:决定哪些信息从之前的记忆中去除掉
>
> ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.el1ixqa3scc.png)
>
> ​                    
>
> - **输入门**
>
> 输入门主要包含三步：1.Tanh激活函数将t时刻的输入和t-1时刻的输出合并值转到（-1，1）作为当前备选的更新内容 2.Sigmod激活函数将t时刻的输入和t-1时刻的合并值转化成（0，1）作为输入的遗忘参数 3.将1得到的备选更新内容与遗忘参数进行点乘，获得输入。
>
> 输入门作用：决定新的输入中哪些信息将保存到记忆中
>
> ![image-20220323215048378](/Users/yhk-home/Library/Application Support/typora-user-images/image-20220323215048378.png)
>
> 
>
> 
>
> 这里不属于输入门，t时刻的单元状态为：（忘记该遗忘的，记住该记住的）
>
> ![image-20220323215117218](/Users/yhk-home/Library/Application Support/typora-user-images/image-20220323215117218.png)
>
> 
>
> - **输出门**
>
> 输出门作用:决定输出什么
>
> ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.o3p36e1vfx.png)
>
> 
>
> **注意：当LSTM网络很深并且仍然使用tanh函数仍然会出现梯度消失**
>
> **参数个数**：是一般CNN的4倍，因为多了三个线性变换。例如LSTM 输入维度为 x_dim， 输出维度为 y_dim，那么参数个数 n 为：
> $$
> n = 4 * ((x_{dim} + y_{dim}) * y_{dim} + y_{dim})
> $$

### 
