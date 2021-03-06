# TCP与UDP

#### 三次握手过程

> 1. 客户端发送一个包含SYN标志的TCP报文，SYN即同步(Synchronize)，同步报文会指明客户端使用的端口以及TCP连接的初始序号;
>
> 2. 服务器在收到客户端的SYN报文后，将返回一个SYN+ACK(即确认Acknowledgement)的报文，表示客户端的请求被接受，同时TCP初始序号自动加1;
>
> 3. 客户端也返回一个确认报文ACK给服务器端，同样TCP序列号被加1。
>
> ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ohseipvl8m.png)
>
>   经过这三步，TCP连接就建立完成。TCP协议为了实现可靠传输，在三次握手的过程中设置了一些异常处理机制。**第三步中如果服务器没有收到客户端的最终ACK确认报文，会一直处于SYN_RECV状态，将客户端IP加入等待列表，并重发第二步的SYN+ACK报文。重发一般进行3-5次，大约间隔30秒左右轮询一次等待列表重试所有客户端。另一方面，服务器在自己发出了SYN+ACK报文后，会预分配资源为即将建立的TCP连接储存信息做准备，这个资源在等待重试期间一直保留。**更为重要的是，服务器资源有限，可以维护的SYN_RECV状态超过极限后就不再接受新的SYN报文，也就是拒绝新的TCP连接建立。

#### 四次挥手过程

> 第一步：客户端构建一份特殊的 TCP 报文，该报文首部字段 FIN 被置为 1，然后发送该报文。
> 第二步：服务端收到该特殊的 FIN 报文，于是响应客户端一个 ACK 报文，告诉客户端，请求关闭的报文已经收到，我正在处理。
> 第三步：服务端发送一个 FIN 报文，告诉客户端，我将要关闭连接了。
> 第四步：客户端返回一个 ACK 响应报文，告诉服务端，我收到你刚才发的报文了，我已经确认，你可以关闭连接了。

#### 为什么要保证三次握手？为什么不是两次？

> 主要针对下面这种情况：当A发出一个连接请求报文并没有丢失而是在某些网络节点长时间滞留了，以导致连接释放后的某个时间点才到达B。本来这应该是一个失效的报文，但是B收到这个报文后，会误认为A又发起了一次新的请求，于是又向A发送确认报文，如果采用两次握手，随着B发送确认报文，连接就已经建立了。
>
> 而采用三次握手，由于A并没有发出连接请求，因此不会理会B的确认报文，因此不会向B发送确认报文，因此连接不会建立。



#### 为什么连接的时候是三次握手，而关闭的时候却是四次挥手？

> 因为在连接时，Sever端接收到Client的SYN请求后，可以直接发送SYN+ACK报文，其中SYN是用来同步的，ACK使用来应答的。
>
> 但是在关闭连接时，**当Sever端收到FIN请求时，并不一定立即关闭socket，所以只能先回答一个ACK报文，告知Client你的请求我已经收到了。只有等到Sever端自己的报文也全部发送完，才能发送FIN报文，告知Clent可以断开连接了**



#### TCP协议中MSS是控制什么的？

> ​	MSS选项用于在TCP连接建立时,收发双方协商通信时**每一个报文段所能承载的最大数据长度**。

#### TCP如何避免拥塞控制

> **慢开始**和**拥塞避免**
>
> ​	慢开始：在最开始时收到确认一个RTT拥塞窗口扩大一倍
>
> ​	拥塞避免：在达到慢开始门限后采用拥塞避免，一个RTT拥塞窗口扩大1
>
> ​	1.当cwnd<ssthresh，使用上述的慢开始算法
>
> ​	2.当cwnd>ssthresh，停止使用慢开始，使用拥塞避免算法
>
> **快重传**
>
> ​	当接收方收到了一个失序的报文，马上报告给发送方，我没收到，赶紧重传（***\*天下武功唯快不破\****）
>
> ​	假如M2收到了，M3没有收到，之后的M4,M5,M6又发送了，此时接收方一共连续给发送方反馈了4个M2确认报文。那么快重传规定，发送方只要连续收到3个重复确认，立即重传对方发来的M3
>
> **快恢复**
>
> ​	1.当发送方连续收到三个重复确认，执行乘法减小，ssthresh减半
>
> ​	2.由于发送方可能认为网络现在没有拥塞，因此与慢开始不同，**把cwnd值设置为ssthresh减半之后的值，然后执行拥塞避免算法，线性增大cwnd**

 

#### 客户端出现异常时TCP如何避免死连接？

> 通过保活计时器来避免出现过多的死连接，如果2个小时（默认值）没有收到客户端的数据，那么服务器端会发送探测报文，以后没75分钟发送一次，如果超过10次Client都没有响应，那么服务器端就认为客户端出现了故障，接着关闭这个连接

#### MSL、TLL、RTT的区别

> **MSL**：是Maximum Segment Lifetime英文的缩写，中文可以译为“报文最大生存时间”，他是任何报文在网络上存在的最长时间，超过这个时间报文将被丢弃
>
> **TTL**：ip头中有一个TTL域，TTL是 time to live的缩写，中文可以译为“生存时间”，这个生存时间是由源主机设置初始值但不是存的具体时间，而是存储了一个ip数据报可以经过的最大路由数，每经 过一个处理他的路由器此值就减1，当此值为0则数据报将被丢弃，同时发送ICMP报文通知源主机。
>
> **RTT**是客户到服务器***\*往返\****所花时间

​	

#### 为什么Client在Time-wait状态时要等待2MSL的时间？

> （1） 保证Clent发送的最后一个ACK报文能够到达B
>
> （2） 不会出现因为滞留而产生的上次上次连接请求报文。因为Clint在最后一个ACK后，经过2MSL的时间后，就可以使本连接中的所有报文在网络中都消失。



#### TCP/UDP的区别

> 1、TCP面向连接（如打电话要先拨号建立连接）;UDP是无连接的，即发送数据之前不需要建立连接
>
> 2、TCP提供可靠的服务。也就是说，通过TCP连接传送的数据，无差错，不丢失，不重复，且按序到达;UDP尽最大努力交付，即不保证可靠交付
>
> 3、TCP面向字节流，实际上是TCP把数据看成一连串无结构的字节流;UDP是面向报文的
>
> UDP没有拥塞控制，因此网络出现拥塞不会使源主机的发送速率降低（对实时应用很有用，如IP电话，实时视频会议等）
>
> 4、每一条TCP连接只能是点到点的;UDP支持一对一，一对多，多对一和多对多的交互通信
>
> 5、TCP首部开销20字节;UDP的首部开销小，只有8个字节
