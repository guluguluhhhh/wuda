“类 NVIDIA 架构简述”（GPGPU）
主要讨论所谓的GPGPU，这种架构有什么特点呢，我觉得就是都有 warp schedule，就是一个处理器内按 warp 来执行任务，然后通过 warp schedule 来切 warp 延迟隐藏。

GPU 编程模型
我们先从编程模型的角度开始。我们知道 GPU 可以同时操作大量的“线程”来执行程序。每个线程的计算能力有限，但是以数量取胜。以往的说法我觉得容易给第一次接触 GPU 的人带来误导，所以我把“线程”打上了引号。传统的“线程”概念的定义是操作系统能够进行运算调度的最小单位，通常在 CPU 端编程时我们对线程的理解是线程是强大且独立的，每个线程是有巨大开销且异步运行的。这可能给我们理解 GPU 的“线程”带来巨大困惑。因为 GPU 的线程在很多情况下是“锁步”的，所谓的SIMT，更多的是一种编程范式的理解，告诉你每个线程该干嘛干嘛，实际上执行的时候还是按warp（线程束）同步执行（在线程分裂的时候sub-warp).

在 GPU 编程中，不管是哪家，基本都遵循这个架构： Block -> Warp(线程束) -> Thread

这三者的关系 be like：

一个 block 中含有多个 warp， 一个 warp 中含有 32 或 64 个 thread（取决于你的硬件，甚至可以做 128thread, nvidia是32个线程一个warp）。
在一个 block 中，所有的 warp 和 thread 都可以共享数据（通过使用同一片 shared memory）。
block 和 block 之间，则只能使用 global memory（ddr）来共享数据（为便于理解，cluster不在本文讨论范畴内）。 每个 thread 有自己的寄存器空间。thread 和 thread 之间可以通过 shuffle 来交换数据（仅限一个 warp 内，这是一种特殊的硬件实现）。
而我们 GPU 实际调度的粒度呢，就是一次一个 warp，而不是一次一个 thread。相当于每个 cycle 都会发射一个或多个（取决于 Issue Array 的设计）指令，这条指令由一个 warp 内所有的 thread 共享，同步执行。

这里还没有讨论线程分裂的情况，这个有点高阶我们后面再讨论，总之你只需要理解的是，GPU 每个 block 在执行的时候是一次执行一个 warp 的 thread，即 32 个线程共享同一条指令，但是使用的是自己的寄存器。

基本架构
不管是 NVIDIA 还是某英国 IP 的芯片，亦或是国产走这条路的芯片（其实好多厂商也是买的英国 IP，不说是谁大家也清楚），大抵从一个 SOC 向下拆的话，我们可以看到这样的结构：

SOC -> CORE -> CLUSTER -> WARP_PROCESSOR -> FU

我们借用一张NV H100的架构图 （感谢评论提醒）


h系列 SM架构图
结合上文，一个 block 的任务会被分给上面这么一坨硬件，然后这个卡，它性能比较强，一次性可以发 4 个 Warp。我们先从整个这一坨开始研究，整个这一坨就对应软件的一个 block。

block 粒度

一个 block 粒度的数据是怎么同步的呢？就是通过最下面的 Shared Memory 来同步。比如左上角的 warp 和右下角的 warp，他们之间是没有硬件互联的，如果左上角的 warp 需要给右下角的 warp 一个数据，呢就必须先写数据到 shared memory，然后所有 block 内的线程做一次 sync，再由右下角的 warp 来取。

warp 粒度

我们再深入看一下，看一下里面的子模块。


在我们启动一个 kernel 的时候，会把多个 warp，都发给这么一个模块的 warp scheduler，假设我们算下来每个这一小坨需要执行 8 个 warp，呢这个模块就需要管理 8*32=256 个线程。这 256 个线程就需要共享这片 register file，所以每个线程都能分到 16384/256 = 64 个 32 位寄存器。注意，每个线程的寄存器都是自己独享的！

在执行程序的时候，warp scheduler 会指定某个 warp 运行，呢这个 warp 所代表的 32 个线程就会被拉到前台来使用所有的资源（绿色的 FU 和红色的 LD/ST 以及 SFU）。

这图里你数一数 FP32 的 function unit，正好是 32 个，正好一个 cycle 就可以吃掉 32 个线程的 Float32 指令。有人问呢为啥 INT32 和 FP64 只有 16 个，甚至 Tensor core 只有一个？FP64 和 TENSOR Core 我认为是带宽限制。具体的参数还是要看芯片的技术参数，不能从这张图里猜。这只是举个例子，我们讨论的也不仅限于 NV 的芯片。

有细心的读者这时候就会发现，既然是所有线程共享所有的寄存器文件，但是前台只有一个 warp 在运行，真实寄存器使用率只有 1/warp_num（在上面的例子中是 1/8）呢是不是有点浪费寄存器空间？为什么不用虚拟寄存器的方式来保证每个线程都能获得最大的资源？

其实这就是 GPU 多线程运行的核心。就是 warp de-schedule! 如果寄存器文件只分配给前台的 32 个线程用，呢每次切换 warp 的时候都需要保存上下文，或者运行完前台的 32 个线程所有代码后才能切后一个。

每次切换 warp 的时候都需要保存上下文的问题是开销太大，而且需要在 ddr 里单独维护一张状态表，读写开销太大。

运行完前台的 32 个线程所有代码后才能切后一个的问题是无法做到延迟隐藏，在当前代码中如果遇到某个执行时间特别长的指令，无法切换其他 warp 来做。

所以 GPU 能高效运行的本质就是通过频繁低开销地切换 warp 来实现延迟隐藏。这也就是所谓的TLP。（感谢评论提醒，已修改ILP为TLP）

补充：ILP（指令级并行）是指通过issue array将不同的指令发给不同的FU，来实现指令的并行。前两天看qserver的论文里提到一个RLP（寄存器并行）的概念，好像是和vector指令有关，感兴趣可以自己去搜一下。




Cycle0： WARP0: INST0

Cycle1： WARP0: INST1

Cycle2: WARP0: INST2

Cycle3： WARP1: INST0

Cycle4： WARP1: INST1

Cycle5： WARP1: INST2

Cycle6： WARP0: INST3

发射7条指令，只需要6个cycle。如果不做de-sechdule，就需要9个cycle。

当 GPU 检测到某个指令会特别长，而且后续的指令会被当前指令 block 的时候，就会主动发起 warp deschedule，来切另外一个 warp 来运行。这个切换的开销非常小，因为我们每个寄存器都有自己的物理寄存器空间，不需要额外做 context 切换，只要不发生 Icache miss 的话，一个 cycle 内就可以切换成功。warp scheduler 这个硬件就是做这个操作用的。

实际在架构设计中切换的时机是精心设计的，可以由 ISA 中单独设置的位域（参考 cpp 中的 yield()函数）来由软件主动发起切换，也有可能是发生 cache miss，也有可能是特定指令（LD 指令通常会发生切换，ST 一般不会，由于有 write buffer），也可能是 FU 的 slot 已经满了或是遇到了特定的 fence。这些设计每家厂家都不尽相同。

Memory Arch
Memory 是 GPU 设计的重头戏。但说实话（前方暴论发表环节），我认为 GPU 的 memory 硬件架构设计相比多核的超标量乱序处理器简单了不止一个维度！因为大部分很复杂的任务都交给了软件的同学！所以说啊，开一家 GPU 公司，最烧钱的规模最大的队伍肯定妥妥的是软件开发。设计一个 GPGPU 架构可能十几二十号人都够够的了，软件你不整个...诶哟扯远了。



I-Cache
正如上文所述，单个 GPU 的核心在同一时刻的所有线程（一个 warp）内会执行相同的代码，warp scheduler 会切换 warp，warp 和 warp 之间并不是锁步的。在切换 warp 的时候，如果发生 I-Cache miss，呢会造成极大的性能损失。实际上 warp scheduler 会提前加载好下一个被调用的 warp 的 IA（issue array），利用 ping-pang buffer 的方式可以做到无痛切换，但如果你的程序太大，一次 I-Cache miss 就有可能阻塞上千个线程的运行。所以通常我们会强烈建议软件开发的内核代码能完全载入 vendor 给定的 I-Cache 范围。开发过 cuda 的同学应该深有体会，明明不管 bank conflict 也好、memory Coalesced 也好都处理的板板正正、结果性能还是炸了，很有可能就是代码太大了，给 I-Cache 挤爆了。可能你辛辛苦苦优化的算子，处处做到极致，结果就因为代码量太多，分分钟让你的努力付之东流。不过这种情况并不多见，一般情况下预留的icache是足够你完成一个kernel的，所以我们很少对icache做perf。

D-Cache
说到 memory hierarchy，呢绕不过的问题就是 cache 一致性问题。GPU 有呢么多的 core，呢么多的线程要跑，难道所有的 cache 都要像 CPU 呢样搞 MOSI 协议来确保一致性吗？每一级的 cache 都要确保一致性吗？答案是：NO! GPU 的一致性问题是由软件通过 memory barrier 来实现的，硬件并不会保证一致性。这是 GPU cache 架构和 CPU 最显著的区别。如果 gpu 每个 cache 都要做到 coherence，硬件支出是不可接受的！另外，从 GPU 的编程模型来看，一般来说数据的空间相关性很强，用这么大的代价去做 cache coherence 不是一笔划算的交易。GPU 硬件的 L1 cache 是 incoherent 的！（L2 可以做coherent，因为L2cache层级很低）

在cuda里，需要通过memory_barrier来确保L1的一致性，最简单的办法就是__syncthreads(),它做线程同步的同时会插一个memory barrier，cuda也有别的原语能实现效果，不过我做cuda比较少啦，一般GPU公司都会有自己的memory fence指令。

So，既然L1都是incoherent的，GPU 的 D-Cache 采用的（我猜）必然都是（lazy）write-through 策略。

"写回法肯定是不能用的啦，这辈子写回是不可能写回的啦，只有 lazy write-through 这样混混日子"————窃 pseudo · LRU

小tip：所谓 lazy write-through 和 write-through 的区别便是 lazy write-through 会等一个 memory barrier 或者计时器定时才会写回，来避免一些琐碎小数据占据总线带宽。变相地使用cache去做一个transaction的储蓄池。（对transaction狠狠进行积分 1/s)

由于单个 GPU 执行的核心数量很多（一次 32threads），所以均分给每个线程的 cache line 就很小，如果所有线程重复访问同一位置会相互驱逐缓存数据，意味着 D-Cache 更多扮演的是内存操作合并 buffer 的角色（合并多个线程需要的数据），而非传统的缓存(针对每个线程提供最好的hit rate)。我换句话说，我们应该尽量把“搬运”任务和“计算”任务分开来做。以往的 CPU 的逻辑是“搬运”即“计算”，CPU 从 cache 里把数据搬出来马上就投入计算，当前线程计算需要什么数据我就去搬什么数据（所以需要很高的缓存性能来保证高频次随机访问的性能）。但是 GPU 应该是所有线程齐心协力先把数据搬上来（搬到 shared memory），然后大家在各自分工去 shared memory 里取自己需要的数据来计算（有规律地访问连续的大片数据）。最常见的例子就是矩阵计算的分块思想。

上面这段话看起来有些晦涩难懂，我举个例子，假设我们要做 128 个数的 load，我一个 cacheline 只能存 64 个数，CPU 的话直接从 0 开始 Load 就好，这样一个空 cache 的话只要加载两条 cache line 就可以完成。但是 GPU 的话，我们需要先组织一大批线程去取 cache0 的数据，拷贝到 shared memory，然后再组织这批线程去取 cache1 的数据。对于每个线程，取的数实际是 thread_id 和 thread_id + cacheline_size 的数据，并不是连续的。这个过程中，我们称其为内存操作合并。

如上文所提的“齐心协力把数据搬上来”，我们结合 GPU 的线程模型，会发现一些重要的规律来帮助我们写出更好的 GPU 代码。这意味着我们所有的访存操作要遵循一个重要的指导思想： GPU 编程中，我们不看一个线程的前后访存关系，而是看同一时间，空间上所有线程的访存关系。同一时间（同一条指令）所有线程的访存是否是连续的，没有空洞且地址对齐（对齐 cache line 大小避免跨 cacheline）。


错误示范
上图就是典型的错误示范，两条指令没有关注空间访存关系，导致两条 inst 都需要两条 cacheline 的数据，对带宽造成 100% 多余的浪费！


√
上面这种方法，每条 inst 就只需要一条 cacheline 的数据，数据从空间角度看没有空洞，这就是好的。大赞赏。

如果你玩N卡的话，Nsight compute里的memory tab有个sector/req指标，可以验证相关性能。

基于以上的出发点，也就理解了为什么会有呢么多的memory layout，什么linear，什么tiled linear，什么twiddle， 都是为了让访存可以空间上连续起来，最后多个线程的访存命令打包成一个transaction发给总线来最大程度上地节省总线带宽。

相同的逻辑也可以用来解决 bank conflect，这里不再赘述了。



看看，我们说了这么半天，结果又搞到软件设计来了。GPU的硬件真的很偷懒啊（笑 。纯属玩笑，绝无冒犯，鄙人也做了很多软件工作，GPU本身就应该是软硬结合的巅峰之作）。

Memory ISA设计
抛砖引玉：假设我们为GPU设计一个D-Cache系统，我们该注意什么呢？每一笔transaction，都有哪些要注意的？CUDA的只读缓存又是什么？

在架构设计中，我们通过LSU（load store unit）来实现读写操作和ISA的接口。ISA提供LOAD或者STORE指令来实现读写。呢我们一般怎么设计这两条指令来保证运行效率？

首先我们知道的是，LOAD,STORE指令和其他计算指令不同，他们属于运行时间不确定的指令，所以可以考虑触发de-schedule（告知scheduler撤掉当前warp，更换新的warp上来发指令）。

STORE指令一般结束的很快，因为可以设置write buffer，让线程先运行剩下的指令，让LSU慢慢处理buffer里的数据来延迟隐藏。这个outstanding设置为多少，全看各个厂家的经验。

LOAD,STORE需要规定在哪一级cache里做同步，一般我们管这个域段叫“scope”，这就是硬件提供给软件管理cache coherent的工具。scope关键词来决定这次load/store到底要用到哪一级的缓存。

LOAD,STORE需要规定某个地址的优先级，或者叫"persistent"，或者叫"cache policy/ evict policy"等等. 这个域段可以告诉cache，哪些cacheline更加地“高贵”，不可以被替代。在cache执行替换策略时常常要优先保留这些cache line。

一般还有个关键词是Memory consistency相关的，比如store release，load acquire之类的，一般这种指令最后拆到硬件实际运行的时候都是先发一条fence和ld/st的组合指令，反正GPU的issue buffer是按顺序发射的，不是乱序发射。（合在一起由lsu处理也可以，但是会增加设计复杂性）。对于store release，一般会先发一条store，然后再发fence来flush对应的cache（保证写穿）和flush write buffer到对应的scope来确保同步。后面会详细解释这一段。

Memory类型与broadcast
以cuda为例，能让软件开发者接触到的memory有这几种：

shared memory
global memory
const memory
local memory
其中global memory就是整个gpu共享的一大片片外的DDR，访问最慢，面积最大。

shared memory是一个SM共享的一片SRAM，访问相对来说比global memory快很多，毕竟离流水线近，还是sram。

const memory比较特殊，一般来说会有专用的缓存和广播（broadcast）硬件。一般和主存放在一起，层级和global差不多，但是由于有独特的访存机制，所以还是会比global快。

local memory是每个线程独享的DDR，你可以理解为发生register spill的话就会借用DDR来存。我们尽力避免这种情况发生。



这里重点讲讲broadcast，什么是broadcast？broadcast就是当你多个线程去读同一个地址（极端一点，32个线程读同一个地址），你不需要连续发32个transaction，而是发一个transaction，从总线读回来之后呢，通过专门设计的broadcast电路，来共享这个数据。所谓broadcast电路，其实就是一路数据线分裂成32路再加一点控制电路，没什么特殊的。

纹理内存
有一些CUDA教程里会说建议大家用纹理内存，因为纹理内存有broadcast能力巴拉巴拉的。这里解释下什么是纹理内存。纹理内存是原先给图形学用的，用硬件来加速图形学纹理采样的一些计算，纹理内存和global memory其实没有区别，一般就是多了几个descriptor，本质区别是用纹理内存会走专门的纹理指令（TEX），而不是load/store,有专门对应的纹理指令的硬件。

一般来说处理纹理指令的硬件会有专门的合并机制和广播单元。所以走纹理指令一定程度上可以提升读取的性能。

我觉得其实最大的区别还是走纹理会多个cache，所以性能会比较好吧，纹理模块也能直接支持一些swizzle或者改变memory layout之类的操作，可以省去一些计算index的指令，但是一般我看很少人用这部分来优化（因为随着硬件发展，LSU一般也会支持这些操作）。

随着现在技术发展，纹理内存用的也少了，现在都是直接异步DMA去搬数据了，比这个不知道高到哪里去。但不是说纹理模块就要被LSU在计算任务中彻底淘汰了，纹理模块也在不停进化，纹理模块有免费的filter硬件资源，对于一些需要插值的2D/3D数据，可以走纹理内存白嫖一下硬件。只是在attention里是很难想到有什么用处。

纹理模块是图形学里非常复杂的模块，感谢架构的进步，大家纯做计算的话不太需要关注这些，CUDA一般封装的也比较好了。后面有机会地话再把纹理单独拎出来说说把（另开新坑）。

访存指令实战解析
从现在开始，我们要上点强度了。我将拿CUDA的PTX给大家解析一下访存指令都干了什么。不知道什么是PTX的请去拷打一下AI或者RTFM。

我们先写了个十分简单的cuda妙妙小程序：

__global__ void loadtest(volatile float* data){
    float m;
    m = data[blockIdx.x * blockDim.x + threadIdx.x];
}
这里加了个volatile主要是防止编译器优化，给我直接把函数优化没了。然后我们来看看nvcc生成的ptx代码（我是4060Ti）

.visible .entry _Z8loadtestPVf(
	.param .u64 _Z8loadtestPVf_param_0
)
{
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<5>;

	ld.param.u64 	%rd1, [_Z8loadtestPVf_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.u32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.volatile.global.f32 	%f1, [%rd4];
	ret;
}
我们看到哈，简简单单一个load，竟然编译出这么多代码。到底编译器干了什么呢？

ld.param.u64 %rd1, [_Z8loadtestPVf_param_0]; 这一步是从param加载参数到寄存器里。这个param字段是什么意思，就是kernel的传参。一般硬件里会有一个模块在运行kernel之前把参数传到特定的位置。

cvta.to.global.u64 如果你狠狠RTFM过的话，你会知道这个指令会返回一个global的地址。呢为什么要有这一步呢，是因为一开始传进来的rd1是一个general的指针。这一步不一定触发MMU（MMU设计的时候一般会留一个touch语义，就是只蹭蹭（x），不用真的发起transaction，主要目的是更新TLB，让TLB去prefetch）。

再后面依托全是计算索引，我们无需在乎。

ld.volatile.global.f32 %f1, [%rd4]; 直到运行这一行，我们算正式地运行了ld。其中global就是指从global的DDR里读，volatile是指编译器不可以优化这一条指令（使用寄存器缓存，删除或重排）。重排这里不解释了，在CPU的设计中也常提到指令重排，作为基本知识不再赘述。



好了，上面只是小试牛刀，我们现在来点强度。之前提到过__ldg()这个函数对吧，我们调一下看看ptx写的啥

m = __ldg(&data[blockIdx.x * blockDim.x + threadIdx.x]);

============== 【PTX】 ================================
ld.global.nc.f32 %f1, [%rd2];
可以看到，多了个.nc，在cuda里就是non-coherent caching的意思。这个.nc总之就是给cache的一个hint，让cache提前做一些动作，或者告诉cache怎么做同步啊，是什么memory model 啊之类的。这里的意思就是不需要在乎一致性，因为是const数据，不会被改动，所以可以走快一点的缓存。（猜测可能是纹理缓存，因为纹理缓存大概率是只读的）

具体其他语义大家请自行阅读PTX文档。总之大家知道，访存命令是有这种域段，可以供用户决定某个数据要不要进cache，进哪个cache，甚至微操cache。（“机枪阵地向左移动5米”）。



STORE指令同理，这不展开了。感兴趣的同学可以查查PTX文档。

Memory consistent Model
这段内容其实和CPU的没什么太大区别，主要有以下几类：

Sequential consistency 最强的约束，这个命令的前后访存指令不可重排，并且要保证可见性。
Relaxed 最弱的约束，通常这个约束只保证atomic操作，指令可以随意重排。
Acquire-Release 最常用的弱同步模式。
上面三种在GPU里都比较常见，一般情况下是写给compiler看的，具体这三个语义是什么含义我就不详述了，互联网上有很多对他们的介绍。我这里讲讲gpu里一般会怎么做。

假设我们有一条指令长这个样子：

store.global.release
这是一条store到global memory的指令，带release语义，呢在GPU中会执行什么操作呢？

GPU会先把这一条指令压到write buffer里，然后再紧接着发一条fence，所以这一条会被拆成两条指令（这里写了三条，后面两条一般是合成一条，这里为了解释明白逻辑给拆了）

store.global            // 压到write buffer里，有没有占用总线发生写操作暂不保证
fence.global            // 发一条fence，保证write buffer全部操作完
drain cache             // 保证l1和l2的cache line 全部写回
相似的，还有load acquire，假设有一条load是这样：

load.global.acquire
compiler会拆成以下步骤：

invalid old cache line
load.global
先无效化对应的cache，来保证一致性。然后再从global拿数据。

如果load.acquire的时候l1内部有脏数据怎么办呢？事实上这种情况在GPU内是不被允许的。GPU的指令都是顺序发射，如果出现这种情况，一般是代码的问题或者compiler进行了错误的重拍。视作undefined behavior。



cuda里经常用的__threadsync()和__threadfence()，一般都是带一个SC或者rel-acq约束。

我们接下来结合ptx的一段example来理解【上强度了】

tensormap.replace.tile.global_address.global.b1024.b64   [gbl], new_addr;            // T0
fence.proxy.tensormap::generic.release.gpu;                                          // T1
cvta.global.u64  tmap, gbl;
fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;                              // T2
cp.async.bulk.tensor.1d.shared::cluster.global.tile  [addr0], [tmap, {tc0}], [mbar0];// T3
没玩过PTX的朋友们你们还在嘛？不要害怕，我们一点点解释，这里不是所有东西你都需要现在就弄明白。

这是一段fence指令的example，我们慢慢一行一行看，我们要从两个角度切入，一个是内存可见性（针对硬件），一个是执行顺序（针对编译器）

第一行：

tensormap.replace.tile.global_address.global.b1024.b64   [gbl], new_addr;            // T0
借用tensormap proxy（理解为一套硬件的处理逻辑，cuda里针对读写有不同的逻辑，有generic，有tensormap，还有async以及alias，这些处理逻辑是由不同的硬件完成的，这不是fence的重点）来把new_addr这个数据存到gbl地址里去，这是一个global 操作。（内存可见性尚未同步到global）

第二行：

fence.proxy.tensormap::generic.release.gpu;           // T1
插入一个release fence。这一行的意思就是之前用tensormap(proxy)做的到global memory（scope = gpu）的写指令（store release）必须全部完成。在这一步就会触发所有global以上的内存架构的写回以及清空写buffer的操作（限定了tensormap proxy发起的）。 （内存可见性同步到global）

前两行确保了 T0 在 T1 之前执行。（确保执行顺序）

第三行不用管，就是个指针变换操作。

第四行：

fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;              // T2
插一个acquire fence，意思是generic proxy的读指令在这一行运行结束后要确保数据能读到最新的，其实就是invalid了cache。这一行添加了一个变量[tmap]，其实就是告诉L0(maybe存在)，L1和L2cache：“都给我找一下有没有这个人，有的话把对应的cacheline invalid掉！等下generic proxy过来读数据不要给我鬼头鬼脑！”（内存可见性清空脏数据，确保直接看到global）

第四行和第二行决定了 T1 在 T2之前执行。因为release-acquire语义需要保前后顺序。（确保fence间执行顺序）

第五行：

cp.async.bulk.tensor.1d.shared::cluster.global.tile  [addr0], [tmap, {tc0}], [mbar0];    // T3
调用generic proxy来读数据，这样就确保了这一行运行的时候读的数据是最新的。

ptx插入这些不管是为了触发写回或者清cache，也是告诉编译器，不要乱给我排指令顺序啊，但也不是不排，是循序渐进地排，有计划地排，辩证地排，至少我release之前的store指令不可以排到fence.store之后， acquire之后的读指令不可以排到fence.acquire之前！（内存可见性：能直接看到global，从global中取正确的数据）

第四第五行确保了 T2在T3之前执行，最后就达成了我们的目的，T0在T3之前执行。（确保执行顺序）

至于其中的单向双向proxy（tensormap::generic这块），这里就不细讲了，感兴趣的可以自行阅读文档。

然后就有好奇宝宝要问了，啊，我看这个第三行你刚没讲的，呢不是有个tmap，被第五行依赖嘛，这个跟fence有关系嘛。

答曰：没关系啊，没关系，可以简单理解为这是寄存器的操作啊，fence是memory读写相关的操作，跟这没关系的啊。寄存器之间的依赖关系是pipeline硬件来确保顺序的，不要搞混概念哈。后文会说说寄存器依赖怎么解决啊。（scoreboard）

Bank Conflict
bank conflict是大家在写cuda的时候经常遇到的问题，为什么会发生bank conflict呢？
物理上，我们会用sram来储存数据，单端口的sram一个cycle可以处理一个地址的读/写。在一个cycle中，对同一个sram出现了两个不同地址的读/写请求，硬件会分两个cycle去完成（2-way bank conflict）。

在编程的时候，如果我们访问shared memory时（一般我们都是在解决shared memory bank conflict），我们的访存合并没有做的很好，就会发生bank conflict（这种情况有时候是算法相关，无法避免的）。shared memory一般是分32个bank，每个bank处理32bit数据，如果一个warp访问连续的地址，呢一般不会发生bank conflict。

解决bank conflict 其实非常非常简单！我们常用的两种方法就是padding或者swizzle！

padding就是申请空间的时候多申请一串空的，用多余的空间去做address interleaving。是非常方便的一种避免bank conflict的方法，缺点就是会浪费shared memory空间，导致occupation下降（maybe）。而且解决bank conflict的能力不足，不是所有的情况都能通过padding简单实现。

这里我们重点讲讲swizzle，swizzle就是更换memory layout来实现bank conflict free，不会增加空间的使用，但是需要增加一点点地址索引的计算。

我给一个二维矩阵的swizzle万能公式：

__shared__ TYPE mem[size_y][size_x];

int x;
int y;
mem[y][x ^ (((y >> a) & b) << c)]
里面的a,b,c都是参数，来指定如何进行swizzle。

简单思路哈，异或操作有一个很明显的性质，就是按mask进行取反。

A ^ mask 的操作，就是把A中对应mask为1的元素进行取反。

我们简单举个例子，如果上面的a = 0b0, b = 0b1, c = 0b0;

呢swizzle之后的数据是这样的（假设 
 ）：

u0	1	2	3
5	4	7	6
8	9	10	11
13	12	15	14
我们发现，当y是奇数的时候(y & 0b1 = 1)，x会按奇偶进行交换(x[0] ^ 1 = !x[0])。当y是偶数(y & 0b1 = 0)的时候，x会按原顺序排序(x[0] ^ 0 = x[0])。

这里我直接给出a,b,c的含义。

a: log2(y方向的最小单元大小） e.g.（1=>0, 2=>1,4 =>2）
b: swiz复杂度掩码
c: log2(x方向的最小单元大小) e.g.（1=>0, 2=>1,4 =>2）
晚点我上点图给大家理解一下，然后讲一下CUTLASS里面的SWIZZLE和这里的a,b,c怎么对应。

GPGPU关键微架构单元
我这里把这一章改名了，是因为涉及到可能讲一些tensorcore相关的部分。关于gpu的指令单元也没什么特别的。

warp scheduler
讲讲warp scheduler。这个东西在文章一开头的基本架构就介绍过了，这边我们再深入看看。

warp scheduler是什么？它是GPU的指令发射单元，是一块管理了所有指令的发射以及调度，管理scoreboard以及各种barrier的电路。如果用CPU里经典的五级流水线来看，这个模块应该介于取指和译码中间。

warp scheduler主要负责不同warp的调度，它会检查当前运行中的warp是不是处于阻塞状态，其他warp有没有蓄势待发准备中的。如果当前warp被认为是处于阻塞状态（由于scoreboard卡住一段时间，或者指令里显式的hint），则会把当前的warp转到后台，调一个可用的新的warp上来执行它的指令。

在老黄的gpu里，一个SM通常有4个warp scheduler，每个warp scheduler对应一个warp的话，4个warp scheduler管控的warp正好对应一个warpgroup（从hopper开始warpgroup的概念开始凸显）。

warp scheduler最主要可以解决什么问题呢？本质上就是为了解决scoreboard stall的问题。当你有两条连续的指令，而前后两条指令存在data hazard的时候，GPU无法像cpu呢样通过各种奇技淫巧（寄存器重映射，data forward）来解决问题，所以推出了warp scheduler，来直接切换指令队列来解决问题（既然解决不了问题，就解决提出问题的线程...）。就好比说warp scheduler是你领导，然后你和你的同事小张都有活要用一下同一台桌子（硬件资源）。你的第一个活是打开烧水壶，第二个活是等烧水壶烧开后倒水。在你打开烧水壶后，领导不会让你白占桌子，而是把你支走，让小张先来桌子上干他的事，等水开了，再把你叫回去。（如果你要问小张的任务万一也是倒水怎么办？呢就叫小王小李小白菜来干活，要是都忙不开，呢就会产生stall了）

warp scheduler发射指令的效率通常是我们关注程序性能的重要参考指标。

tensor core
关于tensorcore的由来众说纷纭，鄙人入行较晚，不曾了解tensorcore具体为何而生，以下为个人推测：

what is tensor core？ tensor core就是专门用来计算矩阵乘法的一块电路，在tensorcore出现之前，都是需要大量的fma（浮点数乘加）指令来计算矩阵乘法，大家就会发现用fma的话，一个矩阵乘需要的指令数量实在太大！一个MN*NK的矩阵，共需要MNK次fma操作，这也便罢了，但是这MNK次fma操作的acc会share 相同的寄存器，这就会产生大量的寄存器依赖问题（data hazard，主要出现的是acc寄存器的WAW）。

单单靠warp scheduler是没办法完美掩盖如此大量的寄存器依赖导致的，所以干脆就做一片硬件，这块硬件直接用硬件流水线来把矩阵乘法给pipe起来，于此同时还能解放通用ALU的资源，让他们overlap去做别的事。



++2026.4.27更新。 感慨万千，有的时候觉得做芯片架构真是个吃翔的活。