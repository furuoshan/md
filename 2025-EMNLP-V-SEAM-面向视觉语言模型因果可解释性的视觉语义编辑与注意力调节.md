# V-SEAM 面向视觉语言模型因果可解释性的视觉语义编辑与注意力调节

**原题**  
V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision-Language Models

**作者**  
Qidong Wang¹、Junjie Hu²、Ming Jiang²*  
¹同济大学，²威斯康星大学麦迪逊分校。*通讯作者。  
wang_qidong@tongji.edu.cn，{junjie.hu, ming.jiang}@wisc.edu

**发表信息**  
Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing（EMNLP 2025），17396–17420 页，2025 年 11 月 4–9 日，©2025 Association for Computational Linguistics。  
[论文页面](https://aclanthology.org/2025.emnlp-main.880/)  
[数据与代码](https://github.com/petergit1/V-SEAM)

> 译文依据用户提供的 `2025.emnlp-main.880.pdf`，覆盖正文及附录 A–I。章节、图表和公式沿用原文编号，双栏转为 Markdown 单栏。图内英文、模型输出词元及注意力头编号便于与原文对照，图注和表格说明译为中文。参考文献保留英文。模型的原始二元输出标签 Yes、No 分别对应“是”“否”。原文不同表格中的汇总值差异及个别编号清单疑点，在相应位置作译注。

## 摘要

因果可解释性研究近期从语言模型扩展到视觉语言模型（VLM），试图通过输入干预揭示内部机制。文本干预常针对语义，视觉干预通常依赖粗粒度像素扰动，因此难以深入理解多模态整合中的语义机制。

本文提出 V-SEAM，将**视觉语义编辑**与**注意力调节**结合，用于 VLM 的因果解释。V-SEAM 支持概念级视觉操作，并在物体、属性、关系三个语义层面，识别对预测具有正向或负向贡献的注意力头。我们发现，正向头通常在同一语义层面共享，但跨层面存在差异，负向头则往往具有更广泛的泛化性。最后，我们提出自动调节关键头嵌入的方法，在三个不同 VQA 基准上提升了 LLaVA 和 InstructBLIP 的表现。数据与代码已在上述仓库公开。

## 1 引言

视觉语言模型已成为多模态理解和生成的重要基础设施，支持视觉问答（VQA）（Liu et al., 2023，Dai et al., 2023，Liu et al., 2024）、图像描述（Zhang et al., 2021，Wang et al., 2024）和导航（Anderson et al., 2018，Zhou et al., 2024）等下游应用。尽管性能出色，内部机制仍研究不足，影响现实部署的可信度。为填补这一缺口，研究者日益关注 VLM 解释，特别是因果干预，因为它既能因果性地揭示模型行为，也能为模型编辑等改进提供系统路径（Zhao et al., 2024，Lin et al., 2025）。

现有因果可解释性主要研究大语言模型（Vig et al., 2020，Meng et al., 2022，Geva et al., 2023，Zhao et al., 2024）。近期研究开始同时扰动视觉与文本输入，将其应用于 VLM（Palit et al., 2023，Basu et al., 2024，Golovanevsky et al., 2025）。文本干预如词元替换，通常聚焦细粒度语义变化（Basu et al., 2024，Zhang and Nanda, 2023），而整图高斯噪声（Palit et al., 2023）和随机视觉遮挡（Neo et al., 2024）等视觉干预，通常造成像素层面的整体破坏，见图 1。

![图 1 视觉干预比较](assets/VSEAM-fig01.png)

**图 1 视觉干预比较示例。** 比较视觉非语义干预与语义干预。原图、高斯噪声、遮挡、Stable Diffusion 和 V-SEAM 对颜色、物体及动作变化产生不同影响，示例包括白色衬衫变蓝、电视变时钟、吃东西变坐着。

Zhang and Nanda（2023）表明，这种干预差异可能产生不一致甚至误导性的机制解释，为 VLM 分析带来不确定性。最新研究尝试通过检索相似图像对，或以文本指导 Stable Diffusion 生成对抗图像，实现语义视觉干预（Golovanevsky et al., 2025）。但如图 1 所示，这仍可能意外改变背景或主体，为分析引入无关干扰。

我们提出细粒度、基于语义的 V-SEAM。理想干预应只改变与特定语义属性相关的图像组成部分，据此提出视觉语义编辑，在物体、物体属性、物体间关系三类语义上操作输入。随后提出注意力调节，识别各语义层面驱动预测的关键头，并调节其嵌入以改进模型。

在 VQA 中分析 LLaVA 和 InstructBLIP，得到四项发现。

1. VLM 通常在较早层，如第 5–10 层，侧重物体级跨模态对齐，随后在第 9–15 层转向属性和关系。
2. 注意力模块主要提取颜色等显著语义线索，MLP 模块对最终决策起关键作用。
3. 存在促进正确预测的正向头，以及引入误导信号的负向头。正向头通常在同语义类别内共享，但跨类别不同。负向头往往跨语义泛化。
4. 对关键注意力头嵌入重缩放，可在三个不同基准上提高两种模型的 VQA 准确率。

主要贡献如下。

- 提出机制可解释性框架 V-SEAM，通过概念级视觉语义编辑与注意力调节，因果追踪 VLM 在物体、属性、关系理解中的关键组件。
- 揭示 VLM 如何处理视觉和文本中的细粒度语义信息。
- 通过嵌入重缩放增强正向头、抑制负向头，改进 VQA。三个不同视觉问答任务上的实验显示持续提升。

## 2 相关工作

### 视觉语言模型

按视觉与文本整合策略，现有 VLM 大致分两组。一组通过交叉注意力将视觉特征嵌入文本特征（Alayrac et al., 2022，Li et al., 2023a）。另一组通过投影层映射视觉与文本特征（Liu et al., 2023，Bai et al., 2023，Zhu et al., 2024，Chen et al., 2024，Li et al., 2024）。投影式模型通常性能更强，因此既有解释工作主要聚焦该组（Bai et al., 2023，Liu et al., 2023）。为提供更全面的因果解释，本文研究两组模型。

### 多模态可解释性

现有工作可分为互补的数据驱动方法（Sood et al., 2023，Xing et al., 2025）和面向模型结构的分析。前者通常通过显著性技术，将像素或视觉词元等输入区域与输出关联，揭示影响最强的输入。后者检查内部组件和机制，包括跨层层次表征（Palit et al., 2023，Zhang et al., 2024）、模块功能（Ben Melech Stan et al., 2024，Basu et al., 2024，Chen et al., 2025）、注意力头贡献及专业化（Golovanevsky et al., 2025），以及单神经元或神经元组行为（Gandelsman et al., 2023）。本文属于结构分析，同时关注层次表征和注意力机制。

结构分析又分两条路线。**相关性解释**诊断不同层与组件编码哪些语义。常用探针，在隐藏状态上训练分类器解码语义属性（Alain and Bengio, 2016，Hendricks and Nematzadeh, 2021），但易受探针设计和伪相关影响，遮蔽真实表征（Belinkov, 2022，Bilodeau et al., 2024）。TextSpan（Gandelsman et al., 2023）和 Logit Lens（Neo et al., 2024）将神经元或视觉词元与离散、可理解概念对齐，提供直接内部视角。MINER（Huang et al., 2024）和 MMNeuron（Huo et al., 2024）识别对任务或语义选择性激活的神经元。但神经元多义性，即单神经元编码多种纠缠概念，限制了解释能力（Liu et al., 2025）。稀疏自编码器在解耦 LLM、ViT 表征上有潜力（Cunningham et al., 2023），应用于 VLM 时仍面临模态不一致和不稳定。

**因果追踪**研究输入干预对组件的因果影响（Vig et al., 2020，Geva et al., 2023）。Basu 等（2024）发现早层编码事实知识，Palit 等（2023）强调深层作用，Golovanevsky 等（2025）分析模态专属头。然而，整图替换、噪声等粗干预会引入伪影，遮蔽语义因果性（Zhang and Nanda, 2023）。本文提出以视觉为中心的框架，结合图像区域语义操作和图文词元激活修补，分析属性、物体、关系如何跨层、跨模态处理和对齐。进一步通过头级干预识别语义敏感的正、负向头，扩展已有模态专属注意力分析。

## 3 预备知识

本节定义数据与模型符号，并在 VLM 背景下介绍激活修补，又称因果追踪（Meng et al., 2022，Golovanevsky et al., 2025）。

### 数据与模型

考虑包含 $N$ 个问题—图像—答案三元组的 VQA 数据集 $\mathcal D$。每组包含关于图像 $z$ 的问题 $x$，以及单词元答案 $y$。每幅图像有物体列表 $B$，记录文本标签、边界框坐标及物体关系边。

基于 Transformer 的 VLM 记为 $f_\theta$，参数为

$$
\theta=\{\theta_{\mathrm{in}},\theta_{\mathrm{out}}\}\cup\{\theta^l_{\mathrm{att}},\theta^l_{\mathrm{mlp}}\}_{l=1}^{L}.
$$

其中包含视觉与文本输入编码层、输出投影层，以及 $L$ 个分别由自注意力和 MLP 组成的 Transformer 层。正确答案 $y$ 对输入 $(x,z)$ 的输出 logit 记为 $\ell(x,z,y)\in\mathbb R$。

### VLM 中的激活修补

给定 $(x,z,y)\in\mathcal D$，先将干净图像 $z$ 修改为受扰图像 $\widetilde z$。独立输入 $(x,z)$、$(x,\widetilde z)$，分别得到自注意力与 MLP 后的逐层嵌入 $H^l_\tau,\widetilde H^l_\tau\in\mathbb R^{T\times d}$，$\tau\in\{\mathrm{att},\mathrm{mlp}\}$，$T$ 为输入序列长，$d$ 为隐藏维数。正确答案对应 logit 分别为 $\ell(x,z,y)$ 和 $\ell(x,\widetilde z,y)$。在图文联合输入中，受扰词元的索引集记为 $\mathcal I$。

按 $\mathcal I$，用干净嵌入中的对应值替换受扰嵌入，得到

$$
\widehat H^l_\tau\leftarrow\operatorname{Patch}(H^l_\tau,\widetilde H^l_\tau,\mathcal I).\tag{1}
$$

将其通过剩余层，得到修补后正确答案 logit $\widehat\ell^l_\tau(x,\widetilde z,y)$，再计算与受扰输出之差

$$
\Delta\ell^l_\tau(x,\widetilde z,y)=\widehat\ell^l_\tau(x,\widetilde z,y)-\ell(x,\widetilde z,y).\tag{2}
$$

较大正值表示，修补第 $l$ 层模块 $\tau$ 后的嵌入有助于正确预测，可据此识别关键模块。为提高估计稳定性，对全数据集取平均，得到最终因果分数

$$
s(\tau,l)=\frac{1}{N}\sum_{(x,z,y)\in\mathcal D}\Delta\ell^l_\tau(x,\widetilde z,y).\tag{3}
$$

## 4 V-SEAM

图 2 展示两阶段框架。第一阶段对局部区域进行视觉语义编辑，再逐层修补激活，识别 VLM 视觉语义理解中的关键模块及其功能，见第 4.1 节。第二阶段进行注意力调节，识别正向或负向影响理解与预测的关键头，并通过重缩放实施针对性干预，见第 4.2 节。

![图 2 V-SEAM 框架](assets/VSEAM-fig02.png)

**图 2 语义层面的因果可解释性框架。** 从属性、物体、关系区域的语义指导扰动出发，经过视觉语言模型推理，最终通过激活修补和注意力头消融分析因果作用。

### 4.1 视觉语义编辑

既有视觉干预通常采用缺乏语义精度的粗粒度扰动，难以隔离颜色、物体身份或空间关系等元素的因果贡献。强加性扰动还可能将输入推离训练分布，造成不可靠或误导性归因（Zhang and Nanda, 2023）。我们只编辑与问题语义对应的图像区域，结合激活修补，评估预测能否因果性恢复，从而追踪语义信号跨层、跨模块传播。

**语义编辑提示词生成。** 改变属性、物体或关系，构造原问题的反事实版本。人工设计 10 个参考示例，用 GPT-4o 生成符合上下文的语义替换，例如红变绿。全部人工核验，以确保语义清晰、事实准确。不到 1% 因“明亮”作为颜色等抽象、难以可视化的描述被过滤。通过核验的问题用作编辑提示，定位对应区域并实施预期语义变化。完整提示见附录 A.1。

**语义区域编辑。** 从数据集标注或现成物体检测模型获取目标区域边界框，使用 SAM 分割（Kirillov et al., 2023），再以生成的反事实提示指导 PowerPaint 局部编辑（Zhuang et al., 2024）。人工检查后，约 10% 因目标外意外改动或编辑失败被过滤。

**VQA 三元组筛选。** 为保证干预具有因果效果，仅保留模型在干净输入上预测正确、对应编辑输入上预测失败的三元组。所选样本须满足

$$
f_\theta(x,z)=y,\qquad f_\theta(x,\widetilde z)\ne y.\tag{4}
$$

这里将预测与给定正确答案比较，该筛选使观测到的因果效应显著且有意义。

### 4.2 注意力调节

激活修补能识别因果性重要的层和词元范围，但头级分析的粒度与效率受限。逐头逐词元修补成本高，且此类方法更擅长识别有益结构，可能忽略妨碍视觉理解的组件。我们提出头级因果注意力调节，高效识别关键头并针对性重缩放。

**关键注意力头识别。** 第 $l$ 层第 $h$ 个头的输出嵌入记为 $A^l_h\in\mathbb R^{T\times d_h}$，$h\in[1,H]$，$l\in[1,L]$，$d_h$ 为头维数。以同层其余头的平均输出替换该头，实现遮蔽

$$
\widetilde A^l_h=\frac{1}{H-1}\sum_{\substack{h'=1\\h'\ne h}}^H A^l_{h'}.\tag{5}
$$

该操作保持层的整体结构，同时遮蔽头 $h$ 的语义贡献。将替换嵌入连同其余头通过剩余模型，得到正确答案概率 $p_\theta(y\mid x,z;\widetilde A^l_h)$。与原始概率比较，因果效应定义为

$$
\Delta p^l_h(x,z,y)=p_\theta(y\mid x,z;\widetilde A^l_h)-p_\theta(y\mid x,z).\tag{6}
$$

将数据集分为原模型预测正确和错误的 $\mathcal D_{\mathrm{correct}}$、$\mathcal D_{\mathrm{incorrect}}$，分别计算平均概率变化

$$
c_\kappa(h,l)=\frac{1}{|\mathcal D_\kappa|}\sum_{(x,z,y)\in\mathcal D_\kappa}\Delta p^l_h(x,z,y),\quad\kappa\in\{\mathrm{correct},\mathrm{incorrect}\}.\tag{7}
$$

据此排序，形成两类头集合。

- **正向头 $\mathcal H_{\mathrm{pos}}$**取 $-c_{\mathrm{correct}}(h,l)$ 最大的前 $K$ 个。遮蔽后，正确样本的正确答案概率下降最多，表明其对正确预测有正向因果作用。
- **负向头 $\mathcal H_{\mathrm{neg}}$**取 $c_{\mathrm{incorrect}}(h,l)$ 最大的前 $K$ 个。遮蔽后，错误样本的正确答案概率上升最多，表明这些头对预测起负向作用。

针对属性、物体、关系各类语义任务，使用相应三元组分别识别，从而发现服务通用理解或特定类别的头。

**注意力头重缩放。** 为验证两组头的因果作用，推理时调节其影响。先定义反映概率变化绝对量的重要性

$$
c(h,l)=
\begin{cases}
|c_{\mathrm{correct}}(h,l)|,&(h,l)\in\mathcal H_{\mathrm{pos}},\\
|c_{\mathrm{incorrect}}(h,l)|,&(h,l)\in\mathcal H_{\mathrm{neg}}.
\end{cases}\tag{8}
$$

再归一化为缩放因子

$$
\lambda(h,l)=\frac{c(h,l)-c_{\min}}{c_{\max}-c_{\min}}\in[0,1].\tag{9}
$$

$c_{\min}$、$c_{\max}$ 分别在正向组和负向组内计算。推理中按权重调节

$$
\widetilde A^l_h=
\begin{cases}
(1+\lambda(h,l))A^l_h,&(h,l)\in\mathcal H_{\mathrm{pos}},\\
(1-\lambda(h,l))A^l_h,&(h,l)\in\mathcal H_{\mathrm{neg}}.
\end{cases}\tag{10}
$$

该策略增强正向头信号、抑制负向头信号，仅在前向传播时修改选定头的输出，模型结构和参数保持原状。

## 5 利用 V-SEAM 解释视觉语言模型

### 5.1 实验设置

我们考察两类模型，具体使用 LLaVA 1.5 7B（Li et al., 2024）和 InstructBLIP 7B（Dai et al., 2023）。数据使用 GQA（Hudson and Manning, 2019），其约 2200 万个实例，具有视觉与文本中的高质量物体、属性、关系标注，适合生成语义编辑图像。

根据研究范围，采用两项子采样原则。首先选择高频语义类别确保代表性，其次优先选择针对特定语义属性、答案明确的二元判别问题，减少数据噪声。平衡二元问题类型，确保均等代表。遵循既有做法（Palit et al., 2023，Golovanevsky et al., 2025），共抽取 12,647 个问题，统计见表 1，示例见附录 B。

**表 1 各概念层级的 VQA 数据规模，同一概念内按规模排序。**

| 语义层级 | VQA 类别 | 样本数 |
|---|---|---|
| 属性 | 材质 | 1,300 |
| 属性 | 颜色 | 1,500 |
| 物体 | 动物 | 1,070 |
| 物体 | 交通工具 | 1,740 |
| 物体 | 室内物品 | 2,092 |
| 关系 | 空间 | 1,950 |
| 关系 | 动作 | 2,995 |

### 5.2 主要发现

**多模态信息传递。** 图 3 展示两种模型都在早层 0–5 层理解视觉，中层 6–15 层进行图文对齐，后层大于 15 层转向问题文本以作答。文本处理需要更多层，说明文本侧使用更多参数记忆。物体信息的跨模态对齐早于属性与关系，类似人类视觉语言理解的进程（Ullman, 1987）。

![图 3 图像与问题词元的修补结果](assets/VSEAM-fig03.png)

**图 3 LLaVA 与 InstructBLIP 中图像词元和问题词元修补的 logit 变化。**

**自注意力与 MLP 的贡献。** 鉴于文本侧需要更多参数记忆，我们进一步分析 VLM 语言模型中的自注意力与 MLP。在不同层用干净问题词元嵌入替换受扰值，测量正确答案 logit 差（Meng et al., 2022）。图 4 显示，两模块都在中间第 6–12 层存储“物体”等目标方面相关词元的显著信息。

![图 4 MLP 与自注意力的逐层因果影响](assets/VSEAM-fig04.png)

**图 4 MLP（蓝）与自注意力（绿）对跨模态语义理解的逐层因果影响。** 列为 Transformer 层，行为固定提示中的问题词元，如“物体是红色的吗？”，颜色深浅为修补后正确答案的 logit 增益。

在最后词元处，最影响作答的自注意力层出现更早，例如 LLaVA 第 16 层、InstructBLIP 第 13 层，MLP 则分别在第 18、23 层。注意力主要在中间深度检索语义，深层 MLP 更影响最终决策。

使用 LogitLens（Neo et al., 2024）将逐层激活投影至词表，图 5 显示关键注意力层主要提取 `hold`、`woman`、`ski` 等答案线索，MLP 直接关注 `yes`、`no`。更多示例见附录 D。

![图 5 动作问答中的 Logit Lens](assets/VSEAM-fig05.png)

**图 5 LLaVA 动作 VQA 中 MLP（蓝）和自注意力（绿）的 Logit Lens 案例。** 各行为前五个预测词元，各列对应 Transformer 层，颜色深浅表示 logit 值。

**关键注意力头。** 表 2 给出最显著的跨语义头。正向头对正确预测具有更强因果效应，负向头比随机头引入更多误导信号。两组总体互斥，表明作用不同。负向头跨语义泛化，正向头通常在同层级共享、跨层级不同。附录 F 的空间对齐验证表明，正向头持续关注目标区域，负向头常关注干扰背景。

**表 2 两种模型跨语义类型共享的正向与负向头。编号采用层.头格式，例如 L26.H11。**

| 语义类型 | LLaVA 正向头 | LLaVA 负向头 | InstructBLIP 正向头 | InstructBLIP 负向头 |
|---|---|---|---|---|
| 属性 | L16.H1, L15.H7, L22.H11 | L31.H11, L30.H11, L29.H11 | L11.H27, L6.H30 | L0.H30, L10.H23, L6.H7, L6.H10, L7.H2, L30.H30 |
| 物体 | L0.H11, L26.H11, L11.H7 | L31.H11, L30.H11, L29.H11, L12.H2, L7.H2, L9.H27 | L11.H28, L7.H20, L23.H30, L12.H31 | L0.H30, L6.H10, L10.H9, L30.H30, L12.H25 |
| 关系 | L13.H8, L14.H15 | L31.H11, L30.H11, L29.H11, L0.H11 | L23.H16, L15.H22, L31.H19, L11.H5, L9.H28, L7.H19, L8.H28 | L12.H25, L10.H12, L6.H27, L30.H30 |

## 6 注意力头调节评估

**有效性。** 在抽样 GQA 数据上，按语义层级与功能正负系统评估关键头编辑。表 3 显示两种 VLM 编辑后平均提高约 5%，关系问题提升最大，物体问题最小。对各语义类别的 1000 个自助抽样组作配对 $t$ 检验，附录 H 显示重缩放在三个语义层面均显著优于四个基线，$p<0.001$。

**表 3 不同注意力头编辑策略下 VQA 准确率（%）。** 原始条件使用所有头，移除负向头和移除正向头分别去除前 10 个对应头，随机移除随机去除 10 个，重缩放调节头嵌入。

| 模型 | 编辑设置 | 属性 | 物体 | 关系 | 平均 |
|---|---|---|---|---|---|
| LLaVA | 原始 | 77.49 | 93.37 | 83.29 | 84.72 |
| LLaVA | 移除负向头 | 81.75 | 94.86 | 88.93 | 88.51 |
| LLaVA | 移除正向头 | 75.01 | 92.27 | 79.75 | 82.34 |
| LLaVA | 随机移除 | 77.46 | 93.24 | 83.00 | 84.57 |
| LLaVA | 重缩放 | 82.79 | 94.99 | 90.66 | 89.48 |
| InstructBLIP | 原始 | 79.46 | 93.12 | 88.55 | 87.04 |
| InstructBLIP | 移除负向头 | 82.78 | 95.24 | 94.52 | 90.85 |
| InstructBLIP | 移除正向头 | 77.85 | 91.08 | 77.84 | 82.26 |
| InstructBLIP | 随机移除 | 79.34 | 93.04 | 89.50 | 87.29 |
| InstructBLIP | 重缩放 | 84.06 | 95.67 | 95.21 | 91.65 |

**数据依赖性。** 随机使用 10%–50% 数据识别和编辑头，再测试全集。每种设置重复 10 次取平均。表 4 显示，仅用 10% 数据也能取得相近性能，说明可用于低资源环境。

**表 4 LLaVA 使用不同比例样本进行头嵌入重缩放的准确率（%），按任务类别汇总。原表以粗体标出各类别最佳值。**

| 类别 | 初始 | 10% | 20% | 30% | 50% | 全部数据 |
|---|---|---|---|---|---|---|
| 属性 | 77.49 | 79.48 | 81.17 | 82.20 | 82.50 | **82.79** |
| 物体 | 90.18 | 92.42 | 93.20 | 93.24 | **93.26** | 93.25 |
| 关系 | 83.03 | 87.01 | 90.96 | 91.02 | 91.07 | **91.11** |
| 平均 | 83.57 | 86.30 | 88.44 | 88.82 | 88.94 | **89.05** |

> 译注：表 3 与表 4 的部分同名类别汇总值不同，例如 LLaVA 原始物体准确率分别为 93.37 和 90.18。各表数值按原文保留。

**泛化性。** 进一步在两个分布外基准评估。POPE（Li et al., 2023b）同样为二元问答，含 Popular、Adversarial、Random 三个子集。表 5 比较原始 LLaVA、用 20% POPE 域内数据编辑的版本，以及用 GQA 编辑的版本。两种编辑均提升，GQA 版本甚至略高于域内数据版本。

**表 5 LLaVA 在 POPE 上的不同重缩放策略。** 初始为原模型，Edit_POPE 使用 POPE 识别的头及分数，Edit_GQA 使用 GQA 识别的头及分数。

| 子集 | 指标 | 初始 | Edit_POPE | Edit_GQA |
|---|---|---|---|---|
| Popular | 准确率 | 0.858 | 0.862 | 0.866 |
| Popular | 召回率 | 0.767 | 0.785 | 0.798 |
| Popular | F1 | 0.844 | 0.857 | 0.859 |
| Adversarial | 准确率 | 0.835 | 0.838 | 0.843 |
| Adversarial | 召回率 | 0.767 | 0.792 | 0.795 |
| Adversarial | F1 | 0.823 | 0.832 | 0.831 |
| Random | 准确率 | 0.867 | 0.880 | 0.885 |
| Random | 召回率 | 0.767 | 0.795 | 0.800 |
| Random | F1 | 0.856 | 0.875 | 0.878 |

另一个基准 COCOQA（Lu et al., 2016）包含颜色、物体、位置的开放问答。直接比较 GQA 编辑模型与原始模型，附录 H 显示两种 VLM 仍得到提升。

## 7 结论

我们提出基于语义的因果可解释性框架 V-SEAM，以视觉语义编辑实现概念级视觉干预，通过注意力调节识别和编辑关键头。对 LLaVA 与 InstructBLIP 的系统分析显示，注意力主要提取颜色等指导预测的显著语义线索，MLP 对最终输出至关重要。我们识别了发挥正、负向作用的关键头。注意力编辑提升了性能，在低资源情境和分布外任务上也保持有效。

## 局限性

1. 为确保因果干预可控，主要使用“物体是红色的吗？”等格式统一的二元判别问题。它便于精确修补与归因，但限制了对计数、排序、多跳推理等复杂任务的解释，留待未来研究。
2. 激活修补需要多次前向传播，干预大量层、词元或头时计算成本显著，限制扩展到更大模型和数据。未来可研究更高效的修补以缩短推理时间。
3. 虽然发现跨任务可复用的正、负向头，内部机制仍不充分清楚。例如各头以合作还是竞争方式交互，跨模态头是否存在领域专属动态路由，仍需系统分析。

## 参考文献

Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, and Iain Barr. 2022. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. 2018. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674– 3683.

Yuxuan Bai, Hao Cheng, Yuwei Gu, and 1 others. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2310.07904.

Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. 2024. Understanding information storage and transfer in multi-modal large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Gabriela Ben Melech Stan, Estelle Aflalo, Raanan Yehezkel Rohekar, Anahita Bhiwandiwalla, Shao-Yen Tseng, Matthew Lyle Olson, Yaniv Gurwicz, Chenfei Wu, Nan Duan, and Vasudev Lal. 2024. Lvlm-interpret: An interpretability tool for large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8182–8187.

Blair Bilodeau, Natasha Jaques, Pang Wei Koh, and Been Kim. 2024. Impossibility theorems for feature attribution. Proceedings of the National Academy of Sciences, 121(2):e2304406120.

Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, and Manling Li. 2025. Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas. arXiv preprint arXiv:2503.01773.

Weijie Chen, Yizhe Zhang, Qian Wu, and 1 others. 2024. Internvl: Scaling up vision-language pretraining with multimodal reinforcement learning. arXiv preprint arXiv:2402.00028.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. 2023. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. Hoi. 2023. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS) 36.

Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. 2023. Interpreting clip’s image representation via text-based decomposition. arXiv preprint arXiv:2310.05916.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. arXiv preprint arXiv:2304.14767.

Michal Golovanevsky, William Rudman, Vedant Palit, Carsten Eickhoff, and Ritambhara Singh. 2025. What do vlms notice? a mechanistic interpretability pipeline for gaussian-noise-free text-image corruption and evaluation. In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11462–11482, Albuquerque, New Mexico. Association for Computational Linguistics.

Lisa Anne Hendricks and Aida Nematzadeh. 2021. Probing image-language transformers for verb understanding. arXiv preprint arXiv:2106.09141.

Kaichen Huang, Jiahao Huo, Yibo Yan, Kun Wang, Yutao Yue, and Xuming Hu. 2024. Miner: Mining the underlying pattern of modality-specific neurons in multimodal large language models. arXiv preprint arXiv:2410.04819.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Jiahao Huo, Yibo Yan, Boren Hu, Yutao Yue, and Xuming Hu. 2024. Mmneuron: Discovering neuron-level domain-specific interpretation in multimodal large language model. arXiv preprint arXiv:2406.11193.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, and 1 others. 2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 conference on empirical methods in natural language processing, pages 388–395.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Junnan Li, Dongxu Li, Yixuan Xie, Yixuan Guo, Xiyang Dai, Jianfeng Gao, Jianwei Sang, and Lijuan Wang. 2023a. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International Conference on Machine Learning (ICML).

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Zihao Lin, Samyadeep Basu, Mohammad Beigi, Varun Manjunatha, Ryan A Rossi, Zichao Wang, Yufan Zhou, Sriram Balasubramanian, Arman Zarei, Keivan Rezaei, and 1 others. 2025. A survey on mechanistic interpretability for multi-modal foundation models. arXiv preprint arXiv:2502.17516.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Y Liu, Y Zhang, and S Yeung-Levy. 2025. Mechanistic interpretability meets vision language models: Insights and limitations. In The Fourth Blogpost Track at ICLR 2025.

Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. 2016. Hierarchical question-image co-attention for visual question answering. Advances in neural information processing systems, 29.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372.

Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. 2024. Towards interpreting visual information processing in vision-language models. arXiv preprint arXiv:2410.07149.

Vedant Palit, Rohan Pandey, Aryaman Arora, and Paul Pu Liang. 2023. Towards vision-language mechanistic interpretability: A causal tracing tool for blip. In Proceedings of the ICCV Workshop on Computational Linguistics for Vision and Language (CLVL).

Ekta Sood, Fabian Kögel, Philipp Müller, Dominike Thomas, Mihai Bâce, and Andreas Bulling. 2023. Multimodal integration of human-like attention in visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2648–2658.

Student. 1908. The probable error of a mean. Biometrika, pages 1–25.

Shimon Ullman. 1987. Visual routines. In Readings in computer vision, pages 298–328. Elsevier.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388– 12401.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, and 1 others. 2024. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499.

Xiaoying Xing, Chia-Wen Kuo, Li Fuxin, Yulei Niu, Fan Chen, Ming Li, Ying Wu, Longyin Wen, and Sijie Zhu. 2025. Where do large vision-language models look at when answering questions? arXiv preprint arXiv:2503.13891.

Fred Zhang and Neel Nanda. 2023. Towards best practices of activation patching in language models: Metrics and methods. arXiv preprint arXiv:2309.16042.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5579–5588.

Xiaofeng Zhang, Chen Shen, Xiaosong Yuan, Shaotian Yan, Liang Xie, Wenxiao Wang, Chaochen Gu, Hao Tang, and Jieping Ye. 2024. From redundancy to relevance: Enhancing explainability in multimodal large language models. arXiv e-prints, pages arXiv– 2406.

Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, and Mengnan Du. 2024. Explainability for large language models: A survey. ACM Transactions on Intelligent Systems and Technology, 15(2):1–38.

Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. 2024. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pages 260–278. Springer.

Deyao Zhu, Xiangxin Zhou, Xiang Wang, Xiubo Geng, Fan Liu, and Jiashen Zhu. 2024. Minigpt4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. 2024. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision, pages 195–211. Springer.

## 附录 A 提示词模板

### A.1 语义扰动提示词

我们设计提示词，精确指出问题含义中应改变的部分。隔离一个属性、物体或关系语义单元，为 PowerPaint 的局部编辑提供有视觉依据、逻辑一致的基础（Zhuang et al., 2024），也使视觉干预与有意义的语言变换一致，支持可控解释。

用 GPT-4o 自动生成语义有效的反事实问题，并以少样本策略约束。人工构造覆盖三类语义的 10 个参考例，演示如何最小修改语义单元，同时保持语法与上下文合理。提示包含任务指令、同类上下文示例和待改写新问题。表 6 列出全部人工参考例。答案列对应数据集原始二元标签，完整回答来自 GQA 的自然语言解释，二者也用作反事实生成的上下文。

**表 6 语义扰动的代表性示例。** 各行给出原问题、答案、完整解释及改变一个语义单元的反事实问题。

| 语义类型 | 原始问题 | 答案 | 完整回答 | 反事实问题 |
|---|---|---|---|---|
| 属性 | 衬衫是蓝色的吗？ | 否 | 衬衫是黑色的。 | 衬衫是**黑色**的吗？ |
| 属性 | 椅子是木制的吗？ | 是 | 椅子是木制的。 | 椅子是**金属制**的吗？ |
| 属性 | 汽车是黑色的吗？ | 否 | 汽车是白色的。 | 汽车是**白色**的吗？ |
| 物体 | 图中有狗吗？ | 否 | 图中有猫。 | 图中有**猫**吗？ |
| 物体 | 图中有椅子吗？ | 是 | 图中有椅子。 | 图中有**桌子**吗？ |
| 物体 | 图中有公共汽车吗？ | 否 | 图中有自行车。 | 图中有**自行车**吗？ |
| 关系 | 这个人在骑马吗？ | 是 | 这个人在骑马。 | 这个人在**喂马**吗？ |
| 关系 | 这个人拿着网球拍吗？ | 是 | 这个人拿着网球拍。 | 这个人在**扔网球拍**吗？ |
| 关系 | 猫在沙发左侧吗？ | 是 | 猫位于沙发左侧。 | 猫在沙发**右侧**吗？ |
| 关系 | 球在桌子下面吗？ | 否 | 球在桌子上面。 | 球在桌子**上面**吗？ |

表 7 给出各类型提示，明确区分原问题、完整回答和目标问题，末行要求只生成改写问题。全部结果经人工核验，确保语义合理、语法正确、改动最小。不到 1% 因“明亮”作为颜色等抽象或含混、无法可靠对应视觉内容的表达被过滤。

**表 7 语义扰动的少样本提示策略。** 原文的两列表格按策略分段排版，各段保留完整提示和示例。

#### 属性扰动少样本提示

> 给定一个询问物体属性的二元问题、标准答案与完整解释。通过改变颜色或材质等属性改写问题，保持原结构和上下文。
>
> 示例 1 的原问题为“衬衫是蓝色的吗？”，答案为“否”，完整回答为“衬衫是黑色的”，反事实问题为“衬衫是黑色的吗？”
>
> 示例 2 的原问题为“椅子是木制的吗？”，答案为“是”，完整回答为“椅子是木制的”，反事实问题为“椅子是金属制的吗？”
>
> 示例 3 的原问题为“汽车是黑色的吗？”，答案为“否”，完整回答为“汽车是白色的”，反事实问题为“汽车是白色的吗？”
>
> 现在完成下列内容，仅输出改写后的反事实问题，无需解释。原问题为“沙发是红色的吗？”，答案为“是”，完整回答为“沙发是红色的”。请生成反事实问题。

#### 物体扰动少样本提示

> 给定一个询问图中是否存在物体的二元问题、答案与完整解释。将所问物体替换为另一个语义合理的物体。
>
> 示例 1 的原问题为“图中有狗吗？”，答案为“否”，完整回答为“图中有猫”，反事实问题为“图中有猫吗？”
>
> 示例 2 的原问题为“图中有椅子吗？”，答案为“是”，完整回答为“图中有椅子”，反事实问题为“图中有桌子吗？”
>
> 示例 3 的原问题为“图中有公共汽车吗？”，答案为“否”，完整回答为“图中有自行车”，反事实问题为“图中有自行车吗？”
>
> 现在完成下列内容，仅输出改写后的反事实问题，无需解释。原问题为“图中有灯吗？”，答案为“否”，完整回答为“图中有架子”。请生成反事实问题。

#### 关系扰动少样本提示

> 给定询问物体间空间或动作关系的二元问题、标准答案和完整解释。改变关系，例如左变右、下变上、骑变喂。
>
> 示例 1 的原问题为“这个人在骑马吗？”，答案为“是”，完整回答为“这个人在骑马”，反事实问题为“这个人在喂马吗？”
>
> 示例 2 的原问题为“这个人拿着网球拍吗？”，答案为“是”，完整回答为“这个人拿着网球拍”，反事实问题为“这个人在扔网球拍吗？”
>
> 示例 3 的原问题为“猫在沙发左侧吗？”，答案为“是”，完整回答为“猫位于沙发左侧”，反事实问题为“猫在沙发右侧吗？”
>
> 示例 4 的原问题为“球在桌子下面吗？”，答案为“否”，完整回答为“球在桌子上面”，反事实问题为“球在桌子上方吗？”
>
> 现在完成下列内容，仅输出改写后的反事实问题，无需解释。原问题为“杯子在书旁边吗？”，答案为“否”，完整回答为“杯子在书上面”。请生成反事实问题。

### A.2 评估提示词模板

为保证输入一致、可解释，我们针对任务格式和模型指令风格，为两种模型设计模板。激活修补和分类等二元任务使用统一 yes/no 格式，明确要求简短回答。COCOQA 等开放任务使用简洁角色描述，要求单词或短语回答。LLaVA 采用 USER 与 ASSISTANT 聊天格式，InstructBLIP 使用独立陈述式提示。表 8 按步骤与模型汇总。

**表 8 两种模型在不同实验步骤中的提示模板。** 为保留模板结构，原表各单元格在下方以代码块排版。占位符和 USER、ASSISTANT 标签沿用原文。

#### 激活修补的二元判断

LLaVA 模板如下。

```text
USER:
<Image>
{Role Description}*
Question: {Question}
请用 Yes 或 No 回答问题。
ASSISTANT:
```

InstructBLIP 模板如下。

```text
<Image>
{Role Description}*
Question: {Question}
请用 yes 或 no 回答问题。
```

#### 封闭式二元判断任务

颜色、材质、室内物品和 POPE 等任务的 LLaVA 模板如下。

```text
USER:
<Image>
{Role Description}*
Question: {Question}
请用 Yes 或 No 回答问题。
ASSISTANT:
```

InstructBLIP 模板如下。

```text
<Image>
{Role Description}*
Question: {Question}
请用 yes 或 no 回答问题。
```

#### 开放式任务

COCOQA 等任务的 LLaVA 模板如下。

```text
一位好奇用户与人工智能助手之间的对话。
USER:
<Image>
{Role Description}*
Question: {Question}
Context: N/A
请用一个词或短语回答问题。
ASSISTANT:
```

InstructBLIP 模板如下。

```text
<Image>
{Role Description}*
Question: {Question}
Short Answer:
```

二元判断使用 yes/no，开放任务要求简短答案。原表在 `{Role Description}` 后带有星号，此处照录。

## 附录 B 语义基准细节

为保证可控因果干预可行、减少语义歧义，按以下原则从 GQA 构建细粒度子集。

- **问题类型筛选。** 只保留基于事实、格式统一、Yes/No 明确的二元问题，通常涉及单实体、单谓词，例如“物体是红色的吗？”或“图中有狗吗？”
- **高频语义。** 在属性、物体、关系中选择常见子类型，保证覆盖、减少稀疏性，支持稳健统计评估。
- **结构一致性。** 只保留语法简单、语义清晰的问题，例如单主谓宾结构，排除三个及以上实体或复合结构。
- **答案平衡。** 各子集的 Yes 和 No 约各占 50%，减少标签偏差，使扰动下的模型行为评估更公平。
- **基准代表性。** 参考既有可解释性研究（Palit et al., 2023，Golovanevsky et al., 2025）选取子集，规模与相关研究相当或更大。

据此组织为属性、物体、关系三层，覆盖现实图像常见概念。七个高频子类型分别为颜色、材质，室内物品、动物、交通工具，以及空间关系、动作关系，提供多样且可解释的评估。表 9 给出统计与示例。

**表 9 语义基准中各 VQA 任务的统计和示例图像。**

| 语义层级 | 子类型 | 问题示例 | 样本数 | 图像示例 |
|---|---|---|---|---|
| 属性 | 材质 | 椅子是木制的吗？ | 1,300 | ![图像示例](assets/VSEAM-table09-img01.png) |
| 属性 | 颜色 | 衬衫是蓝色的吗？ | 1,500 | ![图像示例](assets/VSEAM-table09-img02.png) |
| 物体 | 动物 | 这张图中有狗吗？ | 1,070 | ![图像示例](assets/VSEAM-table09-img03.png) |
| 物体 | 交通工具 | 这张图中有自行车吗？ | 1,740 | ![图像示例](assets/VSEAM-table09-img04.png) |
| 物体 | 室内物品 | 这张图中有镜子吗？ | 2,092 | ![图像示例](assets/VSEAM-table09-img05.png) |
| 关系 | 空间 | 男人在马的右侧吗？ | 1,950 | ![图像示例](assets/VSEAM-table09-img06.png) |
| 关系 | 动作 | 女人拿着滑雪板吗？ | 2,995 | ![图像示例](assets/VSEAM-table09-img07.png) |

## 附录 C 扰动方法比较

为验证语义扰动对全局语义一致性的保持，与高斯噪声、椒盐噪声、遮挡和扩散编辑比较。传统随机噪声或局部遮挡常破坏低层特征、结构连贯性，使编码器提取的特征分布偏移。本文方法利用场景图、SAM 和先进图像编辑，精确修改目标语义区域。图 6、7、8 分别展示属性、物体、关系任务上的细粒度编辑及全局连贯性。

![图 6 天空颜色的扰动比较](assets/VSEAM-fig06.png)

**图 6 “天空是蓝色的吗？”任务的扰动策略视觉比较。** 本文方法位于右侧，在保持整体连贯的同时精确干预天空。高斯噪声、区域遮挡、Stable Diffusion 整图重绘引入较大失真和意外语义变化。

> 译注：图 6 原图注称其为物体级任务，正文按颜色属性任务介绍，此处保留问题并说明该差异。

![图 7 三明治存在性扰动比较](assets/VSEAM-fig07.png)

**图 7 物体级任务“图中有三明治吗？”的扰动策略视觉比较。** 本文方法在保持全局连贯的同时精确修改目标存在性，传统方法有明显失真且缺乏细粒度控制。

![图 8 空间关系扰动比较](assets/VSEAM-fig08.png)

**图 8 关系级任务“男人站在路牌左侧吗？”的扰动策略视觉比较。** 本文方法精确改变男人与路牌的空间关系并保持全局连贯。高斯噪声、区域遮挡及扩散整图重绘会产生显著伪影，难以实现可控语义编辑。

为量化分布偏移，用 LLaVA 视觉编码器 CLIP-ViT-L 提取原图与扰动图的全局特征，计算余弦相似度。若只修改局部语义，整体表征应与原图高度相似。表 10–12 给出三类任务的平均相似度。本文方法持续最高，表明全局表征受到的扰动最小。噪声、遮挡和整图重绘则造成明显低层失真和表征偏移。

**表 10 Color VQA 上原图与扰动图的特征余弦相似度。** 使用 LLaVA 的 CLIP-ViT-L，数值越高表示全局视觉语义保留越好，有利于现实环境中的稳定推理。

| 扰动方法 | 余弦相似度 |
|---|---|
| 高斯噪声，强度 10 | 0.8766 |
| 高斯噪声，强度 30 | 0.6333 |
| 高斯噪声，强度 50 | 0.5025 |
| 高斯噪声，强度 80 | 0.3446 |
| 椒盐噪声，强度 10 | 0.7088 |
| 椒盐噪声，强度 30 | 0.5554 |
| 椒盐噪声，强度 50 | 0.4110 |
| 椒盐噪声，强度 80 | 0.3142 |
| 遮挡 | 0.6005 |
| 扩散模型重绘 | 0.6891 |
| 本文语义扰动 | 0.9168 |

**表 11 Indoor VQA 上原图与扰动图的特征余弦相似度。** 特征编码器、解释与表 10 相同。

| 扰动方法 | 余弦相似度 |
|---|---|
| 高斯噪声，强度 10 | 0.8628 |
| 高斯噪声，强度 30 | 0.8148 |
| 高斯噪声，强度 50 | 0.7206 |
| 高斯噪声，强度 80 | 0.5988 |
| 椒盐噪声，强度 10 | 0.7427 |
| 椒盐噪声，强度 30 | 0.6729 |
| 椒盐噪声，强度 50 | 0.6159 |
| 椒盐噪声，强度 80 | 0.5840 |
| 遮挡 | 0.6821 |
| 扩散模型重绘 | 0.6205 |
| 本文语义扰动 | 0.8970 |

**表 12 Action VQA 上原图与扰动图的特征余弦相似度。** 相较物体级任务，结果更低，反映动作语义对视觉改变更敏感。

| 扰动方法 | 余弦相似度 |
|---|---|
| 高斯噪声，强度 10 | 0.7921 |
| 高斯噪声，强度 30 | 0.7430 |
| 高斯噪声，强度 50 | 0.6204 |
| 高斯噪声，强度 80 | 0.4877 |
| 椒盐噪声，强度 10 | 0.7033 |
| 椒盐噪声，强度 30 | 0.6182 |
| 椒盐噪声，强度 50 | 0.5820 |
| 椒盐噪声，强度 80 | 0.5194 |
| 遮挡 | 0.4213 |
| 扩散模型重绘 | 0.5586 |
| 本文语义扰动 | 0.8542 |

## 附录 D MLP 与注意力投影的 Logit Lens 分析

为理解推理过程，按 Neo 等（2024）的标准流程，用 Logit Lens 分析两种模型语言模块逐层词元分布，比较每层 MLP 路径与注意力输出路径投影的 logits。

在“椅子是木制的吗？”等 VQA 中，早期第 2–7 层两路径主要输出标点、乱码、子词片段等低层或含混词元，尚未形成有意义语义。约第 9 层，注意力开始产生 `yes` 等答案相关词元，标志图文对齐开始。MLP 到第 15 层之后才稳定生成 `yes`、`true`、`number` 等高置信词元。

第 16–25 层，注意力越来越关注 `wood`、`material`、`color` 等有图像依据的概念，MLP 则聚焦决策词元，输出置信度不断上升。最后一层，MLP 高确定性地产生最终答案，注意力倾向 `<0x0A>`、句点等结构词元，体现输出组织作用。

这些发现支持分阶段层次解释。早层处理视觉，中层第 8–16 层对齐模态并建立语义依据，深层第 16–32 层整合语言推理与答案生成。图 9、10 展示预测从噪声或含混输出，发展为对齐概念，最终收敛至任务专属高置信答案。注意力主要承担多模态对齐和语义构建，MLP 侧重决策。

![图 9 材质问答的 Logit Lens](assets/VSEAM-fig09.png)

**图 9 LLaVA 在 Material VQA 中的 MLP（蓝）与自注意力（绿）Logit Lens 案例。**

![图 10 室内物品问答的 Logit Lens](assets/VSEAM-fig10.png)

**图 10 InstructBLIP 在 Indoor VQA 中的 MLP（蓝）与自注意力（绿）Logit Lens 案例。**

## 附录 E 激活修补补充结果

补充材质、交通工具、动物、空间四个子任务的激活修补，系统考察 MLP 和注意力层在信息流中的因果作用，见图 11–14。LLaVA 与 InstructBLIP 在不同语义任务上具有高度一致的因果响应和功能分工，进一步支持主文关于推理阶段与模块分工的分析。

![图 11 材质任务的逐层因果影响](assets/VSEAM-fig11.png)

**图 11 MLP 和自注意力在 Material VQA 中的逐层因果影响。**

![图 12 交通工具任务的逐层因果影响](assets/VSEAM-fig12.png)

**图 12 注意力和 MLP 在 Vehicle VQA 中的逐层因果影响。**

![图 13 动物任务的逐层因果影响](assets/VSEAM-fig13.png)

**图 13 MLP 和自注意力在 Animal VQA 中的逐层因果影响。**

![图 14 空间任务的逐层因果影响](assets/VSEAM-fig14.png)

**图 14 MLP 和自注意力在 Spatial VQA 中的逐层因果影响。**

## 附录 F 语义注意力头补充结果

**按语义类别的完整归因。** 为补充主文共享头分析，表 13 列出每项任务的完整结果。对每个子类型，通过第 4 节的消融和激活修补，识别因果贡献最大的前 10 个推理头和干扰头。编号 `L# H#` 中，L 表示层索引，H 表示层内头索引。

**表 13 LLaVA 与 InstructBLIP 在不同语义 VQA 中识别出的前 10 个正向及负向头。**

| VQA 任务与方向 | LLaVA 注意力头 | InstructBLIP 注意力头 |
|---|---|---|
| 颜色属性，正向 | L6 H19, L26 H15, L16 H4, L11 H29, L15 H7, L16 H1, L0 H11, L22 H11, L7 H23, L24 H11 | L11 H26, L10 H7, L6 H30, L9 H17, L10 H9, L6 H10, L10 H23, L6 H27, L11 H27, L11 H11 |
| 颜色属性，负向 | L31 H11, L30 H11, L29 H11, L9 H12, L26 H11, L7 H27, L15 H26, L10 H18, L10 H1, L15 H3 | L0 H30, L10 H23, L6 H7, L29 H30, L11 H26, L6 H10, L10 H7, L7 H2, L30 H30, L9 H18 |
| 材质属性，正向 | L12 H23, L16 H1, L3 H14, L13 H8, L6 H17, L9 H22, L14 H18, L22 H11, L8 H15, L15 H7 | L23 H30, L7 H5, L6 H30, L11 H5, L8 H21, L7 H19, L11 H27, L5 H28, L11 H20, L12 H6 |
| 材质属性，负向 | L30 H11, L31 H11, L29 H11, L11 H29, L3 H11, L2 H11, L4 H11, L9 H24, L10 H16, L7 H12 | L0 H30, L10 H23, L6 H10, L7 H2, L30 H30, L10 H12, L13 H19, L11 H27, L6 H7, L15 H5 |
| 室内物体，正向 | L0 H11, L0 H23, L26 H11, L7 H20, L2 H16, L1 H17, L14 H18, L6 H26, L22 H11, L11 H7 | L23 H30, L11 H23, L7 H20, L11 H28, L23 H30, L9 H0, L12 H31, L6 H29, L11 H6, L7 H19 |
| 室内物体，负向 | L31 H11, L30 H11, L29 H11, L12 H2, L7 H2, L2 H11, L10 H1, L9 H27, L12 H29, L12 H13 | L0 H30, L8 H27, L6 H10, L30 H30, L12 H25, L10 H9, L13 H19, L14 H25, L30 H4, L10 H8 |
| 动物物体，正向 | L0 H11, L26 H11, L1 H2, L11 H3, L11 H7, L2 H2, L6 H8, L15 H11, L10 H26, L11 H1 | L23 H30, L5 H28, L23 H30, L9 H0, L11 H28, L15 H22, L7 H20, L7 H13, L7 H11, L12 H31 |
| 动物物体，负向 | L31 H11, L30 H11, L12 H2, L29 H11, L2 H31, L7 H2, L9 H27, L7 H9, L5 H0, L7 H1 | L0 H30, L6 H10, L10 H9, L0 H28, L12 H25, L13 H28, L10 H12, L30 H30, L6 H7, L5 H16 |
| 交通工具物体，正向 | L0 H11, L26 H11, L8 H11, L2 H2, L1 H2, L11 H7, L10 H15, L11 H17, L13 H4, L11 H1 | L23 H30, L7 H20, L11 H28, L12 H22, L11 H5, L12 H31, L10 H10, L8 H13, L9 H28, L31 H19 |
| 交通工具物体，负向 | L31 H11, L30 H11, L29 H11, L12 H2, L9 H30, L11 H30, L9 H27, L12 H29, L7 H2, L12 H13 | L0 H30, L6 H10, L10 H9, L30 H30, L12 H25, L7 H4, L6 H7, L6 H27, L8 H27, L10 H12 |
| 动作关系，正向 | L16 H1, L26 H11, L1 H16, L13 H8, L8 H13, L8 H10, L12 H16, L14 H15, L5 H16, L13 H24 | L23 H16, L3 H30, L8 H28, L15 H22, L10 H11, L11 H5, L9 H28, L7 H19, L10 H6, L31 H19 |
| 动作关系，负向 | L31 H11, L30 H11, L29 H11, L12 H14, L10 H1, L1 H4, L12 H2, L5 H19, L0 H11, L2 H11 | L6 H27, L12 H25, L10 H9, L7 H3, L6 H10, L10 H0, L10 H12, L30 H30, L30 H4, L7 H27 |
| 空间关系，正向 | L10 H17, L14 H15, L15 H18, L18 H17, L10 H14, L8 H27, L9 H28, L2 H23, L13 H8, L8 H20 | L23 H16, L15 H22, L31 H19, L11 H5, L11 H28, L10 H1, L7 H19, L9 H28, L5 H30, L8 H28 |
| 空间关系，负向 | L31 H11, L30 H11, L29 H11, L0 H11, L15 H23, L15 H10, L15 H4, L3 H11, L6 H14, L1 H16 | L10 H10, L12 H25, L6 H27, L13 H19, L0 H30, L12 H27, L5 H23, L30 H30, L10 H12 |

> 译注：原表部分清单含重复编号。InstructBLIP 的空间关系负向头一栏列出 9 个编号，尽管表题写前 10 个。清单按原文保留。

跨模型与任务存在以下共同趋势。

- LLaVA 的正向头往往聚集在第 10–23 等中后层，与模型由感知对齐转向抽象推理相符。
- LLaVA 的负向头常集中在第 29–31 等末层，可能反映后期注意力漂移或特征过拟合。
- InstructBLIP 的 L11 H28、L23 H30 等跨任务重复出现，可能发挥通用正向头作用。

这些结果支持模型形成语义分化推理路径的观点，并为重缩放等因果编辑提供具体目标。

**正、负向头的行为差异。** 为量化差异，计算注意力分布与真实物体框的平均空间对齐。注意力重叠率定义为框内权重和除以总注意力质量，对每个模型、数据集的前 10 个正向或负向头取平均。表 14 显示正向头高度对齐，例如 LLaVA-Animal 达 83.8%，负向头低至 7.1%，常关注草地、墙面等无关背景，量化支持两组因果差别。

**表 14 物体边界框内平均注意力重叠率（%）。** 越高表示与目标对齐越强，按每模型前 10 个对应头并跨样本平均。

| 数据集 | 注意力头组 | LLaVA | InstructBLIP |
|---|---|---|---|
| 动物 | 正向 | 83.8 | 79.2 |
| 动物 | 负向 | 7.1 | 8.5 |
| 动作 | 正向 | 87.0 | 85.4 |
| 动作 | 负向 | 15.6 | 13.9 |

## 附录 G 因果注意力头评估的数据集与任务

### G.1 POPE 物体幻觉评估基准

POPE（Polling-based Object Probing Evaluation，Li et al., 2023b）专门诊断大型视觉语言模型在识别任务中的物体幻觉。基于 MS-COCO 图像，通过物体级二元问题，考察分布变化下的鲁棒性与偏差。

它包含三个针对不同目标的子集。

- **Popular**包含预训练语料中常见物体，测试对常见物体先验的依赖。
- **Adversarial**加入误导干扰或共现偏差，评估困难情境下的鲁棒性。
- **Random**随机抽取物体—问题对，作为无偏评估的对照。

每幅图像配多个二元问题，典型形式为“图中有 `<object>` 吗？”，其中占位符为具体名词。实验直接将 GQA 识别的头迁移至 POPE，并与用 10%、20% POPE 训练样本重新识别的结果比较。统计与示例见表 15。

**表 15 POPE 子集概况。** 各子集包含为不同评估目标设计的物体存在性二元问题。

| 子集 | 样本数 | 问题示例 |
|---|---|---|
| Popular | 3,000 | 图中有椅子吗？ |
| Adversarial | 3,000 | 图中有男人吗？ |
| Random | 2,910 | 图中有瓶子吗？ |

### G.2 COCO-QA 开放式视觉问答数据集

COCO-QA（COCO Question Answering，Lu et al., 2016）基于 MS-COCO Captions 图像自动生成，用于开放问答中的图像理解与语言落地评估。

**数据结构。** 约 123,287 幅独立图像，117,684 个问答对，每对关联一幅 COCO 图像。问题通过图像描述的句法、语义变换自动生成，答案通常是一个词或短语。

**问题类型。** 包含物体识别，例如“桌上是什么？”，数量计数，例如“有多少人？”，颜色属性，例如“汽车是什么颜色？”，以及位置关系，例如“猫在哪里？”。表 16 显示物体类最多，其次为数量、颜色和位置。

**表 16 COCO-QA 问题类型分布。**

| 问题类型 | 问答对数量 | 比例 |
|---|---|---|
| 物体 | 69,753 | 59.3% |
| 数量 | 16,594 | 14.1% |
| 颜色 | 16,201 | 13.8% |
| 位置 | 15,136 | 12.8% |
| 合计 | 117,684 | 100% |

**子集选择。** 评估选择物体、颜色和位置三类，其语义清晰，分别对应本文物体识别、属性识别和空间关系理解。数量类语义相对含混，且与本文框架缺乏直接对应，因此排除。

## 附录 H 语义注意力编辑补充结果

补充 InstructBLIP 在 GQA 风格语义子任务及 POPE 上的重缩放结果，以说明方法也适用于指令微调多模态模型。

### 注意力头编辑的有效性

表 17 展示对关键头的系统编辑。两种模型在所有语义层面均显著提高，说明方法能增强语义理解。

**表 17 LLaVA 与 InstructBLIP 在不同编辑策略下的 VQA 准确率（%）。** 属性包含颜色、材质，物体包含动物、交通工具、室内物品，关系包含动作与空间。

| 模型 | 编辑设置 | 颜色 | 材质 | 动物 | 交通工具 | 室内物品 | 动作 | 空间 |
|---|---|---|---|---|---|---|---|---|
| LLaVA | 原始 | 76.13 | 78.84 | 96.45 | 90.29 | 83.80 | 88.21 | 77.85 |
| LLaVA | 移除负向头 | 79.73 | 83.77 | 97.01 | 92.70 | 88.58 | 92.89 | 85.33 |
| LLaVA | 移除正向头 | 74.40 | 75.62 | 95.33 | 89.20 | 79.64 | 86.44 | 73.18 |
| LLaVA | 随机移除 | 76.00 | 78.92 | 96.36 | 90.11 | 83.13 | 88.08 | 77.79 |
| LLaVA | 重缩放 | 80.27 | 85.31 | 97.10 | 92.87 | 89.77 | 93.29 | 88.92 |
| InstructBLIP | 原始 | 77.53 | 81.38 | 96.17 | 90.06 | 87.14 | 90.88 | 87.64 |
| InstructBLIP | 移除负向头 | 80.47 | 85.08 | 97.38 | 93.10 | 92.78 | 95.19 | 95.59 |
| InstructBLIP | 移除正向头 | 75.47 | 80.23 | 94.86 | 87.30 | 80.45 | 81.17 | 71.90 |
| InstructBLIP | 随机移除 | 77.53 | 81.15 | 96.07 | 90.00 | 87.14 | 91.86 | 86.51 |
| InstructBLIP | 重缩放 | 81.27 | 86.84 | 97.48 | 93.85 | 92.83 | 96.03 | 96.77 |

### 统计显著性分析

采用自助重采样（Koehn, 2004）生成 1000 个抽样组，每组从原测试集随机抽 100 例，分别强调物体、关系或属性变化。对每组计算重缩放相对四个基线的性能差值 $\Delta(\mathrm{pp})=\mathrm{rescaling}-\mathrm{baseline}$。

每种语义属性、每组方法比较获得 1000 个差值，再作配对 $t$ 检验（Student, 1908）。表 18 基于 LLaVA，总体结果合并三个语义属性的全部抽样组。各类别相对四个基线都显著提高，所有 $p<0.001$。

**表 18 注意力头重缩放与基线的性能差值。** Δ 以百分点计，星号 `***` 表示 $p<0.001$。

| 类别 | 基线 | 平均差值 Δ（百分点） | 标准差 | t | p 值 |
|---|---|---|---|---|---|
| 属性 | 原始 | 7.92 | 2.96 | 84.79 | *** |
| 属性 | 随机移除 | 7.80 | 3.52 | 70.20 | *** |
| 属性 | 移除正向头 | 10.55 | 3.95 | 84.61 | *** |
| 属性 | 移除负向头 | 2.93 | 1.87 | 49.68 | *** |
| 物体 | 原始 | 3.01 | 1.51 | 63.16 | *** |
| 物体 | 随机移除 | 3.19 | 2.06 | 49.06 | *** |
| 物体 | 移除正向头 | 4.78 | 2.27 | 66.75 | *** |
| 物体 | 移除负向头 | 1.93 | 1.18 | 51.81 | *** |
| 关系 | 原始 | 9.57 | 3.82 | 79.40 | *** |
| 关系 | 随机移除 | 10.43 | 3.93 | 84.08 | *** |
| 关系 | 移除正向头 | 13.62 | 4.39 | 98.29 | *** |
| 关系 | 移除负向头 | 3.65 | 2.14 | 54.00 | *** |
| 总体 | 原始 | 6.62 | 3.61 | 100.49 | *** |
| 总体 | 随机移除 | 6.75 | 3.73 | 99.17 | *** |
| 总体 | 移除正向头 | 9.86 | 4.46 | 121.18 | *** |
| 总体 | 移除负向头 | 2.88 | 1.89 | 83.49 | *** |

### 数据依赖性分析

随机抽取 10%–50% 实验数据识别、编辑关键头，再测试全集。表 19–21 比较不同数据比例。

InstructBLIP 仅用 10% 训练数据，在各类别已接近全量数据结果。表 20 提供七个具体任务的细节，展示数据效率和通用性。LLaVA 的表 21 同样表明，低资源情况下仍能取得相近性能。

**表 19 InstructBLIP 使用不同样本比例进行重缩放的分类别准确率（%）。**

| 类别 | 初始 | 10% | 20% | 30% | 50% | 全部数据 |
|---|---|---|---|---|---|---|
| 属性 | 79.46 | 80.48 | 82.11 | 83.05 | 83.73 | 84.06 |
| 物体 | 91.12 | 92.78 | 93.57 | 94.37 | 94.61 | 94.72 |
| 关系 | 89.26 | 91.42 | 93.56 | 95.27 | 96.10 | 96.40 |
| 平均 | 86.61 | 88.23 | 89.75 | 90.90 | 91.48 | 91.73 |

**表 20 InstructBLIP 在七个语义任务中，使用不同比例样本进行头嵌入重缩放的准确率（%）。**

| 数据集 | 初始 | 10% | 20% | 30% | 50% | 全部数据 |
|---|---|---|---|---|---|---|
| 颜色 | 77.53 | 78.80 | 79.60 | 80.33 | 81.07 | 81.27 |
| 材质 | 81.38 | 82.15 | 84.62 | 85.77 | 86.38 | 86.84 |
| 室内物品 | 87.14 | 89.48 | 90.63 | 92.02 | 92.50 | 92.83 |
| 动物 | 96.17 | 96.72 | 97.20 | 97.48 | 97.48 | 97.48 |
| 交通工具 | 90.06 | 92.13 | 92.87 | 93.62 | 93.85 | 93.85 |
| 动作 | 90.88 | 92.79 | 93.52 | 94.89 | 95.86 | 96.03 |
| 空间 | 87.64 | 90.05 | 93.59 | 95.64 | 96.34 | 96.77 |

**表 21 LLaVA 在不同 VQA 任务中，使用不同比例样本进行重缩放的准确率（%）。**

| 数据集 | 初始 | 10% | 20% | 30% | 50% | 全部数据 |
|---|---|---|---|---|---|---|
| 颜色 | 76.13 | 77.73 | 79.80 | 79.93 | 80.00 | 80.27 |
| 材质 | 78.84 | 81.23 | 82.54 | 84.46 | 85.00 | 85.31 |
| 室内物品 | 83.80 | 87.48 | 89.67 | 89.77 | 89.82 | 89.77 |
| 动物 | 96.45 | 97.01 | 97.10 | 97.10 | 97.10 | 97.10 |
| 交通工具 | 90.29 | 92.76 | 92.82 | 92.84 | 92.87 | 92.87 |
| 动作 | 88.21 | 90.58 | 93.29 | 93.22 | 93.26 | 93.29 |
| 空间 | 77.85 | 83.44 | 88.62 | 88.82 | 88.87 | 88.92 |

### 泛化评估

在 POPE 和 COCOQA 两个分布外基准测试泛化。

**POPE。** 表 22 比较 InstructBLIP 初始模型、用 POPE 数据重缩放、用 GQA 识别头与重要性分数重缩放。即使不在目标数据集重新分析，也能提高表现，展示跨任务泛化。

**表 22 InstructBLIP 在 POPE 上的不同重缩放策略。** 初始为原模型，Edit_POPE 基于 POPE，Edit_GQA 使用 GQA 识别的头和分数。

| 子集 | 指标 | 初始 | Edit_POPE | Edit_GQA |
|---|---|---|---|---|
| Popular | 准确率 | 0.842 | 0.861 | 0.857 |
| Popular | 召回率 | 0.731 | 0.782 | 0.795 |
| Popular | F1 | 0.823 | 0.843 | 0.841 |
| Adversarial | 准确率 | 0.830 | 0.846 | 0.848 |
| Adversarial | 召回率 | 0.731 | 0.787 | 0.783 |
| Adversarial | F1 | 0.811 | 0.828 | 0.829 |
| Random | 准确率 | 0.852 | 0.872 | 0.869 |
| Random | 召回率 | 0.731 | 0.786 | 0.790 |
| Random | F1 | 0.835 | 0.853 | 0.854 |

**COCOQA。** 表 23 直接比较 GQA 编辑模型与原始模型，在颜色、物体、位置开放问答中均提升。补充结果显示，注意力头重缩放的有效性和泛化性覆盖不同模型、数据量和任务，支持正文观点。

**表 23 两种模型在 COCOQA 的颜色、物体、位置子任务上，重缩放前后的准确率（%）。**

| 子任务 | 模型 | 初始 | Edit_GQA |
|---|---|---|---|
| 颜色 | InstructBLIP | 65.54 | 68.49 |
| 颜色 | LLaVA | 80.48 | 82.35 |
| 物体 | InstructBLIP | 78.40 | 80.12 |
| 物体 | LLaVA | 85.15 | 86.40 |
| 位置 | InstructBLIP | 67.39 | 69.01 |
| 位置 | LLaVA | 61.84 | 62.98 |

## 附录 I 计算资源

所有实验使用一台配有两张 NVIDIA Tesla V100 GPU（32GB 显存）的机器，CPU 为 Intel Xeon Gold 6230R，2.10GHz、8 核，内存 62GB。

激活修补、注意力头归因和视觉扰动生成使用 PyTorch 与 Hugging Face Transformers 实现。

**表 24 LLaVA 在七个 VQA 任务中的语义扰动定性示例。** 各行给出问题、原始图像及预测、扰动图像及更新后的预测。

| 问题 | 原始图像 | 原始预测 | 扰动图像 | 扰动后预测 |
|---|---|---|---|---|
| 衬衫是蓝色的吗？ | ![原图](assets/VSEAM-table24-img01.png) | 否 | ![扰动图](assets/VSEAM-table24-img02.png) | 是 |
| 椅子是木制的吗？ | ![原图](assets/VSEAM-table24-img03.png) | 是 | ![扰动图](assets/VSEAM-table24-img04.png) | 否 |
| 这张图中有镜子吗？ | ![原图](assets/VSEAM-table24-img05.png) | 是 | ![扰动图](assets/VSEAM-table24-img06.png) | 否 |
| 这张图中有狗吗？ | ![原图](assets/VSEAM-table24-img07.png) | 否 | ![扰动图](assets/VSEAM-table24-img08.png) | 是 |
| 这张图中有自行车吗？ | ![原图](assets/VSEAM-table24-img09.png) | 是 | ![扰动图](assets/VSEAM-table24-img10.png) | 否 |
| 男人在马的右侧吗？ | ![原图](assets/VSEAM-table24-img11.png) | 否 | ![扰动图](assets/VSEAM-table24-img12.png) | 是 |
| 女人拿着滑雪板吗？ | ![原图](assets/VSEAM-table24-img13.png) | 是 | ![扰动图](assets/VSEAM-table24-img14.png) | 否 |
