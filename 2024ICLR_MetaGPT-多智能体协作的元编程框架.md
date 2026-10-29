# MetaGPT：面向多智能体协作框架的元编程

**原题：** MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework  
**会议：** International Conference on Learning Representations（ICLR），2024  
**作者：** Sirui Hong¹*、Mingchen Zhuge²*、Jonathan Chen¹、Xiawu Zheng³、Yuheng Cheng⁴、Ceyao Zhang⁴、Jinlin Wang¹、Zili Wang、Steven Ka Shing Yau⁵、Zijuan Lin⁴、Liyang Zhou⁶、Chenyu Ran¹、Lingfeng Xiao¹˒⁷、Chenglin Wu¹†、Jürgen Schmidhuber²˒⁸。

**作者单位：** ¹DeepWisdom；²阿卜杜拉国王科技大学（KAUST）人工智能计划；³厦门大学；⁴香港中文大学（深圳）；⁵南京大学；⁶宾夕法尼亚大学；⁷加利福尼亚大学伯克利分校；⁸瑞士人工智能实验室 IDSIA / USI / SUPSI。

\* 两位作者对本工作贡献相同。  
† 通讯作者为 DeepWisdom 的 Chenglin Wu（alexanderwu@fuzhi.ai）。

> **译本说明（2026-10-28 核查）：** 本译文以 [ICLR 2024 正式论文集提供的免费 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf) 为底本，[正式论文页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6507b115562bb0a305f1958ccc87355a-Abstract-Conference.html)。保留原文节次、段落内容、图表编号、数学公式、引文和脚注；图中原有英文、代码与符号保留，并配中文图注及必要释义。参考文献和附录不翻译，按原版页面保留在文末。会议 PDF 自带的页眉为“Preprint”，这是下载文件的原有标记。最新作者版本与底本的比较见文末“版本核查记录”。

## 摘要

由大语言模型（Large Language Models，LLM）智能体组成的群体，在自动化问题求解方面已经取得显著进展。现有基于 LLM 的多智能体系统已经能够解决简单的对话任务。然而，对于更复杂的任务，简单串联 LLM 会引发级联幻觉，进而造成逻辑不一致，使求解更加困难。本文提出 MetaGPT：一种创新的元编程框架，将高效的人类工作流程引入基于 LLM 的多智能体协作。MetaGPT 将标准操作规程（Standardized Operating Procedures，SOP）编码为提示序列，使工作流更加顺畅，从而让具有人类式领域专长的智能体验证中间结果并减少错误。MetaGPT 采用流水线范式，为不同智能体分配多样化角色，高效地将复杂任务分解为需要多个智能体共同完成的子任务。在协作式软件工程基准上，MetaGPT 比以往基于聊天的多智能体系统生成的解决方案更加连贯。项目地址：[https://github.com/geekan/MetaGPT](https://github.com/geekan/MetaGPT)。

## 1 引言

使用大语言模型（LLM）的自主智能体，为增强和复现人类工作流带来了富有前景的机会。然而，在现实应用中，现有系统（Park et al., 2023；Zhuge et al., 2023；Cai et al., 2023；Wang et al., 2023c；Li et al., 2023；Du et al., 2023；Liang et al., 2023；Hao et al., 2023）往往过度简化其中的复杂性。它们难以实现有效、连贯且准确的问题求解过程，尤其是在需要有意义的协作交互时（Chen et al., 2024；Zhang et al., 2023；Dong et al., 2023；Zhou et al., 2023；Qian et al., 2023）。

通过广泛的协作实践，人类已经在各个领域形成了广为接受的标准操作规程（SOP）（Belbin, 2012；Manifesto, 2001；DeMarco & Lister, 2013）。这些 SOP 对任务分解和有效协调起着关键作用。此外，SOP 还规定每位团队成员的职责，并为中间产出制定标准。定义明确的 SOP 能够提高任务执行的一致性和准确性，使其符合既定角色及质量标准（Belbin, 2012；Manifesto, 2001；DeMarco & Lister, 2013；Wooldridge & Jennings, 1998）。例如，在软件公司中，产品经理分析市场竞争和用户需求，按照标准化结构编写产品需求文档（Product Requirements Document，PRD），以指导开发过程。

受到上述思想的启发，我们设计了一个富有前景、基于 GPT 的元编程框架 MetaGPT，它显著受益于 SOP。与其他工作（Li et al., 2023；Qian et al., 2023）不同，MetaGPT 要求智能体生成结构化输出，例如高质量需求文档、设计产物、流程图和接口规范。

![图1：MetaGPT 与现实人类团队的软件开发 SOP](assets/metagpt-figure1.png)

**图1：MetaGPT 与现实人类团队的软件开发 SOP。** 在软件工程中，SOP 促进不同角色之间的协作。MetaGPT 展示了将复杂任务分解为具体可执行流程，并将其分配给不同角色（如产品经理、架构师、工程师等）的能力。

**图1内文字译注：** 左侧“人类开发 SOP”依次为规划、需求分析、架构设计、系统设计、编码、测试、验收。中间为“基于开发 SOP 的 MetaGPT 智能体协作”：产品经理产出需求文档，架构师产出系统设计，项目经理产出任务，工程师产出代码，质量保证工程师产出测试；五阶段分别为定义、设计、规划与编码、测试、验收。右侧为“人机交互”：一句话需求为“编写一个经典、简单的 Flappy Bird 游戏”；老板验收并付款：“很不错！我可以直接通过界面和键盘玩 Flappy Bird。”

使用结构化中间输出能够显著提高目标代码生成的成功率，因为它有助于保持沟通的一致性，尽量减少协作中的歧义和错误。更形象地说，在 MetaGPT 模拟的公司中，所有员工都遵循严格而顺畅的工作流，全部工作交接都必须符合既定标准。这降低了 LLM 之间闲聊引发幻觉的风险，特别是在角色扮演框架中，例如：“嗨，你好，最近怎么样？”——Alice（产品经理）；“很好！你吃午饭了吗？”——Bob（架构师）。

受益于 SOP，MetaGPT 为元编程提供了一种富有前景的方法。在本文语境中，我们将元编程[^1]理解为“通过编程来编程”（programming to program），与更广泛的元学习及“学会学习”（learning to learn）领域相区别（Schmidhuber, 1987；1993a；Hochreiter et al., 2001；Schmidhuber, 2006；Finn et al., 2017）。

这种元编程概念也涵盖了 CodeBERT（Feng et al., 2020）等早期工作，以及 CodeLlama（Rozière et al., 2023）、WizardCoder（Luo et al., 2023）等近期项目。然而，MetaGPT 提供了一种独特的解决方案：通过一组组织良好、具有专长的智能体，实现高效的元编程。每个智能体都有明确的角色与专长，并遵循既定标准。这使得运行期间的需求分析、系统设计、代码生成、修改、执行与调试能够自动完成，体现了基于智能体的技术如何增强元编程。

为了验证 MetaGPT 的设计，我们使用公开的 HumanEval（Chen et al., 2021a）和 MBPP（Austin et al., 2021）进行评估。值得注意的是，在代码生成基准上，MetaGPT 的 Pass@1 分别达到 85.9% 和 87.7%，取得新的最先进水平（state of the art，SoTA）。与 AutoGPT（Torantulino et al., 2023）、LangChain（Chase, 2022）、AgentVerse（Chen et al., 2023）和 ChatDev（Qian et al., 2023）等用于创建复杂软件项目的流行框架相比，MetaGPT 在处理更高软件复杂度和提供广泛功能方面同样表现突出。在我们的实验评估中，MetaGPT 的任务完成率达到 100%，体现了这一设计的稳健性以及效率（时间与 token 成本）。

我们的贡献概括如下：

- 提出 MetaGPT：一种基于 LLM 的多智能体协作元编程框架。它方便、灵活，具有角色定义、消息共享等定义明确的功能，是开发基于 LLM 的多智能体系统的实用平台。
- 在 MetaGPT 的整体设计中创新地融入人类式 SOP，显著增强框架的稳健性，减少基于 LLM 的智能体之间无效的协作。此外，我们引入一种新的可执行反馈机制，在运行期间调试并执行代码，显著提高代码生成质量，例如在 MBPP 上取得 5.4 个百分点的绝对提升。（译注：此处原文用词为 executive feedback，按全文第 3.3 节的 executable feedback 统一译名。）
- 在 HumanEval（Chen et al., 2021a）和 MBPP（Austin et al., 2021）上达到最先进性能。大量结果有力验证了 MetaGPT，表明它是开发基于 LLM 的多智能体系统的一种富有前景的元编程框架。

[^1]: 原文脚注1：[Metaprogramming](https://en.wikipedia.org/w/index.php?title=Metaprogramming)。

## 2 相关工作

**自动编程。** 自动编程的起源可以追溯到上个世纪。1969 年，Waldinger & Lee（1969）提出“PROW”，该系统接受以谓词演算编写的程序规范，生成算法并创建 LISP 实现（McCarthy, 1978）。Balzer（1985）和 Soloway（1986）致力于推进自动编程，并指出实现它的潜在方法。近期方法使用自然语言处理（NLP）技术（Ni et al., 2023；Skreta et al., 2023；Feng et al., 2020；Li et al., 2022；Chen et al., 2018；2021b；Zhang et al., 2023）。自动编程已经发展成为一个产业，提供 Microsoft Copilot 等付费功能。近来，基于 LLM 的智能体（Yao et al., 2022；Shinn et al., 2023；Lin et al., 2023）推动了自动编程的发展。其中，ReAct（Yao et al., 2022）和 Reflexion（Shinn et al., 2023）使用思维链提示（Wei et al., 2022），借助 LLM 生成推理轨迹与行动计划。两项工作都证明，ReAct 风格的推理循环是增强自动编程能力的一种有效设计范式。此外，ToolFormer（Schick et al., 2023）可以通过简单的 API 学习如何使用外部工具。与本文最密切相关的 Li et al.（2023）提出了一个用于编程的简单角色扮演框架，让扮演不同角色的智能体相互交流。Qian et al.（2023）使用多个智能体进行软件开发。尽管已有论文（Li et al., 2023；Qian et al., 2023）提高了生产力，但它们尚未充分利用具有结构化输出格式的有效工作流。这使其难以处理复杂的软件工程问题。

==**基于 LLM 的多智能体框架。** 近期，基于 LLM 的自主智能体受到工业界和学术界的广泛关注（Wang et al., 2023b）。许多工作（Chen et al., 2024；Wang et al., 2023c；Du et al., 2023；Zhuge et al., 2023；Hao et al., 2023；Akata et al., 2023）通过整合多个智能体之间的讨论，提高了 LLM 的问题求解能力。Stable-Alignment（Liu et al., 2023）通过 LLM 智能体在沙盒中的交互，就价值判断达成共识，从而创建指令数据集。另一些工作关注社会学现象。例如，Generative Agents（Park et al., 2023）创建了一个由 25 个智能体组成的“小镇”，用于研究语言交互、社会理解和集体记忆。在基于自然语言的心智社会（Natural Language-Based Society of Mind，NLSOM）（Zhuge et al., 2023）中，不同功能的智能体通过多轮“心智风暴”（mindstorms）相互交互，以解决复杂任务。Cai et al.（2023）提出一种降低成本的模型：将用作工具制造者的大模型与用作工具使用者的小模型结合起来。==

==有的工作强调与规划和策略有关的合作与竞争（Bakhtin et al., 2022），还有的提出基于 LLM 的经济系统（Zhuge et al., 2023）。这些工作侧重于开放世界中的人类行为模拟，而 MetaGPT 旨在将人类实践引入多智能体框架。此外，基于 LLM 的智能体还面临“助手重复指令”或“消息无限循环”的挑战（Talebirad & Nadiri, 2023；Li et al., 2023）。在面向任务的协作中，这些挑战更加迫切，因为此类协作要求一致且互惠的交互（Elazar et al., 2021；Wang et al., 2022；Jiang et al., 2023）。这促使我们着重将软件开发中的标准操作规程等先进理念应用到多智能体框架中。== 

![图2：通信协议与带可执行反馈的迭代编程](assets/metagpt-figure2.png)

**图2：通信协议（左）与带可执行反馈的迭代编程（右）示例。** 左：智能体使用共享消息池发布结构化消息，也可以根据自身角色档案订阅相关消息。右：生成初始代码后，工程师智能体运行代码并检查错误。如果出现错误，它会检查记忆中保存的历史消息，并将这些消息与 PRD、系统设计及代码文件进行对照。

**图2内文字译注：** 左侧包含共享消息池、订阅、发布，以及网页搜索工具、调试工具和绘图工具。结构化消息的字段 `content` 为内容，`instruct_content` 为指令内容，`cause_by: WriteTasks` 表示由任务编写动作触发，`sent_from: ProjectManager` 表示发自项目经理，`send_to: Engineer` 表示发往工程师。右侧智能体配置：姓名 Alex，角色为工程师，目标是“编写优雅、可读、可扩展、高效的代码”，约束为“代码应符合 PEP8 等编码标准，采用模块化设计，易于阅读与维护”。记忆检索取回包含 PRD 文档、系统设计和代码的历史消息；迭代编程过程为编写 `game.py` 后执行，出错则调试并反馈，成功则添加代码。

## 3 MetaGPT：一种元编程框架

MetaGPT 是一个面向基于 LLM 的多智能体系统的元编程框架。第 3.1 节解释框架中的角色专门化、工作流和结构化通信，并说明如何在 SOP 语境下组织多智能体系统。第 3.2 节提出一种提高角色间通信效率的通信协议。我们还实现了结构化通信接口和有效的发布-订阅机制。这些方法让智能体既能从其他角色获取定向信息，也能从环境中获取公共信息。最后，第 3.3 节介绍可执行反馈（executable feedback，即运行代码和测试所得的反馈）：一种在运行期间进一步提高代码生成质量的自我纠正机制。

### 3.1 标准操作规程中的智能体

**角色专门化。** 明确的角色专门化能够将复杂工作分解为更小、更具体的任务。解决复杂任务或问题通常需要具备不同技能与专长的智能体协作，每个智能体都针对特定问题提供专业化输出。

在软件公司中，产品经理通常负责面向业务的分析并形成洞见，而软件工程师负责程序编写。如图1所示，我们在模拟的软件公司中定义了五种角色：产品经理、架构师、项目经理、工程师和质量保证工程师（QA 工程师）。在 MetaGPT 中，我们为每个角色指定智能体的档案，其中包括名称、角色描述、目标和约束。我们还为每个角色初始化特定的上下文和技能。例如，产品经理可以使用网络搜索工具，而工程师可以执行代码，如图2所示。所有智能体都遵循 Yao et al.（2022）所述的 ReAct 风格行为。每个智能体都监测环境（即 MetaGPT 中的消息池），以发现重要观察结果，例如来自其他智能体的消息。这些消息既可以直接触发行动，也可以辅助完成工作。

**智能体之间的工作流。** 通过定义智能体的角色与操作技能，可以建立基本工作流。在本文中，我们遵循软件开发的 SOP，使全部智能体按顺序工作。

![图3：MetaGPT 软件开发过程](assets/metagpt-figure3.png)

**图3：MetaGPT 软件开发过程示意图，突出其对 SOP 的显著依赖。** 更详细的演示见附录 B。

**图3内文字译注（代码名、文件名及 UML 符号保留原文）：**

- 人类输入：“制作 2048 滑块数字益智游戏。”人类直接操作游戏。
- 产品经理（`WritePRD`，编写 PRD）：思考“老板要求制作 2048 滑块数字益智游戏”。产品目标为“创建易于游玩的网页版 2048 游戏，并加入提高用户参与度与留存率的功能”。用户故事包括“作为用户，我想轻松了解游戏玩法，以便立即开始”和“作为用户，我希望游戏足够有趣，能够长时间游玩”。竞品分析：Gabriele Cirulli 的原版 2048 简单易玩；Ketchapp 的 2048 有多种游戏模式；Androbaby 的 2048 界面整洁并可跟踪最高分。需求分析：“产品应是易于使用的 2048 滑块数字益智游戏。”需求池：“开发简单直观的用户界面”（P0）、“实现引人入胜的游戏机制”（P0）。竞品象限图横轴为覆盖范围（低至高），纵轴为参与度（低至高）；左上“需要推广”，右上“应扩大规模”，左下“重新评估”，右下“可以改进”。
- 架构师（`WriteDesign`，编写设计）：思考“本项目将使用 Pygame，这是一个用于以 Python 制作电子游戏的开源库”；产出程序调用流程和文件列表。
- 项目经理（`WriteTasks`，编写任务）：思考“列出符合需求和设计的任务清单”。共享知识：“`game.py` 包含 `Game` 类，其中包括游戏逻辑。”任务列表包含 `main.py`、`game.py`、`gui.py`、`database.py`。逻辑分析：“`main.py` 包含游戏主循环并处理用户输入”；“`game.py` 实现游戏逻辑，包括得分和游戏结束条件”。
- 工程师（`WriteCode`，编写代码）和 QA 工程师（`WriteCodeReview`，编写代码审查）：图中重复上述 Pygame 的思考；工程师输出程序代码，QA 工程师进行代码质量审查（`test_gui.py`）。各阶段间的询问为“有不清楚的地方吗？”


具体而言，如图1所示，获取用户需求后，产品经理进行深入分析，制定详细的 PRD，其中包括用户故事（User Stories）和需求池（Requirement Pool），形成初步的功能分解。然后，结构化 PRD 交给架构师，架构师将需求转化为系统设计组成部分，例如文件列表、数据结构和接口定义。信息被纳入系统设计后，传递给项目经理进行任务分配。工程师按照说明实现指定的类和函数（详见图2）。接下来，QA 工程师制定测试用例，严格把控代码质量。最后，MetaGPT 产出精心构建的软件解决方案。我们提供了 MetaGPT 中 SOP 工作流的详细示意图（图3）和具体实例（附录 B）。

### 3.2 通信协议

**结构化通信接口。** 目前大多数基于 LLM 的多智能体框架（Li et al., 2023；Zhuge et al., 2023；Zhang et al., 2023；Park et al., 2023）采用不受约束的自然语言作为通信接口。

然而，尽管自然语言适用范围广，仍有一个问题：纯自然语言通信是否足以解决复杂任务？例如，在传话游戏（telephone game，亦称 Chinese whispers）[^2]中，经过多轮传递，原始信息可能发生很大失真。受到人类社会结构的启发，我们提出使用结构化通信来规范智能体之间的交流。我们为每种角色建立模式和格式，并要求各个智能体根据自身的具体角色与上下文提供必要输出。

如图3所示，架构师智能体生成两项输出：系统接口设计和时序流程图。它们包含系统模块设计和交互顺序，是交付给工程师的重要产物。与 ChatDev（Zhao et al., 2023）不同，MetaGPT 中的智能体通过文档和图示（结构化输出）而非对话进行通信。这些文档包含所有必要信息，从而防止内容无关或缺失。

**发布-订阅机制。** 信息共享在协作中至关重要。例如，架构师和工程师经常需要查阅 PRD。然而，按照先前工作（Li et al., 2023；Zhao et al., 2023；Zhang et al., 2023）的方式，每次都一对一地传达这些信息，会使通信拓扑变复杂，导致效率低下。

应对这一挑战的一种可行方法是将信息存储在全局消息池中。如图2（左）所示，我们引入共享消息池，允许所有智能体直接交换消息。智能体不仅将自身的结构化消息发布到池中，还能透明地访问其他实体的消息。任意智能体都可以直接从共享池中检索所需信息，无须向其他智能体询问并等待回复，从而提高通信效率。

将全部信息共享给每个智能体可能导致信息过载。在任务执行过程中，智能体通常希望只接收与任务相关的信息，避免被无关细节干扰。因此，有效管理和传播这些信息至关重要。

我们提出一种简单而有效的解决方案：订阅机制（图2左）。智能体根据特定角色的关注点提取相关信息，而不依赖对话。它们可以根据角色档案选择要关注的信息。在实际实现中，智能体只有接收到全部前置依赖信息后才会启动行动。如图3所示，架构师主要关注产品经理提供的 PRD，而 QA 工程师等其他角色的文档可能不是其主要关注对象。

### 3.3 带可执行反馈的迭代编程

在日常编程任务中，调试和优化过程发挥着重要作用。然而，现有方法往往缺乏自我纠正机制，导致代码生成失败。先前工作引入了不执行代码的代码审查与自我反思（Zhao et al., 2023；Yao et al., 2022；Shinn et al., 2023；Dong et al., 2023），但在确保代码可执行性和运行时正确性方面仍然面临挑战。

由于 LLM 幻觉（Manakul et al., 2023），MetaGPT 的早期实现在审查过程中会漏掉某些错误。为此，我们在初次代码生成后引入可执行反馈机制，以迭代改进代码。更具体地说，如图2所示，工程师被要求根据原始产品需求和设计编写代码。

这种机制使工程师能够利用自身的历史执行与调试记忆，持续改进代码。为了获取更多信息，工程师编写并执行相应的单元测试用例，随后获得测试结果。如果结果令人满意，就开始其他开发任务；否则，工程师先调试代码，再继续编程。这一迭代测试过程持续进行，直到测试通过，或达到最多 3 次重试。

[^2]: 原文脚注2：[Chinese whispers](https://en.wikipedia.org/wiki/Chinese_whispers)。

## 4 实验

### 4.1 实验设置

**数据集。** 我们使用两个公开基准 HumanEval（Chen et al., 2021a）和 MBPP（Austin et al., 2021），以及自行构建、更具挑战性的软件开发基准 SoftwareDev：（1）HumanEval 包含 164 个手写编程任务，包括函数规范、描述、参考代码和测试。（2）MBPP 包含 427 个 Python 任务，覆盖核心概念和标准库功能，并包含描述、参考代码和自动测试。（3）SoftwareDev 数据集汇集 70 个具有代表性的软件开发任务示例，每个示例都有自己的任务提示（见表8）。这些任务覆盖不同范围（见图5），例如小游戏、图像处理算法和数据可视化，为真实开发任务提供了稳健的测试平台。与以往数据集（Chen et al., 2021a；Austin et al., 2021）不同，SoftwareDev 侧重工程方面。在对比中，我们随机选择七个具有代表性的任务进行评估。

**评估指标。** 对于 HumanEval 和 MBPP，我们遵循 Chen et al.（2021a）和 Dong et al.（2023）提出的无偏 Pass@k，评估前 $k$ 个生成代码的功能准确性：

$$
\operatorname{Pass@}k = \mathbb{E}_{\mathrm{Problems}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right].
$$

> 公式符号释义（译注）：$n$ 是每道题生成的代码样本总数，$c$ 是其中通过测试的样本数，$k$ 是考虑的样本数；期望取遍待评估问题。

对于 SoftwareDev，我们优先考虑实际用途，并通过人工评估（A、E）或统计分析（B、C、D）衡量性能：

- **（A）可执行性：** 对代码从 1（失败／不能正常工作）到 4（完全正确）评分。“1”表示不能正常工作，“2”表示可以运行但不完善，“3”表示接近完全正确，“4”表示完全正确。
- **（B）成本：** 本文的成本评估包括（1）运行时间、（2）token 使用量及（3）费用。
- **（C）代码统计：** 包括（1）代码文件数、（2）每个文件的代码行数及（3）总代码行数。
- **（D）生产力：** 基本定义为 token 使用量除以代码行数，即每行代码消耗的 token 数。
- **（E）人工修改成本：** 指人工纠正代码的次数，用来解决包导入错误、类名错误或引用路径不完整等问题。通常每次纠正涉及不超过 3 行代码。

**基线。** 我们将方法与近期代码生成领域的专用 LLM 进行比较，包括 AlphaCode（Li et al., 2022）、Incoder（Fried et al., 2022）、CodeGeeX（Zheng et al., 2023）、CodeGen（Nijkamp et al., 2023）、CodeX（Chen et al., 2021a）和 CodeT（Chen et al., 2022），以及 PaLM（Chowdhery et al., 2022）与 GPT-4（OpenAI, 2023）等通用 LLM。部分基线（如 Incoder、CodeGeeX）的结果由 Dong et al.（2023）提供。在 HumanEval 和 MBPP 中，我们对提示做了少量修改，以符合响应格式要求。这些修改旨在处理特定格式问题，即 Python 题目的格式问题。对于 SoftwareDev 基准，我们全面比较 MetaGPT、AutoGPT（Torantulino et al., 2023）、配备 Python“读取-求值-打印循环”（Read-Eval-Print Loop，REPL）工具[^3]的 LangChain（Chase, 2022）、AgentVerse（Chen et al., 2023）以及 ChatDev（Qian et al., 2023）。

### 4.2 主要结果

![图4：MBPP 与 HumanEval 单次尝试的通过率](assets/metagpt-figure4.png)

**图4：MBPP 与 HumanEval 上单次尝试的通过率。** 图中“w/o Feedback”表示“不使用反馈”；“Pass@1 of MBPP and HumanEval (%)”表示“MBPP 与 HumanEval 的 Pass@1（%）”。

**性能。** 图4表明，MetaGPT 在 HumanEval 和 MBPP 两个基准上都优于所有先前方法。MetaGPT 与 GPT-4 配合时，相较单独使用 GPT-4，显著提高了 HumanEval 基准上的 Pass@k。它在这两个公开基准上分别达到 85.9% 和 87.7%。

[^3]: 原文脚注3：[Read-eval-print loop](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)。

![图5：MetaGPT 开发的软件演示](assets/metagpt-figure5.png)

**图5：由 MetaGPT 开发的软件演示。**

**图5内界面说明：** 拼图展示植物识别、打地鼠、绘图、情感分析、太空入侵者、数据分析、时钟、2048、扫雷、花店、客户管理、系统性能、数独、手写板、连珠、日期差计算、音乐网站、批量文件更名、井字棋、天气穿衣建议、图像旋转、打砖块、亲属关系计算、词云、交通密度热图、备忘录、Excel 数据处理与 CSV 处理等演示。原始界面截图完整保留。

此外，如表1所示，在具有挑战性的 SoftwareDev 数据集上，MetaGPT 几乎在全部指标上都优于 ChatDev。例如，在可执行性方面，MetaGPT 得分为 3.75，十分接近 4（完全正确）。此外，它所需时间较少（503 秒），明显少于 ChatDev。就代码统计和人工修改成本而言，MetaGPT 也显著优于 ChatDev。虽然 MetaGPT 需要更多 token（24,613 或 31,255，而 ChatDev 为 19,292），但生成一行代码仅需 126.5 / 124.3 个 token；相比之下，ChatDev 需要 248.9 个 token。这些结果突出体现了 SOP 在多智能体协作中的益处。另外，我们通过可视化示例展示了 MetaGPT 自主生成软件的能力（图5）。更多实验与分析见附录 C。

> 译注：本段原文将 503 秒写作 MetaGPT 的时间；按照表1，该数值对应“不使用反馈的 MetaGPT”，完整 MetaGPT 对应 541 秒。这里保留原文表述，表中数值亦按底本保留。

**表1：SoftwareDev 上的统计分析。**

| 统计指标 | ChatDev | 不使用反馈的 MetaGPT | MetaGPT |
|---|---:|---:|---:|
| （A）可执行性 | 2.25 | 3.67 | **3.75** |
| （B）成本 #1：运行时间（秒） | 762 | **503** | 541 |
| （B）成本 #2：token 使用量 | **19,292** | 24,613 | 31,255 |
| （C）代码统计 #1：代码文件数 | 1.9 | 4.6 | **5.1** |
| （C）代码统计 #2：每个文件的代码行数 | 40.8 | 42.3 | **49.3** |
| （C）代码统计 #3：总代码行数 | 77.5 | 194.6 | **251.4** |
| （D）生产力 | 248.9 | 126.5 | **124.3** |
| （E）人工修改成本 | 2.5 | 2.25 | **0.83** |

### 4.3 能力分析

与 AutoGPT 等开源基线方法，以及 AgentVerse 和 ChatDev 等自主智能体相比，MetaGPT 提供了面向软件工程任务的功能。如表2所示，我们的框架涵盖广泛能力，能够高效处理复杂、专业化的开发任务。融入 SOP（例如角色扮演专长、结构化通信、顺畅工作流）能够显著改进代码生成。

**表2：MetaGPT 与其他方法的能力比较。** ✓ 表示相应框架具备该特性，✗ 表示不具备。

| 框架能力 | AutoGPT | LangChain | AgentVerse | ChatDev | MetaGPT |
|---|:---:|:---:|:---:|:---:|:---:|
| 生成 PRD | ✗ | ✗ | ✗ | ✗ | ✓ |
| 生成技术设计 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 生成 API 接口 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 生成代码 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 预编译执行 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 基于角色的任务管理 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 代码审查 | ✗ | ✗ | ✓ | ✓ | ✓ |

**表3：角色消融研究。** “#”表示“数量”；“产品”表示“产品经理”，“项目”表示“项目经理”。✓ 表示加入相应角色。“修改次数”指“人工修改成本”。

| 工程师 | 产品经理 | 架构师 | 项目经理 | 智能体数 | 代码行数 | 费用 | 修改次数 | 可执行性 |
|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| ✓ | ✗ | ✗ | ✗ | 1 | 83.0 | **\$0.915** | 10 | 1.0 |
| ✓ | ✓ | ✗ | ✗ | 2 | 112.0 | \$1.059 | 6.5 | 2.0 |
| ✓ | ✓ | ✓ | ✗ | 3 | 143.0 | \$1.204 | 4.0 | 2.5 |
| ✓ | ✓ | ✗ | ✓ | 3 | 205.0 | \$1.251 | 3.5 | 2.0 |
| ✓ | ✓ | ✓ | ✓ | **4** | **191.0** | \$1.385 | **2.5** | **4.0** |

其他基线方法也可以方便地融入类似 SOP 的设计来改进性能，就像向 LLM 注入思维链（Wei et al., 2022）一样。

### 4.4 消融研究

**角色的有效性。** 为理解不同角色对最终结果的影响，我们执行两个任务，生成有效代码并计算平均统计量。当移除某些角色时，会生成无法正常工作的代码。如表3所示，在只有工程师的基础上加入其他角色，始终能够改善修改次数和可执行性。虽然增加角色会使费用略有上升，但整体性能明显提升，证明了不同角色的有效性。

**可执行反馈机制的有效性。** 如图4所示，在 MetaGPT 中加入可执行反馈后，HumanEval 和 MBPP 的 Pass@1 分别显著提升 4.2 和 5.4 个百分点。此外，表1显示，反馈机制提高了可行性（从 3.67 提高到 3.75），并降低人工修改成本（从 2.25 降低到 0.83）。这些结果说明，我们设计的反馈机制能够产出更高质量的代码。表4和表9给出了 MetaGPT 以及不使用可执行反馈的 MetaGPT 的更多定量结果。

## 5 结论

本文提出 MetaGPT：一种新的元编程框架，利用 SOP 增强基于大语言模型（LLM）的多智能体系统的问题求解能力。MetaGPT 将一组智能体建模为模拟的软件公司，类似于模拟小镇（Park et al., 2023）以及 Voyager 中的 Minecraft 沙盒（Wang et al., 2023a）。MetaGPT 利用角色专门化、工作流管理，以及消息池和订阅等高效共享机制，成为适用于自主智能体和多智能体框架的灵活、可移植平台。它使用可执行反馈机制，在运行期间提高代码生成质量。在大量实验中，MetaGPT 在多个基准上取得最先进性能。成功融入人类式 SOP，为人工多智能体系统中受人类启发的技术研究提供了启示。我们也将本工作视为规范基于 LLM 的多智能体框架的一次早期尝试。另见展望（附录 A）。

## 致谢

感谢 KAUST 人工智能计划执行秘书 Sarah Salhi，以及 KAUST 人工智能计划博士后研究员 Yuhui Wang 帮助润色部分文字。我们感谢 KAUST 人工智能计划博士生 Wenyi Wang 对论文提供全面反馈，并与 Mingchen 一起协助起草展望（附录 A）。我们也感谢 DeepWisdom 副总裁 Zongze Xu 提供 AgentStore 的说明素材。

## 作者贡献

Sirui Hong 完成了大部分实验并设计了可执行反馈模块。她在 Ceyao Zhang，以及 Jinlin Wang 和 Zili Wang 的支持下，主导了初稿撰写。Mingchen Zhuge 设计了自我改进模块，讨论了额外实验，并主导了当前版本的撰写。Jonathan Chen 协助开展 MBPP 实验，拟定方法部分的提纲，并参与当前版本的撰写。Xiawu Zheng 提供了宝贵指导，并审阅和编辑论文。Yuheng Cheng 参与评估指标设计及 HumanEval 实验。Steven Ka Shing Yau、Zijuan Lin、Liyang Zhou 和 Lingfeng Xiao 协助开展 MBPP 实验，以及与开源基线方法的比较。Chenyu Ran 绘制了大部分说明图。Chenglin Wu 是 DeepWisdom 的首席执行官，发起了 MetaGPT，对其代码作出了最重要的贡献，并指导了本项目。Jürgen Schmidhuber 是 KAUST 人工智能计划主任和 IDSIA 科学主任，指导了本项目并协助论文撰写。

## 参考文献（原文保留，不翻译）

以下保留底本第 10—14 页参考文献的原始版面；第 10 页图片仅裁取参考文献部分，致谢与作者贡献已在上文翻译。

![参考文献，底本第10页](assets/metagpt-original-page10.png)

![参考文献，底本第11页](assets/metagpt-original-page11.png)

![参考文献，底本第12页](assets/metagpt-original-page12.png)

![参考文献，底本第13页](assets/metagpt-original-page13.png)

![参考文献，底本第14页](assets/metagpt-original-page14.png)

## 附录（原文保留，不翻译）

为完整保留参考信息、图表、提示示例与代码，以下按底本页序嵌入附录原页。附录中的命令、提示和角色指令均为论文内容。

### A Outlook

![附录A，底本第15页](assets/metagpt-original-page15.png)

![附录A，底本第16页](assets/metagpt-original-page16.png)

### B A Demo of the Execution

![附录B，底本第17页](assets/metagpt-original-page17.png)

![附录B，底本第18页](assets/metagpt-original-page18.png)

![附录B，底本第19页](assets/metagpt-original-page19.png)

![附录B，底本第20页](assets/metagpt-original-page20.png)

![附录B，底本第21页](assets/metagpt-original-page21.png)

![附录B，底本第22页](assets/metagpt-original-page22.png)

### C Experiments

第 23 页上部续接附录 B，下部开始附录 C；按原版完整保留。

![附录B末尾及附录C，底本第23页](assets/metagpt-original-page23.png)

![附录C，底本第24页](assets/metagpt-original-page24.png)

### D Limitation and Ethics Concerns

第 25 页上部续接附录 C，下部开始附录 D。

![附录C末尾及附录D，底本第25页](assets/metagpt-original-page25.png)

### E More Discussions

![附录及浮动表格，底本第26页](assets/metagpt-original-page26.png)

![附录及浮动表格，底本第27页](assets/metagpt-original-page27.png)

![附录及浮动表格，底本第28页](assets/metagpt-original-page28.png)

![附录及浮动表格，底本第29页](assets/metagpt-original-page29.png)

## 版本核查记录（译者补充）

截至 2026-10-28，[arXiv 记录](https://arxiv.org/abs/2308.00352)显示最新作者稿为 **v7（2024-11-01）**。[v7 PDF](https://arxiv.org/pdf/2308.00352v7) 与正式会议站提供的 PDF 均为 29 页，但不是同一文件。对两者逐页文字层进行比较，主要发现：

1. 正式站底本的页眉为“Preprint”；v7 改为“Published as a conference paper at ICLR 2024”，并带有 arXiv 版本标识。
2. 正式站底本及正式论文页使用作者名 **Jonathan Chen**；v7 的作者列表和贡献段使用 **Jiaqi Chen**。本译文忠实保留底本姓名。
3. v7 在引言和相关工作中补充若干文献及 ToolLLM 等表述，并相应扩充参考文献，包括 Data Interpreter、ML-Bench、OctoPack、ToolLLM、BioCoder、MedAgents、Igniting Language Intelligence 和 Agents 等；部分引文的年份后缀随之调整。
4. 文字层比较未见第 3 节方法、第 4 节实验数据、第 5 节结论及附录的实质性内容改动；这不等于声称两份 PDF 的图像二进制一致。

因此，本译文采用**正式会议站所提供的免费会议版本**，不把较晚上传的作者修订稿自动视为正式最终版，也未将其后补引文和作者改名混入底本。

### 术语说明

| 原文 | 本译文 | 核对依据或含义 |
|---|---|---|
| metaprogramming / meta-programming | 元编程 | [DolphinDB 官方中文文档](https://docs.dolphindb.com/zh/tutorials/hybrid_programming_paradigms.html)采用此译名；本文特指“通过编程来编程”。 |
| Standardized / Standard Operating Procedures（SOPs） | 标准操作规程 | [中国国家流感中心官方 SOP 文件](https://ivdc.chinacdc.cn/sysgl/swaq/jszl/201203/P020120323392850626391.pdf)采用此中英文对应。 |
| publish-subscribe | 发布-订阅 | [AWS 官方中文文档](https://docs.aws.amazon.com/zh_cn/prescriptive-guidance/latest/cloud-design-patterns/publish-subscribe.html)。 |
| executable feedback | 可执行反馈 | 未找到作者官方固定中文译名；保留英文对应，按第 3.3 节理解为执行代码及测试所得的反馈。 |
| Product Manager / Architect / Project Manager / Engineer | 产品经理／架构师／项目经理／工程师 | [MetaGPT 作者官方中文介绍](https://docs.deepwisdom.ai/main/zh/guide/get_started/introduction.html)。 |
| QA Engineer | 质量保证工程师 | QA 为 Quality Assurance，文中简称 QA 工程师。 |
| language agent / workflow / prompt | 语言智能体／工作流／提示 | 与配套 GPTSwarm 译文保持一致。 |

保留模型名称、数据集名称、API、代码及文件名；token 保留英文，以免与字符或单词混淆。表格、图中数值与论文提出的性能主张均对应底本实验时点。
