任务卡
======

只记录了绝对路点的方案，无法跟随移动过的物体。而如果同时记录下 *定位了什么*
以及 *每个路点离那次读数有多远*，就可以：偏移是任务逻辑，换个布局依然成立，
坐标不是。

**任务卡**就是后一种形式的方案，从一个已解出的 episode 里录制一次。重放时被
替换的只有感知——动作、动作顺序、给策略的提示词、夹爪指令，全都来自任务卡本身，
定位器负责给出它们执行时的坐标。

语料
----

任务卡放在 ``resources/libero/task_card``，与 ``resources/libero/memory`` 下
整理好的 memory 并列，和这份 payload 的其余部分一起流转：从 HuggingFace 资源
数据集同步，或直接读磁盘上已有的副本。和 ``resources/`` 下的所有东西一样，该
目录不纳入 git。

一个任务只有一张卡，所以运行时不做任何挑选：任务决定了卡，卡加上实时定位就
生成轨迹。

.. code-block:: text

   resources/libero/task_card/
     index.json                 全部卡片：任务、来源 episode、任务指令
     object/swap_t3/
       anchors.json             定位过的短语、各自的读数，以及由此得到的锚点
       plan.json                每个动作，标出其坐标背后的锚点与偏移
       trace.md                 同样内容的可读版本，穿插记录 planner 的推理

语料里不出现 seed。一张卡无论重放到哪个布局上都服务于它那个任务；它从哪个
episode 录来，作为溯源信息保存在卡内部的 ``source`` 字段里，而不是一个可调项。

Molmo 配置
----------

重放用 **Molmo** 定位人工选取的像素，由
``rpent/robots/components/molmo_server.py`` 提供服务。SAM3 回答的是"哪些像素
是这个短语"，Molmo 回答的是"你会把夹爪放在哪里"——一个开放词表的点，面向那些
掩膜候选叫不出名字的短语。

Molmo 需要比策略更新的 ``transformers``，因此通常运行在自己的解释器下，通过
``PYTHONPATH`` 找到 RPent：

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

重放单个 episode
----------------

``--planner task_card`` 和 ``api``、``codex`` 一样是一个 planner 后端，只是由
任务卡决定动作，不调用任何模型。运行的其余部分完全不变：

.. code-block:: bash

   MOLMO_ENDPOINT=http://127.0.0.1:20703 \
     rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0

语料里每个任务只有一张卡，所以 seed 选的是要解决的布局，而不是用来解决它的
方案。

重放整轮扫描
------------

``robots.libero.task_card.run`` 不经过 CLI 驱动大量 episode。它连接已经在提供服务的策略、
分割器和定位器，并为每个 episode 启动一个环境服务：

.. code-block:: bash

   python -m robots.libero.task_card.run \
     --family object --tasks swap_t3 --seeds 0 1 2 \
     --vla-endpoint http://127.0.0.1:20701 \
     --sam3-endpoint http://127.0.0.1:20702 \
     --molmo-endpoint http://127.0.0.1:20703

评测是单次尝试且不重置环境：失败的 episode 记为失败，不会从干净状态重来。

读数如何变成路点
----------------

每个锚点都 **用它最初被读取的那个接口** 重新读取。``segment`` 来源的锚点重新
分割，因为它的偏移是相对掩膜质心算的，而斜视角下的宽口容器，其质心离指点模型
所指的位置有相当距离。只有记录时 planner 人工选取的像素才交给定位器回答。

定位分两级，与记录时 planner 本身的做法一致：先对开局画面粗看全场，然后机械臂停到
每个人工锚点上方，用腕部相机再问一次——此时物体填满视野。近距读数只在与粗读数
相差 5 厘米以内时才采纳，这样腕部视野里认错的东西不会覆盖掉正确答案。

对 *被握持* 物体的腕部读数会落在其中心之前，物体越高偏得越远。该修正与实测的
物体高度成线性关系。

配置项
------

两个入口都支持 ``--help``。要点如下：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 选项
     - 含义
   * - ``--family``
     - 扰动族：``object``、``goal``、``spatial`` 或 ``10``。
   * - ``--tasks``
     - 任务键，如 ``swap_t3 swap_t5``。默认该族下的全部卡。
   * - ``--seeds``
     - 每张卡要重放到哪些 seed 上。默认 ``0``--``9``。
   * - ``--cards``
     - 任务卡语料目录，默认 ``resources/libero/task_card``。
   * - ``--output-dir``
     - 每个 episode 的输出写到哪里。
   * - ``--vla-endpoint`` / ``--sam3-endpoint`` / ``--molmo-endpoint``
     - 已在提供服务的策略、分割器、定位器地址（仅批量重放）。

``--planner task_card`` 的定位器地址取自 ``MOLMO_ENDPOINT``——planner 的构建
早于机器人运行时解析自己的参数。

哪次录制成为了卡
----------------

一张卡是一个已解出的 episode，但并非每个解出的 episode 都是好卡。把
``object/swap_t0`` 全部八次成功录制各跑满十个 seed，成绩从 9/10 到 1/10，而
两个极端的动作数、pick 数、chunk 预算完全相同：解出一次只说明那个方案在那个
布局上有效，不说明它可迁移，而方案的形状看不出它属于哪一种。

所以语料里的卡是实测出可迁移的那一次录制，而不是看上去最好的那一次。这是语料
构建方式的属性，在发布前就已确定——运行时不存在选卡这件事。
