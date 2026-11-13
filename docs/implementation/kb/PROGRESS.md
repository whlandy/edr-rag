# KB 实施进度（引用证据目录，不复制日志）

## 2026-11-12 [1745] 已用 holdout 迁移+新留出集（[1714] 待办①闭环）
- pro-11/12/14 split holdout→dev（[1433审核§3]；迁移前快照留证）
- 新增 h-01..04 未见留出集：alert-definition×2（告警级别枚举/告警-事件对比，
  仅 duli 含「管理业务告警」专节）、terminal-auto-grouping（NAT 映射，双手册
  同 ID 近同文）、no-answer（打印机驱动，八语料 0 命中，避开「微信」他义词面）
- 评测集 29→33 主查询，dev 18/holdout 15，组 20→22 与既有零重叠；
  verify_sources PASS
- 固定快照回归（hybrid 117 变体，隔离索引）：旧 103 变体 duli 逐查询
  0 变化；anquan 2 条变化=[1714] 已知 #dup 效应完全复现，零新增；
  anquan h-01/02 全 0=预期（gold 本不在 anquan 语料，per-manual workdir
  如实反映）
- h-* provisional（程序化核验，待人工评审）；冻结：8 JSONL sha256 与
  [1714] 逐一相同，正式 vector_index mtime 精确一致，代码仓库零改动
- 证据：optimization-runs/20261112-1745-holdout-migration/
- 下一最小任务：① ingest 层 ID 碰撞根因 ② 跨修订变体版本维度（P2 输入）
  ③ 词表消融（[1112] 待办②）④ h-* 人工确认
- 需通知用户：否 — holdout 隔离语义按计划恢复

## 2026-11-12 [1714] 同 ID 变体入库后双手册 hybrid 重跑对照（[1617] 待办①闭环）
- 隔离索引（EDR_RAG_VECTOR_DIR）双手册各 P1/P2 双进程逐行一致；命令与
  [1537] 同参（hybrid 103 变体 --paraphrases），vector=active ×4 进程
- **duli 与 [1433] 真基线 103 条逐查询完全一致（0 变化）**，全部指标逐位相同
- **anquan 103 条中 2 条变化**（R@K/Success/in_vocab/oov/引用正确率全部
  持平；MRR −0.0057 / nDCG −0.0043）：a-02#p1 gold rank1→2、vd-02#p2
  gold rank3→4，新 rank1 均为 `登录_管理面…操作步骤`——[1617] 修复前
  upsert 覆盖不可检索、修复后可检索的跨修订 #dup 变体（磁盘 9 行/5 内容），
  差值与汇总精确吻合；定性=修复直接效果非回归
- 新发现登记为 P2 设计输入：同 ID 跨修订变体检索层无版本维度，多修订
  变体可共同占据 top-k 挤压异 ID gold（对应设计 B01）
- 冻结：8 JSONL sha256 逐一相同；正式 vector_index mtime 与 [1617] 记录
  相同（零写盘）；公开 ID 0 #dup 泄漏；评测集/设计文档哈希无变更
- 证据：optimization-runs/20261112-1714-dupid-hybrid-replay/
- 下一最小任务：① 已用 holdout 迁移+新留出集 ② ingest 层 ID 碰撞根因
  ③ 跨修订变体版本维度问题已登记 P2 输入（见本轮报告归因节）
- 需通知用户：否 — 待办①按计划闭环，行为与修复预期一致

## 2026-11-12 [1617] 同 ID 异内容身份契约修复（审核契约漏洞 #2 闭环）
- 量化：等价去重后仍同 ID 异内容——sections duli 4 组/anquan 6 组，
  chunks 67/119 组，nodes 11/17 组，procedures 9/14 组；真实形状=duli
  「登录_管理面」同 ID 三变体浏览器要求实质不同（版本差异非等价副本）
- 缺陷两形状：整批索引 upsert 后写覆盖只留最后变体向量（其余向量路
  静默不可检索）；增量索引按 ID 收敛——前位变体变化以最后变体文本在
  错误位置重嵌入（doc_idx 错乱）
- 修复：JsonlCorpus 读取侧消歧（`_normalize` 后调用，变体
  `_doc_id=<id>#dupN`，公开 id 不动，`_to_result` 输出原始 id——gold/
  引用/get() 零语义变化）；`_embed_upsert`/`index_records_incremental`
  同批防御同一方案；磁盘 JSONL 不修改
- TDD 5 例红→绿（git stash 复验实施前全红）；相邻 92 + 模块 386 passed；
  全量 **1092 passed**（1087+5），29 errors 基线即有，零回退
- 真实链（正式 workdir 只读）：双手册 from_file 后 upsert 键全唯一
  （duli 1494/1494 含 5 变体、anquan 1355/1355 含 9 变体）、公开 ID 0
  泄漏、离线 keyword/bm25 结果无 #dup 泄漏；vector_index 6 meta mtime
  全未动；sections/eval_set sha256 与 [1605] 逐一相同
- 证据：optimization-runs/20261112-1617-duplicate-id-identity/
- 下一最小任务：① 隔离索引双手册 hybrid 重跑对照真基线（变体入库后
  向量路 hit 集变化）② holdout 迁移+新留出集 ③ ingest 层 ID 碰撞根因
  （关联方案 A/B/C 决策，[1617] 已提供碰撞范围量化）
- 需通知用户：否 — 审核确认契约漏洞 #2 按计划闭环

## 2026-11-12 [1605] 模型指纹持久化修复（审核契约漏洞 #1 闭环）
- 缺陷形状（[1433 审核 §54] 源码确认）：`_restore_sig_from_store` 用当前
  embedder 拼接磁盘旧 content_sig，跨进程换模型必然"匹配"，静默服务旧
  模型向量；同进程测试测不到（不经磁盘恢复路径）
- 修复：PersistentVectorStore meta 新增 `embedder_sig` 字段 + 存取器；
  VectorIndex 恢复短路前校验指纹（缺失=旧格式 / 不匹配 → fail-closed
  warning + 强制重建）；`_note_indexed_sig` 写回 content+embedder 双签名
- TDD 4 例红→绿（跨进程换模型负例 / 旧格式 fail-closed / 同模型短路
  防回退 / meta 字段落盘）；vectorstore 34 passed；相邻回归
  （channel/singleton/incremental/dedupe/doctype/synrep/security）
  122+21 passed；p1/p2/eval/llm 252 passed 1 skipped
- 真实链验证：隔离 /tmp 三态行为（A 落盘含指纹 → 换 B 拒绝复用重建 →
  同 B 短路保留）；正式 anquan sections meta（旧格式 count=1344）只读
  load 兼容、指纹 None、目录 6 文件 sha256+mtime 逐字节未变（零写盘）
- 冻结：8 JSONL 与 [1433]/[1537] 逐一相同；正式 vector_index mtime 全未动；
  eval_set_v2 哈希相同
- 证据：optimization-runs/20261112-1605-model-fingerprint/（REPORT.md +
  VERSIONS.txt + diff.patch）
- 下一最小任务：① 同 ID 异内容身份契约（审核契约漏洞 #2，高优先级）
  ② holdout 迁移+新留出集 ③ 词表消融在真基线上继续
- 需通知用户：否 — 审核确认契约漏洞按计划闭环

## 2026-11-12 [1112] P2 评测集扩批 23→29 主查询
- 新增 pro-09~pro-14 六主题（溯源狩猎跳数/资产来源/资产冲突判定/终端登记绑定/
  网络隔离白名单/防火墙规则上限），全部双手册同 ID 同文，verify_sources PASS
- 扩批后 hybrid 基线（103 变体）：duli R@K 0.8398 / anquan 0.7670（新集口径，
  与 79 变体旧基线不可直接比较）；pro-12/13 主查询零分为词法敏感（标题词回避
  改写可达），失败分析与 d-04/pro-08 同型，不改题
- success@K 遗留确认：metrics/runner 已内置（0819 前接入），表格已渲染
- 冻结 8 JSONL 哈希与 [0819]/[0930]/[1031] 逐一相同；代码仓库零改动
- 证据：optimization-runs/20261112-1112-eval-p2batch/

## 2026-11-12 [20261112-0512] 内容级去重实施（P2）
- `JsonlCorpus` 读取侧两级去重 + `representative_id()` 公共映射口（edr_rag/p1/search.py）
- 6 新回归；p1+tests 404 passed；全量 1037 passed（29 service errors 基线即有 [1125]）
- 正式链评测：duli hybrid R@K 0.9462→0.9831、MRR 0.5806→0.6523，26 升/0 降；anquan 零回退
- [0310] pro-01/pro-07 内联口径结论修正（正式链 gold 本在 top-5）
- 事故：冻结 workdir vector_index 派生缓存被覆盖（JSONL 本体无损）；派生缓存冻结语义待确认
- 证据: optimization-runs/20261112-0512/（report.md、VERSIONS.txt、remapped eval 集、dedup workdir、逐行 rows）
- 未提交（工作树 dirty 属多轮累积，commit 决策属用户）

## 2026-11-12 [20261112-0625] synthesize 层接入 representative_id（P2）
- `synthesize_answer(rep_of=...)`：来源引用（steps/warnings/assertions/citations）
  指向被移除 ID 时归一为代表 + `same_content_as` 声明，不静默替换（[0440] 建议二）；
  `RetrievalService.representative_map()` 公共口，api/eval 两链路共用（[0440] 结论三）
- 9 新回归；p1+tests 413 passed；全量 1046 passed（29 基线 errors）
- 真实链实证：duli procedures step source 归一路径 + 两手册引用分 kind 可定位 0 失败；
  冻结 JSONL 本体 shasum 未动；procedures 向量缓存首次创建（[0512] 事故同类副作用）
- 证据: optimization-runs/20261112-0625/
- 未提交（工作树 dirty 属多轮累积）

## 20261112-0659 缺陷③最小修复（content|text 同口径）
- `_dedupe_content_rows` 取值口径 content → content|text（与 _normalize 一致，1 行）；
  新增 TestDedupeTextFieldCorpora 4 回归；p1 48 passed；全量 1050 passed（29 基线 errors）
- 真实 chunks 重放：anquan 4877→4456（-8.6%）/ duli 6354→5617（-11.6%）；
  duli 词法重放 0 未补偿回归；anquan 2 题未补偿移除，根因=「同 content 不同
  metadata.title」组去重改变检索可发现性（title/operation 参与 _searchable 打分）
- 影响面：chunks title 差异组 anquan 18 / duli 13，抽样含跨功能同文组
  （创建私有镜像 vs 创建虚拟机 等）——chunks 去重键需升级（方案 B：text+title），
  待决策（详见 DECISIONS.md D-20261112-0659-1）
- 证据: optimization-runs/20261112-0659/（report.md + VERSIONS.txt）

## 2026-11-12 08:13 [0745] D-0659-1 方案 B 实施
- `_dedupe_content_rows` 键升级 content|text+title+operation（1 处改动）+ 6 回归；
  TDD 红 3→绿 25；全量 1056 passed（基线 1050+6），29 errors 基线即有。
- 真实链：anquan chunks 4478（CA管理面 18/24 全保留，实施前 17）、duli 5635；
  sections 与 [0512] 逐数一致（1353/1473，零影响）。[0659] 两处未补偿移除恢复。
- 证据：optimization-runs/20261112-0745/（report.md + VERSIONS.txt + 3 重放脚本）。
  待用户追认方案 B（回退成本 1 行）。

## 2026-11-12 08:35 [0819] 缺陷④ DOCTYPE 泄漏两层修复
- 根因层 parse_html._walk：PreformattedString 前置排除（Doctype/Comment/PI
  均为其子类；bs4 4.15 需从 bs4.element 导入，带 fallback）。
- 防御层 JsonlCorpus._strip_doctype_leak：签名 `PUBLIC "//DTD…" "URL"`
  （前导 html 可缺，覆盖 chunks 中部型「恢复: PUBLIC …」）；构造函数 +
  from_file 去重前两处接入；content/text 双字段。
- 真实链：8 语料 10009 泄漏行 → 0，误伤 0；正则两轮收敛（首版残留 117）。
- 回归：+14 用例红→绿；全量 1070 passed（1056+14）零回退；bm25@5 词法重放
  strip 前后逐变体 0 翻转（anquan 69/80、duli 49/79；chunks 基线 0 既有口径）；
  冻结 JSONL sha256 与 [0659] 逐字节相同（本轮全程 in-memory 只读）。
- 影响面：_searchable 剥离 → 向量签名失配，下次查询一次性重建（正确性方向）。
- 证据：optimization-runs/20261112-0819/。

## 2026-11-12 11:00 [1031] nodes 语料重建（构建期剥离生效）
- 发现：磁盘冻结 sections/procedures 仍含泄漏（防御层只在运行时）→ 先修
  构建期（build_structured_nodes_file 输入过 _strip_doctype_leak，+1 回归）
  再重建双手册 nodes.jsonl（备份 + 原子写，[1449]/[1912] 惯例）。
- 验证：磁盘泄漏 0；逐行结构等价（关系字段 0 差异，文本差异全为 strip±
  空白归一化）；representative_id 对齐 0 失配；运行时路径端到端泄漏 0；
  validate-index 通过；全量 1075 passed（基线 1074+1）。
- 冻结保护：sections/procedures 哈希不变；vector_index 未动（nodes 向量
  下次查询一次性重建，既知方向）；nodes 旧产物备份于 run 目录 backup/。
- 证据：optimization-runs/20261112-1031-nodes-rebuild/。

## 2026-11-12 19:05 [1824] ingest 层 ID 碰撞根因消歧（范围限定 A，实现+离线验证）
- 新模块 id_disambiguation.py：sections 同 ID 异内容组 @fileslug 消歧 +
  procedure 层2消歧（_slug_path 空格折叠型，duli 登录业务面实证）；
  run_ingest / extract_procedures_from_file 两处接入，[1617] 读取层保留纵深。
- 离线快照（冻结产物只读）：消歧后全链重建 diff-body 0、upsert 键全唯一、
  引用零悬空（双手册）。
- 全量 1114 passed（基线 1092+22）零回退；正式产物零写盘。
- 决策 D-20261112-1824-1 需用户追认；gold 暴露面仅 a-03/a-04。
- 证据：optimization-runs/20261112-1824-id-disambiguation/。

## 2026-11-12 20:47 [2003] gold 同步+隔离重建+离线评测对照
- 隔离重建（/tmp/edr-rag-isolated-rebuild-1824）：消歧 sections→P1 全链，
  残余同 ID 组全部语义等价体、引用零悬空（双手册）。
- 发现 gold 契约问题：@0180928815/@0179303529 内容孪生，读取侧 content
  去重使单文件 gold 不可检索（旧共享 ID 记分虚高）；事实级 gold 修正
  （D-20261112-2003-1，自决待追认）。
- 117 变体逐查询对照：5 条变化全部归因（4 条虚高消除+fractional 口径、
  1 条 [1714] 同型变体入库排位效应）；无检索退化证据。
- 正式产物零写盘；代码仓库零改动。
- 证据：optimization-runs/20261112-2003-rebuild-goldsync/。

## 2026-11-12 21:30 [2003-closure] 用户六项指示最小收尾（任务终止）
- 候选固化：candidates/20261112-2003-id-disambiguation/（35 文件 SHA256 清单
  + 代码快照 + 复现命令；/tmp 原件保留；正式知识库未激活）。
- 消歧顺序依赖修复：fileslug 再碰撞序号按 relative_path 字典序（来源→ID
  映射输入顺序无关）；+3 回归；真实语料 0 序号后缀，候选 ID 零变化。
- 事实级 gold 修订（D-2003-1 修订版）：relevant 恢复旧 gold，fact_groups
  附加注解（命中任一成员覆盖该事实，组间独立）；metrics +fact_recall@k/
  fact_mrr（缺省不输出，普通文档级指标零变化）+ 正负例；verifier 扩展
  fact_groups 校验 + 4 例自测；runner 透传修复 + CLI 渲染。
- 受影响候选重建：8/8 产物逐字节一致；引用完整性 0 悬空；双手册真 hybrid
  重跑（隔离索引，vector=active，无静默降级）；对照归因见候选目录
  HYBRID-COMPARISON.md（文档级下降=共享 ID 虚高消除+变体竞争；a-03 事实
  factR=1.0；a-04 排位下降保留为已知项）。
- 全量测试：**1130 passed** + 7 skipped + **29 errors**——errors 全部为
  edr_rag/service/test_service.py fixture 作用域问题，与 [1044] 起逐轮
  记录的 29 errors 同源（单独运行该文件同样 29 errors，0.12s 复现），
  非本轮引入；**不是全绿**。
- 正式产物：5/5 JSONL sha256 与 [1824] 一致；duli vector_index 未动；
  anquan sections 派生缓存在对照期间按 [1537] 预告的显式重建路径重写
  （JSONL 本体无损，已记录）。
- 状态：候选已固化、正式版未激活、任务已停止（用户指示）。停等审核。
