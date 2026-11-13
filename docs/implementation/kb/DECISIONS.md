# 实施决策记录

## D-20261112-0512-1 派生缓存冻结语义（待用户确认）
vector_index 目录属可重建派生缓存：[1654] 签名修复后，代码升级必然触发重建，
重建行集随读取侧语义（[1125]/本轮两级去重）变化。建议冻结承诺限定于 JSONL
产物本体 + manifest + source_chm；vector_index 豁免。历史 VERSIONS 若哈希了
vector_index 目录，后续轮应标注「派生缓存，不校验」。

## D-20261112-0512-2 内容级去重的代表选择依赖 jsonl 行序
代表 = 组内首现行。重 ingest 行序变化会换代表（内容相同，检索等价；ID 稳定性
弱）。gold 兼容性由 representative_id() 口保证，不依赖行序。

## D-20261112-0512-3 内联重放口径与正式链口径不可混用
[0310] 的内联重放（直接构造 JsonlCorpus + 默认参数）与 runner 完整链在部分
行为上不等价（如 pro-01#p2 冻结基线 rank 4 vs 内联 None）。后续可行性评估
若声称「正式链收益」，须用 runner 完整链重放验证。

## D-20261112-0652-1 缺陷④（DOCTYPE 泄漏）修复层级（待用户确认）
源头上 parse_html._walk 应排除 NavigableString 的非文本子类（Doctype/Comment/
Declaration/CData/ProcessingInstruction）——需重 ingest（破坏性，与 ingest A/B/C
决策合流）。非破坏替代是读取侧 content 归一化剥 doctype 前缀，但原始产物缺陷
留存、JSONL 与"原始来源不可变"原则的张力需在决策时权衡。建议：解析层修复 +
读取侧防御剥离两层都做；重 ingest 时机与 A/B/C 合并执行一次。

## D-20261112-0652-2 chunks 去重实施前置项
[0440] 结论三已明确：chunks 语料自身去重若实施，溯源映射（representative_map）
必须先于实施覆盖 chunks 的 source.section_id 引用面。缺陷③修复（字段名）与
该前置项同批，避免只修检索挤占、遗留引用断链。

## D-20261112-0659-1 chunks 内容级去重键升级（待决策，倾向 B）
缺陷③修复后 chunks 语料实际参与内容级去重（anquan -8.6% / duli -11.6%）。
但 chunks 与 sections 不同：组内 metadata.title 可不同（anquan 18 组 / duli 13 组），
title/operation 参与 _searchable 打分 → 并组改变检索可发现性。抽样含跨功能
同文组（创建私有镜像 vs 创建虚拟机、检查短消息服务器连通性 vs 检查邮箱服务器
连通性），用户按功能名查询时命中面变化。
- 方案 A：维持现修复（仅 text 相同即并组），接受可发现性变化；重放未观察到
  eval gold 回退，但机制风险留存。
- 方案 B（倾向）：去重键 sha1(text + metadata.title + metadata.operation)，
  仅并组「内容+标题」都相同的行；跨功能同文组保留，去重量略降。
- sections 语料组内差异经 [0512] 逐字段核验仅 section_path，方案 B 对其无影响
  （title 一致时行为不变）。

## D-20261112-0659-1（决议）chunks 内容级去重键升级 — 方案 B 已实施 [0745]
自动化轮按 [0659] 记录倾向执行：去重键 sha1(content|text + title + operation)，
title/operation 取值与 _normalize 同口径（顶层缺失回落 metadata）。sections
组内差异仅 section_path（[0512] 核验）→ 行为不变（实测 1353/1473 逐数一致）。
[0659] 两处未补偿移除（r003 CA管理面、r014 ER证书）真实链重放恢复；duli 0
查询变化。去重量 anquan -399 / duli -719 行（较方案 A 少去重 22/18 行，已接受）。
待用户追认；回退方案 A 成本 1 行。

## D-20261112-2003-1（自决，待追认）
- a-03/a-04 gold 事实级口径：relevant=双手册中逐字承载全部 gold excerpts
  的全部变体 ID（限行所属手册；a-03 7 个 / a-04 8 个），fractional recall。
- 依据：设计 §5 身份=来源+定位、事实=内容承载；读取侧 content 去重
  保留孪生变体之首，单文件 gold 与服务语义错位；消歧前共享 ID 记分虚高。
- 回退成本：还原 eval_set_v2.jsonl 两行 relevant。
- 证据：optimization-runs/20261112-2003-rebuild-goldsync/REPORT.md。
