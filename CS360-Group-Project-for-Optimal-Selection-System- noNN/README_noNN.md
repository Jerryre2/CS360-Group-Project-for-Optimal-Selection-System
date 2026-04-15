# README_noNN

这个目录是 `CS360-Group-Project-for-Optimal-Selection-System` 的无神经网络消融副本，用于和原版做公平对照实验。

## 和原版的区别

- 移除了 Local Search 中的在线神经网络推理与训练逻辑。
- 移除了特征提取器和神经网络样本缓冲、前向传播、反向传播相关实现。
- `ImprovedLocalSearch` 保留原有搜索流程，但所有 move 选择都改为纯启发式规则。
- Greedy、Redundancy Elimination、Simulated Annealing、ILP、结果存储、CLI、测试框架都保留。
- CLI 仍然接受 `--disable-neural-guidance` 参数以兼容旧脚本，但在这个副本中该参数不会改变行为。

## 这个副本适合做什么

- 做 NN vs noNN 的消融测试。
- 测量“纯启发式版本”的真实运行时间和解质量。
- 验证当前性能瓶颈是否主要来自在线 NN 引导。

## 建议跑法

- 若要和原版做最公平对比，固定 `samples`、`seed`、`restarts`。
- 建议优先比较三项指标：`Best family size`、总耗时 `Elapsed time`、每次 `Local search` 的耗时。
- 如果要记录结果，建议分别在两个目录下运行，避免混淆日志和输出文件。

## 推荐命令

原版仓库：

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --seed 42 --restarts 5
```

noNN 副本：

```bash
cd "/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System- noNN"
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --seed 42 --restarts 5
```

兼容旧脚本的 noNN 跑法：

```bash
cd "/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System- noNN"
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --seed 42 --restarts 5 \
  --disable-neural-guidance
```

## 回归验证

在这个副本里可以直接运行：

```bash
cd "/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System- noNN"
python -m unittest discover -s tests -v
```

## 当前已知结论

- 这个 noNN 副本保持了覆盖语义和可行性检查不变。
- 它适合回答一个很具体的问题：当前版本中的在线 NN，到底是在帮忙，还是在增加运行开销。
