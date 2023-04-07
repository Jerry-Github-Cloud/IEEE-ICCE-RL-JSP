# Deep Reinforcement Learning Based on Graph Neural Networks for Job-shop Scheduling 
在這篇論文中，我們提出了一個基於圖神經網絡（GNN）的深度強化式學習 (DRL) 方法，藉由選擇 priority dispatching rule 求解 Job shop scheduling problem(JSP)。
我們的方法在小型實例上訓練後，在大型測資上也表現良好。我們的實驗表明，我們的方法優於PDR方法和大多數其他DRL方法，特別是對於大型測資。


## Experiment
比較我們的方法(Ours)和priority dispatching rule和其他DRL方法的結果

Priority dispatching rule
* MOR
* FIFO
* SPT
* MWKR


其他DRL方法
* L2D 
    * C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, and C. Xu, ”Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning,” in NeurIPS, 202
* ScheduleNet 
    * J. Park, S. Bakhtiyar, and J. Park, ”ScheduleNet: Learn to solve multi- agent scheduling problems with reinforcement learning,” in CoRR,

我們使用Taillard資料集(E. D. Taillard, ”Benchmarks for basic scheduling problems,” in Eur. J. Oper. Res., 1993)作為比較基準，這些實例分為兩組
* 小型測資: 15x15, 20x15, 20x20, 30x15, 30x20
* 大型測資: 50x15, 50x20, 100x20

實驗結果顯示，我們的方法在大型測資上的 gap(與最佳解或是已知最佳)的差距比起其他方法更好，在小型測資上小輸 ScheduleNet

|        | Ours  | ScheduleNet | L2D   | MOR   | FIFO  | SPT   | MWKR  |
| ------ | ----- |:----------- |:----- |:----- |:----- |:----- |:----- |
| Gap(S) | 20.0% | 17.7%       | 30.7% | 22.7% | 29.0% | 31.2% | 22.7% |
| Gap(L) | 10.5% | 11.3%       | 20.8% | 14.8% | 19.4% | 21.3% | 14.5% |
| Time   | 1.55s | N/A         | 5.10s | 1.18s | 1.12s | 1.01s | 1.13s |

## usage
### Run PDRs 
```
python3 run_rules.py
```

### Evaluate
```
python3 eval_dqn.py
```

### Training
```
python3 train_dqn.py
```

### Parallel evaluate
```
python3 para_eval_dqn_2.py
```