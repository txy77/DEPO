# Towards High Data Efficiency in Reinforcement Learning with Verifiable Reward (ICLR 2026)

## 😀 Contributions

+ 1️⃣ We are the first to integrate both offline and online data selection strategies to enhance data efficiency in RLVR training.
+ ⏰ In the offline phase, we employ a multi-dimensional data curation strategy based on diversity, influence, and difficulty. Then, during online training, we dynamically filter samples by their explorability and replay under-explored samples to further improve training efficiency.
+ 🧪 Extensive experiments across five reasoning datasets and three LLMs demonstrate the effectiveness and efficiency of our proposed method under both offline and online data selection scenarios.

## 🌟 Highlights

<img width="533" height="205" alt="image" src="https://github.com/user-attachments/assets/2b16d557-adeb-4c94-a4a7-11282561813a" />

Overview of our approach DEPO. (a) Our approach improves the data efficiency in RLVR training via an end-to-end offline and online data selection strategy. (b) In the offline phase, we first construct a sample graph based on the representations, then apply PageRank-weighted Determinantal
Point Process to select a diverse and influential subset, and finally sample from this subset with difficulty following a normal distribution. (c) In the online phase, we evaluate the explorability of each sample based on its historical training dynamics and retain high-explorability ones for rollout, and actively replay under-explored samples to ensure sufficient training of all samples.

<img width="896" height="626" alt="image" src="https://github.com/user-attachments/assets/983b06f5-f468-4011-a73c-bad2c6138f06" />

Performance comparison of various data selection methods. “Offline” and “Online” refer to the offline and online data selection methods, respectively. “Ratio“, “Time”, and “RN” denote the ratio of selected data, total training time, and total rollout numbers, respectively. We highlight the best performance across different data selection methods.

<img width="896" height="645" alt="image" src="https://github.com/user-attachments/assets/5d6f9ecd-923f-419b-8953-651b57a30768" />

## 🚀 Quick Start


