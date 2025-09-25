# create a class DataProfiler
# each data is identified by a unique string id
# each data has a reward trace which is a list of tuples (epoch, reward)
# the class should have a method to add reward tuple. If the data is not in the list, it should be added
# the class should have a method to get the reward trace for a given data id
import torch
import numpy as np

class Rollout_Purning:
    def __init__(self):
        self.data = {}

    def add_reward(self, epoch: int, data_id: str, reward: float):
        if data_id not in self.data:
            self.data[data_id] = []
        self.data[data_id].append((epoch, reward))

    # the input is a list of datasource (str) and a list of index (int);
    # return a list of unique data_id which is combined from the datasource and index
    def get_data_id_list(self, data_sources: list, indices: list):
        data_ids = []
        for data_source, index in zip(data_sources, indices):
            data_ids.append(f"{data_source}_{index}")
        return data_ids

    def purning_by_entropy(self, current_epoch, batch, idx2acc, idx2pos_entropys, idx2neg_entropys, alpha, init_sampling_ratio, purning_decay_weight, history_k, neg_pos_ratio, total_gpus):
        
        if current_epoch == 0:
            return batch
        
        else:
            data_index = list(set(batch.non_tensor_batch['index'].astype(int).tolist()))
            selected_index = []
            eps = 1e-8

            threshold = init_sampling_ratio - purning_decay_weight * current_epoch
            threshold = max(0.40, min(1.0, threshold))
            print('Current Threshold:', threshold)

            batch_acc_list = {data_id: idx2acc[idx] for data_id, idx in enumerate(data_index)}
            batch_idx2pos_entropys = {data_id: idx2pos_entropys[idx] for data_id, idx in enumerate(data_index)}
            batch_idx2neg_entropys = {data_id: idx2neg_entropys[idx] for data_id, idx in enumerate(data_index)}

            for idx in batch_acc_list:
                acc_list = batch_acc_list[idx]
                if len(acc_list) == 0: 
                    selected_index.append(idx)

            exploratory_score_list = []

            for idx in batch_acc_list:
                
                if idx in selected_index:
                    continue
                
                acc_list = batch_acc_list[idx]
                pos_entropys = batch_idx2pos_entropys[idx]
                neg_entropys = batch_idx2neg_entropys[idx]
                assert len(pos_entropys) == len(neg_entropys) == len(acc_list)

                history_window = min(history_k, len(pos_entropys))
                pos_entropys_windows = pos_entropys[-history_window:]
                neg_entropys_windows = neg_entropys[-history_window:]
                history_acc_windows = acc_list[-history_window:]

                sample_exploratory_score = []

                for history_epoch_num, (epoch_pos_entropys, epoch_neg_entropys) in enumerate(zip(pos_entropys_windows, neg_entropys_windows)):
                    epoch_pos_entropys_num = len(epoch_pos_entropys)
                    epoch_neg_entropys_num = len(epoch_neg_entropys)
                    epoch_exp_score = 0

                    sample_acc_list = [1] * epoch_pos_entropys_num + [0] * epoch_neg_entropys_num
                    acc_mean = np.mean(sample_acc_list)
                    acc_std = np.std(sample_acc_list)
                    pos_adv = (1.0 - acc_mean) / (acc_std + eps)
                    neg_adv = (0.0 - acc_mean) / (acc_std + eps)
                    avg_epoch_correct_entropys = sum(epoch_pos_entropys) / len(epoch_pos_entropys) if len(epoch_pos_entropys) > 0 else 0

                    filtered_neg_entropys = []
                    for epoch_neg_entropy in epoch_neg_entropys:
                        if epoch_neg_entropy <= neg_pos_ratio * avg_epoch_correct_entropys:
                            filtered_neg_entropys.append(epoch_neg_entropy)

                    for epoch_pos_entropy in epoch_pos_entropys:
                        epoch_exp_score += abs(pos_adv * epoch_pos_entropy)

                    for epoch_neg_entropy in filtered_neg_entropys:
                        epoch_exp_score += abs(neg_adv * epoch_neg_entropy)
                    
                    sample_exploratory_score.append(epoch_exp_score)

                avg_history_sample_exp_score = sum(sample_exploratory_score) / len(sample_exploratory_score)
                exploratory_score_list.append((idx, avg_history_sample_exp_score, len(pos_entropys)))   # (index, exploration_score, exploration_time)

            exploratory_score_list.sort(key=lambda x: (-x[1], x[2]))  
            top_n = int(len(exploratory_score_list) * threshold)
            selected_index.extend([idx for idx, _, _ in exploratory_score_list[:top_n]])

            print("Len(exploration score)", len(exploratory_score_list))
            print("Index Info:", [(idx, batch_acc_list[idx], exploration_score, exploration_time) for idx, exploration_score, exploration_time in exploratory_score_list[:top_n]])

            print("Before Fill:", len(selected_index))

            remainder = len(selected_index) % total_gpus
            if remainder == 0:
                remainder_num = 0  
            else:
                remainder_num = total_gpus - remainder  

            if remainder_num > 0:
                remaining_candidates = exploratory_score_list[top_n:]
                remaining_candidates.sort(key=lambda x: (x[2], -x[1]))
                print("remaining_candidates", remaining_candidates)
                selected_index.extend([idx for idx, _, _ in remaining_candidates[:remainder_num]])

            print("After Fill:", len(selected_index))

            batch = batch.select_via_index(selected_index)
            return batch



                    



    # get a list of id, and a list of reward float; add them to the profiler
    def add_reward_list(self, epoch, data_ids: list, rewards: list):
        for data_id, reward in zip(data_ids, rewards):
            self.add_reward(epoch, data_id, reward)

    def get_reward_trace(self, data_id: str):
        return self.data.get(data_id, [])

    def get_all_data_ids(self):
        return list(self.data.keys())

    def save(self, path: str):
        torch.save(self.data, path)

    def load(self, path: str):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

