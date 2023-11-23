import os
import json
import numpy as np
import pickle
import torch

def remove_mid_file(save_dir, num_processes):
    filename = [f'prompt{i}.json' for i in range(num_processes)] + [f'sample{i}.pkl' for i in range(num_processes)] + [f'{i}.txt' for i in range(num_processes)]
    print('mid file: ', filename)
    for file in filename:
        try:
            os.remove(os.path.join(save_dir, file))
            print(f"{file} delete successfully!")
        except OSError as e:
            print(f"Error: {e}")

def check_data(save_dir, num_processes, sample0_shape):
    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        sample: dict = pickle.load(f)
    for key, value in sample.items():
        assert sample0_shape[key][1:]==value.shape[1:] and num_processes*sample0_shape[key][0]==value.shape[0], f'{sample0_shape[key]}{value.shape}'
    print('---------start remove---------')
    remove_mid_file(save_dir, num_processes)


def post_processing(save_dir, num_processes):
    print(f'data save dir: {save_dir}')
    prompts = []
    for i in range(num_processes):
        with open(os.path.join(save_dir, f'prompt{i}.json'), 'r') as f:
            prompts_ = json.load(f)
            prompts += prompts_
    print('---------write prompt---------')
    with open(os.path.join(save_dir, 'prompt.json'), 'w') as f:
        json.dump(prompts, f)
    samples = {}
    sample0_shape = {}
    for i in range(num_processes):
        with open(os.path.join(save_dir, f'sample{i}.pkl'), 'rb') as f:
            sample_: dict = pickle.load(f)
            if i==0:
                for key, value in sample_.items():
                    sample0_shape[key] = value.shape
                samples = sample_
            else:
                for key, value in sample_.items():
                    assert sample0_shape[key] == value.shape, f'{key}.shape in sample{i}.pkl({sample0_shape[key]}) is different with {key}.shape in sample0.pkl({value.shape}). '
                samples = {k: torch.cat([s[k] for s in [samples, sample_]]) for k in samples.keys()}
    print('---------write sample---------')
    with open(os.path.join(save_dir, 'sample.pkl'), 'wb') as f:
        pickle.dump(samples, f)
    print('---------start check---------')
    check_data(save_dir, num_processes, sample0_shape)

def load_data_from_json(path):
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    all_data = []
    for file in json_files:
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            all_data.append(data)
    min_length = min(map(len, all_data))
    data_clip = np.array([l[:min_length] for l in all_data])
    all_data_np = np.array(data_clip)
    mean_values = np.mean(all_data_np, axis=0)
    return mean_values

def load_sample(path):
    with open(os.path.join(path,"sample.pkl"),'rb') as f:
        sample = pickle.load(f)
    return sample
