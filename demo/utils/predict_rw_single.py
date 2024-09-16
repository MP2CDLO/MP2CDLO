import os
import sys
import torch
import yaml
import glob
import shutil
import numpy as np

sys.path.append('../../')
from models.build import MODELS


class ConfigObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return ConfigObject(config)


def custom_load_model(base_model, ckpt_path, logger=None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print(f'Loading weights from {ckpt_path}...')

    state_dict = torch.load(ckpt_path, map_location='cpu')
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", "", 1): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", "", 1): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict=True)
    if 'prior_points' in base_ckpt.keys():
        base_model.prior_points.data = base_ckpt['prior_points']

    epoch = state_dict.get('epoch', -1)
    metrics = state_dict.get('metrics', 'No Metrics')
    if not isinstance(metrics, dict):
        metrics = metrics.state_dict() if hasattr(metrics, 'state_dict') else str(metrics)
    print(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})')
    return


class ModelInference():
    def __init__(self, config_path, weights_path, dir_path, output_folder):
        self.config_path = config_path
        self.weights_path = weights_path
        self.dir_path = dir_path
        self.output_folder = output_folder

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        config = load_config(self.config_path)
        print(config.model.NAME)
        # Load model
        self.model = MODELS.get(config.model.NAME)(config.model)
        custom_load_model(self.model, weights_path)
        self.model.to(device)

    def save_results(self, result, input_file_name):
        # Generate output file name
        base_name = os.path.basename(input_file_name)
        output_file_name = base_name.replace("pointcloud", "pred")

        result_path = os.path.join(self.output_folder, output_file_name)
        # print(result.shape)
        np.save(result_path, result.cpu().numpy()[0])
        # print(f'Saved result to {result_path}')

    def load_data(self, group_num):
        pattern = os.path.join(self.dir_path, f'*_{group_num}.npy')
        files = glob.glob(pattern)
        shutil.copy(files[0], self.output_folder)
        if len(files) != 1:
            raise ValueError(f"Found {len(files)} files, expected 1. Please check the naming convention and try again.")
        file_path = files[0]

        data = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
        return data, file_path

    def inference(self, group_num):
        data, file_name = self.load_data(group_num)

        self.model.eval()

        with torch.no_grad():
            partial = data.cuda()
            ret = self.model(partial)

        self.save_results(ret, file_name)


if __name__ == "__main__":
    config_path = '../experiments/config.yaml'
    weights_path = '../experiments/ckpt-last.pth'
    dir_path = '../experiments/output_data_robot/partial'
    output_folder = '../experiments/results/'
    group_num = '0100'
    model_inference = ModelInference(config_path, weights_path, dir_path, output_folder)
    model_inference.inference(group_num)
