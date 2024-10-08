import os
import io
import torch
import json
import configparser
import pickle as pk
import os

class FileOpen(object):
    def __init__(self, filename, **kwargs):
        self.handle = FileHelper.load_data(filename, **kwargs)

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_value, exc_trackback):
        del self.handle


class FileHelper(object):

    _File_helper = None
    open = FileOpen

    def __init__(self):
        FileHelper._File_helper = self
    
    @staticmethod
    def load_data(path, mode='r'):
        assert os.path.exists(path), f'No such file: {path}'
        return open(path, mode)

    @staticmethod
    def load_pk(path, mode='rb'):
        assert os.path.exists(path), f'No such file: {path}'
        return pk.load(open(path, mode))

    @staticmethod
    def load_json(path, mode='r'):
        assert os.path.exists(path), f'No such file: {path}'
        return json.load(open(path, mode))

    @staticmethod
    def download_json(path, local_path, mode='r'):
        assert os.path.exists(path), f'No such file: {path}'
        js = json.load(open(path, mode))
        with open(local_path, 'w') as f:
            json.dump(js, f)

    @staticmethod
    def download_file(path, local_path, mode='r'):
        assert os.path.exists(path), f'No such file: {path}'
        with open(path, mode) as f:
            data = f.read()
        with open(local_path, 'wb') as f:
            f.write(data)   

    @staticmethod
    def list(path, extension="json"):
        assert os.path.exists(path), f'No such file dir: {path}'
        filenames = []
        for name in os.listdir(path):
            if name.endswith(extension) or extension == "all":
                filenames.append(name)
        return filenames

    @staticmethod
    def load_pretrain(path, map_location=None):
        assert os.path.exists(path), f'No such file: {path}'
        return torch.load(path, map_location=map_location)

    @staticmethod
    def load(path, **kwargs):
        if '.ini' in path:
            path = path[:-4]
        if not os.path.exists(path) and os.path.exists(path + '.ini'):
            # get realpath
            conf = configparser.ConfigParser()
            conf.read(path + '.ini')
            path = conf['Link']['ceph']
        return FileHelper.load_pretrain(path, **kwargs)

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model, path)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def save(model, path):
        return FileHelper.save_checkpoint(model, path)

__file_helper = FileHelper()