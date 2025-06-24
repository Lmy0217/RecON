import os

import configs

__all__ = ['env']


class Env(configs.BaseConfig):

    def __init__(self):
        super(Env, self).__init__({})
        cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../res/env'))
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        for file in os.listdir(cfg_dir):
            setattr(self, os.path.splitext(file)[0], configs.BaseConfig(os.path.join(cfg_dir, file)))
        if hasattr(self, 'paths') and hasattr(self.paths, 'root_folder'):
            self.paths.root_folder = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', self.paths.root_folder))
        else:
            raise ValueError('Lack of `res/env/paths.json` file or `root_folder` value')

    def getdir(self, path):
        return os.path.abspath(os.path.join(self.paths.root_folder, path))

    def chdir(self, path):
        os.chdir(self.getdir(path))


env = Env()
