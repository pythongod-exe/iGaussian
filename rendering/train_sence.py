import os
import random
import json
from gaussian.utils.system_utils import searchForMaxIteration
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.arguments import ModelParams
from gaussian.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import pdb

class Scene:

    gaussians : GaussianModel

    def __init__(self, gaussians : GaussianModel):
        self.model_path = '/data1/whao_model/3dgs_model/lego'
        self.gaussians = gaussians
        self.loaded_iter = 30000

        self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))