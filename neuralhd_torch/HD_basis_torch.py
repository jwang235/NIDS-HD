from .Config import config, Generator
import time
import torch 

def generate_vector(vector_length, vector_type, param):
    if vector_type == "Gaussian":
        mu = param["mu"]
        sigma = param["sigma"]
        return torch.empty(vector_length).normal_(mean=mu,std=sigma)

class HD_basis:
    param_req = {Generator.Vanilla: []}
    param_config = ["nFeatures", "nClasses", "D", "sparse", "s", "vector", "mu", "sigma", "binarize"]

    def __init__(self, gen_type, param):
        for req in self.param_req[gen_type]:
            if req not in param:
                raise Exception("required parameters not received in HD_Basis, abort.\n")
        self.param = param
        self.param["id"] = str(int(time.time()) % 10000)

        for term in self.param_config:
            if term not in self.param:
                self.param[term] = config[term]
        self.param["gen_type"] = gen_type
        if gen_type == Generator.Vanilla:
            self.vanilla(param)
            
    def vanilla(self, param):
        self.basis = torch.empty(param["D"], param["nFeatures"])
        for i in range(self.param["D"]):
            self.basis[: i] = generate_vector(self.param["nFeatures"], self.param["vector"], self.param)

    def updateBasis(self, toChange = None):
        for i in toChange:
            self.basis[i] = generate_vector(self.param["nFeatures"], self.param["vector"], self.param)

    def getBasis(self):
        return self.basis

    def getParam(self):
        return self.param


