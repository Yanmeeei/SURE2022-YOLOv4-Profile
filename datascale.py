import torch


class DpdtRec:
    def __init__(self, src_name, src):
        self.src_name = src_name
        self.src = src
        self.dest_list = []

    def update(self, dest):
        self.dest_list.append(dest)


class DataScale(object):
    def __init__(self):
        self.scaleRec = {}
        self.dependencyRec = {}

    def weight(self, tensor_src, data):
        if tensor_src not in self.scaleRec:
            self.scaleRec[tensor_src] = data.size()

    def dependency_check(self, tensor_name, src, dest):
        src_name = src + "_" + tensor_name
        if src_name not in self.dependencyRec:
            self.dependencyRec[src_name] = DpdtRec(src_name, src)
        self.dependencyRec[src_name].update(dest)

    def report(self):
        print("Data size of each tensor: \n")
        for key, value in self.scaleRec.items():
            print(f"{key}       ::      {list(value)}")

        print("Data Dependency of each layer \n")
        for key, value in self.dependencyRec.items():
            for dest in value.dest_list:
                print(f"{key}       ::      {value.src}       ::      {dest}")
