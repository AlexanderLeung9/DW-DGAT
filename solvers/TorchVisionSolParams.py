import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as to
import torch.nn.functional as f
import torchvision.models as tvm
import logging as l
import typing as t
import tqdm
import arguments as ag
import datasets as ds
import networks as nw
import solvers as sol


class TorchVisionSolParams(sol.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: t.Optional[ag.TorchVisionArgs]):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_data(self):
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDataset(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

        if self.args.feature_num > 0:
            all_data = torch.zeros(self.sample_num, self.args.feature_num, ag.BDArguments.ROI_NUM, ag.BDArguments.ROI_NUM, dtype=torch.float).to(ag.Arguments.device)
        else:
            # BrainNetTransformerSolParams
            all_data = torch.zeros(self.sample_num, ag.BDArguments.ROI_NUM, ag.BDArguments.ROI_NUM, dtype=torch.float).to(ag.Arguments.device)

        print(f"Loading all data of {self.args.txt_indices}...")
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            datum0 = datum[0]
            assert isinstance(datum0, torch.Tensor)
            networks = datum0[0]
            if self.args.feature_num > 0:
                all_data[i] = networks
            else:
                all_data[i] = networks[0]

        self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)

    def _prepare_network(self):
        # e.g. VGGNet-19-BN
        if self.args.net_name.startswith("VGGNet-"):
            bn = self.args.net_name.endswith("BN")
            length = len("VGGNet-")
            layer_num = int(self.args.net_name[length:length+2])
            create_cls_params = VggParams(layer_num, bn, self.args)
            self.create_cls_factory = VGGNetFactory(create_cls_params)

        elif self.args.net_name.startswith("ResNet-"):
            layer_num = int(self.args.net_name[len("ResNet-"):])
            create_cls_params = TVMParams(layer_num, self.args)
            self.create_cls_factory = ResNetFactory(create_cls_params)

        elif self.args.net_name.startswith("DenseNet-"):
            layer_num = int(self.args.net_name[len("DenseNet-"):])
            create_cls_params = TVMParams(layer_num, self.args)
            self.create_cls_factory = DenseNetFactory(create_cls_params)

        else:
            raise NotImplementedError(f"net_name={self.args.net_name}")


class TVMParams(nw.NetworkParams):
    def __init__(self, layer_num: int, args: ag.TorchVisionArgs):
        super().__init__(args)

        self.layer_num = layer_num
        self.args = args


class VggParams(TVMParams):
    def __init__(self, layer_num: int, bn: bool, args: ag.TorchVisionArgs):
        super().__init__(layer_num, args)
        self.bn = bn


class VGGNetFactory(nw.NetworkFactory):
    def __init__(self, create_params: VggParams):
        super().__init__(create_params, "")
        self.create_params = create_params

    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        pretrained_weight = None
        if self.create_params.layer_num == 11:
            if self.create_params.bn:
                network = tvm.vgg11_bn(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg11_bn-6002323d.pth")
            else:
                network = tvm.vgg11(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg11-bbd30ac9.pth")
        elif self.create_params.layer_num == 13:
            if self.create_params.bn:
                network = tvm.vgg13_bn(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg13_bn-abd245e5.pth")
            else:
                network = tvm.vgg13(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg13-c768596a.pth")
        elif self.create_params.layer_num == 16:
            if self.create_params.bn:
                network = tvm.vgg16_bn(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg16_bn-6c64b313.pth")
            else:
                network = tvm.vgg16(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg16-397923af.pth")
        elif self.create_params.layer_num == 19:
            if self.create_params.bn:
                network = tvm.vgg19_bn(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg19_bn-c79401a0.pth")
            else:
                network = tvm.vgg19(num_classes=self.create_params.args.class_num)
                if self.create_params.args.pretrained_weights_folder is not None:
                    pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/vgg19-dcbb9e9d.pth")
        else:
            raise NotImplementedError(self.create_params.layer_num)
    
        if self.create_params.args.pretrained_weights_folder is not None:
            network.load_state_dict(pretrained_weight)
    
        network.loss = f.cross_entropy
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []


class ResNetFactory(nw.NetworkFactory):
    def __init__(self, create_params: TVMParams):
        super().__init__(create_params, "")
        self.create_params = create_params

    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        pretrained_weight = None
        if self.create_params.layer_num == 18:
            network = tvm.resnet18(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/resnet18-5c106cde.pth")
        elif self.create_params.layer_num == 34:
            network = tvm.resnet34(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/resnet34-333f7ec4.pth")
        elif self.create_params.layer_num == 50:
            network = tvm.resnet50(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/resnet50-19c8e357.pth")
        elif self.create_params.layer_num == 101:
            network = tvm.resnet101(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/resnet101-5d3b4d8f.pth")
        elif self.create_params.layer_num == 152:
            network = tvm.resnet152(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/resnet152-b121ed2d.pth")
        else:
            raise NotImplementedError(self.create_params.layer_num)
    
        if self.create_params.args.pretrained_weights_folder is not None:
            network.load_state_dict(pretrained_weight)
    
        network.loss = f.cross_entropy
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []


class DenseNetFactory(nw.NetworkFactory):
    def __init__(self, create_params: TVMParams):
        super().__init__(create_params, "")
        self.create_params = create_params   
    
    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        pretrained_weight = None
    
        if self.create_params.layer_num == 121:
            network = tvm.densenet121(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/densenet121-a639ec97.pth")
        elif self.create_params.layer_num == 161:
            network = tvm.densenet161(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/densenet161-8d451a50.pth")
        elif self.create_params.layer_num == 169:
            network = tvm.densenet169(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/densenet169-b2777c0a.pth")
        elif self.create_params.layer_num == 201:
            network = tvm.densenet201(num_classes=self.create_params.args.class_num)
            if self.create_params.args.pretrained_weights_folder is not None:
                pretrained_weight = torch.load(f"{self.create_params.args.pretrained_weights_folder}/densenet201-c1103571.pth")
        else:
            raise NotImplementedError(self.create_params.layer_num)
    
        if self.create_params.args.pretrained_weights_folder is not None:
            network.load_state_dict(pretrained_weight)
    
        network.loss = f.cross_entropy
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []
