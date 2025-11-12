import os
import enums as es
import arguments as ag
import programs as pg


if __name__ == "__main__":
    ag.Arguments.log_root_dir = os.path.join(".", "logs")
    ag.Arguments.business = es.EBusiness.PD
    ag.BDArguments.initialize_globally(0)

    args = ag.DW_DGATArgs(list(range(26)))
    args.single_graph_module = ag.ESingleGraphModule.vit_small
    args.adj_graph_type = es.EAdjGraphType.Phenotype
    args.cls_gen_train_ratio = 1
    # args.set_validation_mode(1)
    # args.net_state_file = r"E:\BDdata\PD_middle_data\MRI_DTI\preprocessed_data\DW_DGAT\DW_DGAT-PD-0,1,2-01.pth"
    # args.learning_mode = True
    # args.log_dir_comment = ""
    # args = ag.TorchVisionArgs("ResNet-34")
    # args = ag.LG_GNNArgs()
    # args = ag.MA_GCNNArgs(list(range(3)))
    # args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
    # args = ag.MV_GCNArgs(es.EAdjGraphType.Euclidean)
    # args = ag.ChAdaViTArgs()
    # args = ag.SwinTransformerArgs()
    # args = ag.GraphormerArgs()
    # args = ag.MLPArgs(list(range(3)))
    # args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Phenotype)
    # args = ag.GATArgs(list(range(3)), es.EAdjGraphType.Phenotype)
    # args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Phenotype)
    # args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Phenotype)
    # args = ag.BrainNetCNNArgs()
    # args = ag.BrainNetTransformerArgs()
    # args = ag.GCN_MHSAArgs(list(range(26)), es.ENetworksMergeMode.L1Norm)
    # args = ag.BrainGNNArgs()
    # args = ag.ViTArgs()
    # args.stop_value = 0
    # args.lowest_accuracy = 0.1
    tExperiment = pg.BDExperiment(args)
    tExperiment.initialize()
    tExperiment.run()
