import os
import enums as es
import arguments as ag
import programs as pg


class Paper2(object):
    @staticmethod
    def ablation_experiments():
        pg.JobManager.update_state("Preprocess data")
        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Phenotype
        args.cls_gen_train_ratio = 0
        args.stop_value = 0
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        pg.JobManager.update_state("baseline")
        args = ag.DW_DGATArgs(list(range(3)))
        args.single_graph_module = ag.ESingleGraphModule.none
        args.adj_graph_type = es.EAdjGraphType.NoGraph
        args.cls_gen_train_ratio = 0
        args.stop_value = 150
        args.log_dir_comment = f" {pg.JobManager.current_state}"
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        pg.JobManager.update_state("DF")
        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.none
        args.adj_graph_type = es.EAdjGraphType.NoGraph
        args.cls_gen_train_ratio = 0
        args.stop_value = 150
        args.log_dir_comment = f" {pg.JobManager.current_state}"
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        pg.JobManager.update_state("DF+SGA")
        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.NoGraph
        args.cls_gen_train_ratio = 0
        args.log_dir_comment = f" {pg.JobManager.current_state}"
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        # pg.JobManager.update_state("GGA")
        # args = ag.DW_DGATArgs(list(range(26)))
        # args.single_graph_module = ag.ESingleGraphModule.none
        # args.adj_graph_type = es.EAdjGraphType.Phenotype
        # args.cls_gen_train_ratio = 0
        # args.log_dir_comment = f" {pg.JobManager.current_state}"
        # tExperiment = pg.BDExperiment(args)
        # tExperiment.initialize()
        # tExperiment.run()

        pg.JobManager.update_state("DF+SGA+GGA")
        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Phenotype
        args.cls_gen_train_ratio = 0
        args.log_dir_comment = f" {pg.JobManager.current_state}"
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        pg.JobManager.update_state("complete")
        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Phenotype
        args.cls_gen_train_ratio = 1
        args.log_dir_comment = f" {pg.JobManager.current_state}"
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def vision_networks():
        args = ag.MLPArgs(list(range(3)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.TorchVisionArgs("VGGNet-19-BN")
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.TorchVisionArgs("ResNet-34")
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.TorchVisionArgs("DenseNet-121")
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.ViTArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        # args = ag.SwinTransformerArgs()
        # tExperiment = pg.BDExperiment(args)
        # tExperiment.initialize()
        # pg.JobManager.update_state(args.net_name)
        # tExperiment.run()

    @staticmethod
    def ChAdaViT_experiment():
        args = ag.ChAdaViTArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        assert args.class_num == 2, "80GB GPU is not enough for more classes."
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

    @staticmethod
    def GNN_networks():
        args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        # args = ag.GATArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        # pg.JobManager.update_state(args.net_name)
        # tExperiment = pg.BDExperiment(args)
        # tExperiment.initialize()
        # tExperiment.run()

        args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        # args = ag.GraphormerArgs()
        # pg.JobManager.update_state(args.net_name)
        # tExperiment = pg.BDExperiment(args)
        # tExperiment.initialize()
        # tExperiment.run()

        # args = ag.NetworkGATv2Args()
        # pg.JobManager.update_state(args.net_name)
        # tExperiment = pg.BDExperiment(args)
        # tExperiment.initialize()
        # tExperiment.run()

        args = ag.MA_GCNNArgs(list(range(3)))
        stop_value = args.stop_value
        args.stop_value = 0
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args.stop_value = stop_value
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GCN_MHSAArgs(list(range(26)), es.ENetworksMergeMode.L1Norm)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def BD_networks():
        args = ag.BrainNetCNNArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.BrainNetTransformerArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.BrainGNNArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.LG_GNNArgs()
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

    @staticmethod
    def comparative_experiments():
        Paper2.vision_networks()
        Paper2.GNN_networks()
        Paper2.BD_networks()

    @staticmethod
    def euclidean_experiments():
        args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Euclidean)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Euclidean)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Euclidean)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Euclidean
        args.cls_gen_train_ratio = 0
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Euclidean)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def unweighted_experiments():
        args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Unweighted
        args.cls_gen_train_ratio = 0
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def phenotype_experiments():
        args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.DW_DGATArgs(list(range(26)))
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Phenotype
        args.cls_gen_train_ratio = 0
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def no_data_fusion_experiments():
        args = ag.MLPArgs(list(range(3)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GATv2Args(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(3)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.MA_GCNNArgs(list(range(3)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(3)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def data_fusion_experiments():
        args = ag.MLPArgs(list(range(26)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GCNArgs(list(range(26)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.GATv2Args(list(range(26)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.ChebNetIIArgs(list(range(26)), es.EAdjGraphType.Phenotype)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.MA_GCNNArgs(list(range(26)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.RA_GCNArgs(list(range(26)), es.EAdjGraphType.Unweighted)
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

    @staticmethod
    def relationship_experiments():
        # global GCN + MLP
        # metric_len=3 will lead to crash.
        # Tried to allocate 4.30 GiB (GPU 0; 23.69 GiB total capacity; 17.60 GiB already allocated; 2.53 GiB free; 17.62 GiB reserved in total by PyTorch
        ag.Arguments.history_splitting_epoch = 101
        args = ag.DW_DGATArgs(list(range(1)))
        args.stop_value = ag.Arguments.history_splitting_epoch - 1
        args.single_graph_module = ag.ESingleGraphModule.none
        args.adj_graph_type = es.EAdjGraphType.Relationship
        args.cls_gen_train_ratio = 0
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state("1")
        tExperiment.run()

        args = ag.DW_DGATArgs(list(range(26)))
        args.stop_value = ag.Arguments.history_splitting_epoch - 1
        args.single_graph_module = ag.ESingleGraphModule.vit_small
        args.adj_graph_type = es.EAdjGraphType.Relationship
        args.cls_gen_train_ratio = 0
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state("26")
        tExperiment.run()

    @staticmethod
    def temporary_experiments():
        args = ag.GCN_MHSAArgs(list(range(26)), es.ENetworksMergeMode.L1Norm)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.MA_GCNNArgs(list(range(3)))
        pg.JobManager.update_state(args.net_name)
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        tExperiment.run()

        args = ag.TorchVisionArgs("VGGNet-19-BN")
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()

        args = ag.TorchVisionArgs("ResNet-34")
        tExperiment = pg.BDExperiment(args)
        tExperiment.initialize()
        pg.JobManager.update_state(args.net_name)
        tExperiment.run()


if __name__ == "__main__":
    # ag.Arguments.log_root_dir = os.path.join(".", "logs")
    ag.Arguments.business = es.EBusiness.PD
    ag.BDArguments.initialize_globally(0)
    pg.JobManager.begin_job(f"ablation_experiments")

    Paper2.ablation_experiments()
    # Paper2.comparative_experiments()
    # Paper2.euclidean_experiments()
    # Paper2.unweighted_experiments()
    # Paper2.phenotype_experiments()
    # Paper2.no_data_fusion_experiments()
    # Paper2.data_fusion_experiments()
    # pg.BDExperiment.iterate_businesses(Paper2.relationship_experiments)
    # Paper2.temporary_experiments()

    pg.JobManager.finish_job(0)
    # pg.JobManager.shutdown_cloud_server()
