"""改进模块：频域融合 / 边缘支路 / 边界精修 / 属性头 / 属性图卷积。
Improvement modules: frequency fusion / edge branch / boundary smooth / per-attr heads / attribute GCN.

所有模块都是即插即用，不改现有训练/推理管线，通过参数开关启用。
All modules are plug-and-play; enabled via flags without modifying existing pipelines.

用法 / Usage:
    from .improvements.ewt_fusion import EWT_Fusion
    from .improvements.edge_branch import EdgeBranch, edge_supervised_loss
    from .improvements.boundary_smooth import boundary_smooth
    from .improvements.task2_per_head import PerAttrHead, AttrGCN, build_attr_graph_from_gt
"""
