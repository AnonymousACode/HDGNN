"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from IPython import embed
import logging
import time

# from ocpmodels.models.hdgnn.cg import generate_clebsch_gordan
import numpy as np
import math
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.models.hdgnn.sampling import CalcSpherePoints
from ocpmodels.models.hdgnn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from ocpmodels.models.hdgnn.fast_activation import Activation, Gate
from ocpmodels.models.hdgnn.layer_norm import EquivariantLayerNormV2, EquivariantLayerNormV2_channel
from ocpmodels.models.hdgnn.tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from ocpmodels.models.hdgnn.spherical_harmonics import SphericalHarmonicsHelper_4
# from ocpmodels.models.hdgnn.spherical_harmonics_4 import SphericalHarmonicsHelper_4

try:
    import e3nn
    from e3nn import o3, io
except ImportError:
    pass


seg_l = 0
seg = seg_l ** 2
seg_reduce = 0

@registry.register_model("hdgnn")
class HDGNN(BaseModel):
    """HDGNN

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_num_neighbors (int): Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_interactions (int): Number of layers in the GNN
        lmax (int):             Maximum degree of the spherical harmonics (1 to 10)
        mmax (int):             Maximum order of the spherical harmonics (0 or 1)
        num_resolutions (int):  Number of resolutions used to compute messages, further away atoms has lower resolution (1 or 2)
        sphere_channels (int):  Number of spherical channels
        sphere_channels_reduce (int): Number of spherical channels used during message passing (downsample or upsample)
        hidden_channels (int):  Number of hidden units in message passing
        num_taps (int):         Number of taps or rotations used during message passing (1 or otherwise set automatically based on mmax)

        use_grid (bool):        Use non-linear pointwise convolution during aggregation
        num_bands (int):        Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)

        num_sphere_samples (int): Number of samples used to approximate the integration of the sphere in the output blocks
        num_basis_functions (int): Number of basis functions used for distance and atomic number blocks
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float): Width of distance basis function
        distance_resolution (float): Distance between distance basis functions in Angstroms

        show_timing_info (bool): Show timing and memory info
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        use_pbc=False,
        regress_forces=False,
        otf_graph=False,
        max_num_neighbors=40,
        cutoff=8.0,
        max_num_elements=90,
        num_interactions=12,
        lmax=6,
        mmax=1,
        lmid=2,
        num_resolutions=2,
        sphere_channels=128,
        sphere_channels_reduce=128,
        sphere_channels_l0=256,
        hidden_channels=512,
        num_taps=-1,
        use_grid=True,
        num_bands=1,
        num_sphere_samples=128,
        num_basis_functions=128,
        distance_function="gaussian",
        basis_width_scalar=1.0,
        distance_resolution=0.02,
        show_timing_info=False,
        direct_forces=True,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error(
                "You need to install e3nn v0.2.6 to use the hdgnn model"
            )
            raise ImportError

        # assert e3nn.__version__ == "0.2.6"

        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.num_atoms = 0
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.sphere_channels_reduce = sphere_channels_reduce
        self.sphere_channels_l0 = sphere_channels_l0
        self.max_num_neighbors = self.max_neighbors = max_num_neighbors
        self.num_basis_functions = num_basis_functions
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax = lmax
        self.mmax = mmax
        self.lmid = lmid
        self.basis_width_scalar = basis_width_scalar
        self.sphere_basis = (self.lmax + 1) ** 2
        self.use_grid = use_grid
        self.distance_function = distance_function

        # variables used for display purposes
        self.counter = 0
        print('hdgnn_invl0_D-1')
        self.act = nn.SiLU()

        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]

        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        if num_resolutions == 1:
            self.num_resolutions = 1
            self.hidden_channels_list = torch.tensor([self.hidden_channels])
            self.lmax_list = torch.tensor(
                [self.lmax, -1]
            )  # always end with -1
            self.cutoff_list = torch.tensor([self.max_num_neighbors - 0.01])
        if num_resolutions == 2:
            self.num_resolutions = 2
            self.hidden_channels_list = torch.tensor(
                [self.hidden_channels, self.hidden_channels // 4]
            )
            self.lmax_list = torch.tensor([self.lmax, max(4, self.lmax - 2)])
            self.cutoff_list = torch.tensor(
                [12 - 0.01, self.max_num_neighbors - 0.01]
            )

        self.sphharm_list = []
        for i in range(self.num_resolutions):
            self.sphharm_list.append(
                SphericalHarmonicsHelper_4(
                    self.lmax_list[i],
                    self.mmax,
                    num_taps,
                    num_bands,
                )
            )

        self.sphharm_node = []
        # for i in range(self.num_resolutions):
        #     self.sphharm_node.append(
        #         SphericalHarmonicsHelper_4(
        #             self.lmax_list[i],
        #             self.mmax,
        #             num_taps,
        #             num_bands,
        #         )
        #     )

        self.lmax_reduce = 4
        # self.sphharm_node = SphericalHarmonicsHelper_4(
        #             self.lmax_reduce,
        #             self.mmax,
        #             num_taps,
        #             num_bands,
                # )

        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_l0
        )
        self.fc_l0 = nn.Linear(self.sphere_channels_l0, self.sphere_channels)
        # self.l0_bn = nn.BatchNorm1d(self.sphere_channels)

        self.edge_blocks = nn.ModuleList()
        for i in range(self.num_interactions):
            if i == self.num_interactions - 1:
                time_flag = 0
            else:
                time_flag = 1
            block = EdgeBlock(
                self.num_resolutions,
                self.sphere_channels_reduce,
                sphere_channels_l0,
                self.hidden_channels_list,
                self.cutoff_list,
                self.sphharm_list,
                self.sphere_channels,
                self.distance_expansion,
                self.max_num_elements,
                self.num_basis_functions,
                self.num_gaussians,
                self.use_grid,
                self.act,
                self.sphharm_node,
                time_flag,
            )
            self.edge_blocks.append(block)

        # Energy estimation
        self.energy_fc1 = nn.Linear(self.sphere_channels, self.sphere_channels)
        self.energy_fc2 = nn.Linear(
            self.sphere_channels, self.sphere_channels_reduce
        )
        self.energy_fc3 = nn.Linear(self.sphere_channels_reduce, 1)

        # self.energy_res1 = nn.Linear(self.sphere_channels * self.num_interactions, self.sphere_channels)
        # self.energy_res2 = nn.Linear(self.sphere_channels, 1)

        # Force estimation
        if self.regress_forces:
            self.force_fc1 = nn.Linear(
                self.sphere_channels, self.sphere_channels
            )
            self.force_fc2 = nn.Linear(
                self.sphere_channels, self.sphere_channels_reduce
            )
            self.force_fc3 = nn.Linear(self.sphere_channels_reduce, 1)

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.device = data.pos.device
        data.natoms = data.ptr[1:] - data.ptr[:-1]
        self.num_atoms = len(data.batch)
        self.batch_size = len(data.natoms)
        # torch.autograd.set_detect_anomaly(True)

        start_time = time.time()

        outputs = self._forward_helper(
            data,
        )

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            print(
                "{} Time: {}\tMemory: {}\t{}".format(
                    self.counter,
                    time.time() - start_time,
                    len(data.pos),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        self.counter = self.counter + 1

        return outputs

    # restructure forward helper for conditional grad
    def _forward_helper(self, data):
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Calculate which message block each edge should use. Based on edge distance rank.
        edge_rank = self._rank_edge_distances(
            edge_distance, edge_index, self.max_num_neighbors
        )

        # Reorder edges so that they are grouped by distance rank (lowest to highest)
        last_cutoff = -0.1
        message_block_idx = torch.zeros(len(edge_distance), device=pos.device)
        edge_distance_reorder = torch.tensor([], device=self.device)
        edge_index_reorder = torch.tensor([], device=self.device)
        edge_distance_vec_reorder = torch.tensor([], device=self.device)
        cutoff_index = torch.tensor([0], device=self.device)

        for i in range(self.num_resolutions):
            mask = torch.logical_and(
                edge_rank.gt(last_cutoff), edge_rank.le(self.cutoff_list[i])
            )
            last_cutoff = self.cutoff_list[i]
            message_block_idx.masked_fill_(mask, i)
            edge_distance_reorder = torch.cat(
                [
                    edge_distance_reorder,
                    torch.masked_select(edge_distance, mask),
                ],
                dim=0,
            )
            edge_index_reorder = torch.cat(
                [
                    edge_index_reorder,
                    torch.masked_select(
                        edge_index, mask.view(1, -1).repeat(2, 1)
                    ).view(2, -1),
                ],
                dim=1,
            )
            edge_distance_vec_mask = torch.masked_select(
                edge_distance_vec, mask.view(-1, 1).repeat(1, 3)
            ).view(-1, 3)
            edge_distance_vec_reorder = torch.cat(
                [edge_distance_vec_reorder, edge_distance_vec_mask], dim=0
            )
            cutoff_index = torch.cat(
                [
                    cutoff_index,
                    torch.tensor(
                        [len(edge_distance_reorder)], device=self.device
                    ),
                ],
                dim=0,
            )

        edge_index = edge_index_reorder.long()
        edge_distance = edge_distance_reorder
        edge_distance_vec = edge_distance_vec_reorder
        
        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # center = torch.mean(pos, dim=0, keepdim=True)
        # node_vec = pos - center
        node_vec = None
        # node_neigh_vec = node_vec[edge_index[0, :]]

        # node_rot_mat = self._init_node_rotinv_mat(
        #     node_vec
        # )
        # node_vec = torch.norm(node_vec, dim=1, p=2)

        # edge_rot_mat = node_rot_mat[edge_index[1, :]] @ edge_rot_mat

        # TODO norm the atom information
        # node_sh = o3.spherical_harmonics(torch.arange(0, self.lmax + 1).tolist(), node_vec, False).to(edge_distance.device)

        edge_sh = o3.spherical_harmonics(torch.arange(0, self.lmax + 1).tolist(), edge_distance_vec, normalize=True, normalization='component')

        # node_neigh_sh = o3.spherical_harmonics(torch.arange(0, self.lmax_reduce + 1).tolist(), node_neigh_vec, False).to(edge_distance.device)

        # node_neigh_sh = node_neigh_sh.unsqueeze(1).repeat(1, self.sphere_channels, 1)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        eq_edge = []
        for i in range(self.num_resolutions):
            self.sphharm_list[i].InitWignerDMatrix(
                edge_rot_mat[cutoff_index[i] : cutoff_index[i + 1]],
            )
            eq_edge_temp = torch.zeros(
                num_atoms,
                self.sphharm_list[i].sphere_basis,
                dtype=pos.dtype,
                device=pos.device,
            )

            eq_edge_temp.index_add_(
                0, edge_index[1, cutoff_index[i] : cutoff_index[i + 1]], edge_sh[cutoff_index[i] : cutoff_index[i + 1]][:, :self.sphharm_list[i].sphere_basis].to(pos.dtype)
            )

            eq_edge.append(eq_edge_temp / 1.3)
            # self.sphharm_node[i].InitWignerDMatrix(
            #     node_rot_mat
            # )
        # self.sphharm_node.InitWignerDMatrix(
        #         node_rot_mat
        #     )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = torch.zeros(
            num_atoms,
            self.sphere_basis,
            self.sphere_channels,
            device=pos.device,
        )
        x_0 = self.sphere_embedding(atomic_numbers)
        x[:, 0, :] += self.act(self.fc_l0(x_0))

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        # ans = torch.tensor([], device=x.device)
        x_time = None
        for i, interaction in enumerate(self.edge_blocks):
            if i > 0:
                x, x_time = interaction(
                    x, x_time, atomic_numbers, edge_distance, edge_distance_vec, edge_index, cutoff_index, eq_edge, edge_sh, node_vec)
                # ans = torch.cat([ans, x_time.mean(dim=1)], dim=1)
            else:
                x, x_time = interaction(
                    x, x_0, atomic_numbers, edge_distance, edge_distance_vec, edge_index, cutoff_index, eq_edge, edge_sh, node_vec)
                # ans = torch.cat([ans, x_time.mean(dim=1)], dim=1)

        ###############################################################
        # Estimate energy and forces using the node embeddings
        ###############################################################

        # Create a roughly evenly distributed point sampling of the sphere
        sphere_points = CalcSpherePoints(
            self.num_sphere_samples, x.device
        ).detach()
        sphharm_weights = o3.spherical_harmonics(
            torch.arange(0, self.lmax + 1).tolist(), sphere_points, False
        ).detach()

        # Energy estimation
        node_energy = torch.einsum(
            "abc, pb->apc", x, sphharm_weights
        ).contiguous()
        # # node_energy = 
        # node_res = self.act(self.energy_res1(ans))
        # node_res = self.energy_res2(node_res)
        node_energy = node_energy.view(-1, self.sphere_channels)
        node_energy = self.act(self.energy_fc1(node_energy))
        node_energy = self.act(self.energy_fc2(node_energy))
        node_energy = self.energy_fc3(node_energy)
        node_energy = node_energy.view(-1, self.num_sphere_samples, 1)
        node_energy = (torch.sum(node_energy, dim=1) / self.num_sphere_samples)# + node_res
        # node_energy = self.energy_com(torch.cat([node_en]))
        energy = torch.zeros(len(data.natoms), device=pos.device)
        energy.index_add_(0, data.batch, node_energy.view(-1))

        # Force estimation
        if self.regress_forces:
            forces = torch.einsum(
                "abc, pb->apc", x, sphharm_weights
            ).contiguous()
            forces = forces.view(-1, self.sphere_channels)
            forces = self.act(self.force_fc1(forces))
            forces = self.act(self.force_fc2(forces))
            forces = self.force_fc3(forces)
            forces = forces.view(-1, self.num_sphere_samples, 1)
            forces = forces * sphere_points.view(1, self.num_sphere_samples, 3)
            forces = torch.sum(forces, dim=1) / self.num_sphere_samples

            data.forces = forces

        # data.total_energy = energy
        # data.esp_charge = node_energy
        return energy

        # if not self.regress_forces:
        #     return energy
        # else:
        #     return energy, forces
    
    def _init_node_rotinv_mat(self, node_vec):
        node_vec_0 = node_vec
        node_vec_0_distance = torch.sqrt(torch.sum(node_vec_0**2, dim=1))

        if torch.min(node_vec_0_distance) < 0.0001:
            print(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(node_vec_0_distance)
                )
            )
            (minval, minidx) = torch.min(node_vec_0_distance, 0)

        norm_x = node_vec_0 / (node_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(node_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
            -1, 1
        )
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
            -1, 1
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
        )
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
        )
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
        )
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
        )

        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        # edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat_inv.detach()

    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        if torch.min(edge_vec_0_distance) < 0.0001:
            print(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(edge_vec_0_distance)
                )
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            print(
                "Error edge_vec_0_distance: {} {} {} {} {}".format(
                    minidx,
                    edge_index[0, minidx],
                    edge_index[1, minidx],
                    data.pos[edge_index[0, minidx]],
                    data.pos[edge_index[1, minidx]],
                )
            )

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
            -1, 1
        )
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
            -1, 1
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
        )
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
        )
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
        )
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
        )

        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    def _rank_edge_distances(
        self, edge_distance, edge_index, max_num_neighbors
    ):
        device = edge_distance.device
        # Create an index map to map distances from atom_distance to distance_sort
        # index_sort_map assumes index to be sorted
        output, num_neighbors = torch.unique(edge_index[1], return_counts=True)
        index_neighbor_offset = (
            torch.cumsum(num_neighbors, dim=0) - num_neighbors
        )
        index_neighbor_offset_expand = torch.repeat_interleave(
            index_neighbor_offset, num_neighbors
        )

        index_sort_map = (
            edge_index[1] * max_num_neighbors
            + torch.arange(len(edge_distance), device=device)
            - index_neighbor_offset_expand
        )

        num_atoms = torch.max(edge_index) + 1
        distance_sort = torch.full(
            [num_atoms * max_num_neighbors], np.inf, device=device
        )
        distance_sort.index_copy_(0, index_sort_map, edge_distance)
        distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
        no_op, index_sort = torch.sort(distance_sort, dim=1)

        index_map = (
            torch.arange(max_num_neighbors, device=device)
            .view(1, -1)
            .repeat(num_atoms, 1)
            .view(-1)
        )
        index_sort = index_sort + (
            torch.arange(num_atoms, device=device) * max_num_neighbors
        ).view(-1, 1).repeat(1, max_num_neighbors)
        edge_rank = torch.zeros_like(index_map)
        edge_rank.index_copy_(0, index_sort.view(-1), index_map)
        edge_rank = edge_rank.view(num_atoms, max_num_neighbors)

        index_sort_mask = distance_sort.lt(1000.0)
        edge_rank = torch.masked_select(edge_rank, index_sort_mask)

        return edge_rank

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



class EdgeBlock(torch.nn.Module):
    def __init__(
        self,
        num_resolutions,
        sphere_channels_reduce,
        sphere_channels_l0,
        hidden_channels_list,
        cutoff_list,
        sphharm_list,
        sphere_channels,
        distance_expansion,
        max_num_elements,
        num_basis_functions,
        num_gaussians,
        use_grid,
        act,
        sphharm_node,
        time_flag,
    ):
        super(EdgeBlock, self).__init__()
        self.num_resolutions = num_resolutions
        self.act = act
        self.hidden_channels_list = hidden_channels_list
        self.sphere_channels = sphere_channels
        self.sphere_channels_reduce = sphere_channels_reduce
        self.distance_expansion = distance_expansion
        self.cutoff_list = cutoff_list
        self.sphharm_list = sphharm_list
        self.max_num_elements = max_num_elements
        self.num_basis_functions = num_basis_functions
        self.use_grid = use_grid
        self.num_gaussians = num_gaussians
        self.sphharm_node = sphharm_node
        self.lmax = int(self.sphharm_list[0].lmax)
        self.lmax_reduce = (self.lmax + 1) // 2
        self.sphere_channels_l0 = sphere_channels_l0

        # Edge features
        self.dist_block = DistanceBlock(
            self.num_gaussians,
            self.num_basis_functions,
            self.distance_expansion,
            self.max_num_elements,
            self.act,
        )

        self.layer_norm = EquivariantLayerNormV2_channel(self.sphharm_list[0].lmax, self.sphere_channels_reduce)

        # Create a message block for each cutoff
        self.message_blocks = nn.ModuleList()
        # self.layer_norm = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = MessageBlock(
                self.sphere_channels_reduce,
                self.sphere_channels_l0,
                int(self.hidden_channels_list[i]),
                self.num_basis_functions,
                self.sphharm_list[i],
                self.act,
                self.sphharm_node,
                time_flag,
            )
            self.message_blocks.append(block)
            # self.layer_norm.append(EquivariantLayerNormV2_channel(self.sphharm_list[i].lmax, self.sphere_channels_reduce))

        # Downsampling number of sphere channels
        # Make sure bias is false unless equivariance is lost
        if self.sphere_channels != self.sphere_channels_reduce:
            self.downsample = nn.Linear(
                self.sphere_channels,
                self.sphere_channels_reduce,
                bias=False,
            )
            self.upsample = nn.Linear(
                self.sphere_channels_reduce,
                self.sphere_channels,
                bias=False,
            )

        # Use non-linear message aggregation?
        if self.use_grid:
            # Network for each node to combine edge messages
            self.fc1_sphere = nn.Conv2d(
                self.sphharm_list[0].num_bands
                * 4
                * self.sphere_channels_reduce,
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                kernel_size=1,
            )

            self.fc2_sphere = nn.Conv2d(
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                2 * self.sphere_channels_reduce,
                kernel_size=1,
            )

            self.fc3_sphere = nn.Conv2d(
                2 * self.sphere_channels_reduce, self.sphere_channels_reduce,
                kernel_size=1,
            )

            self.GAP = nn.AdaptiveAvgPool2d(1)
            self.SEattention = nn.Sequential(
                nn.Linear(2 * self.sphere_channels_reduce, 2 * self.sphere_channels_reduce // 16),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.sphere_channels_reduce // 16, 2 * self.sphere_channels_reduce),
                nn.Sigmoid()
                )

        # self.moe1 = MOE(self.sphere_channels_l0, self.sphere_channels_reduce, branch=2, mode='self')
        # self.moe2 = MOE(2 * self.sphere_channels_reduce, self.sphere_channels_reduce)

        self.inv_3 = 2 / 3
        self.time_flag = time_flag
        if time_flag == 1:
            self.fc_edge = nn.Linear(self.num_basis_functions, self.sphere_channels_l0)
            # self.bn
            # self.fc_l0 = nn.Linear(self.sphere_channels, self.sphere_channels_l0)
            # self.l0_bn = nn.BatchNorm1d(self.sphere_channels_l0)

    def forward(
        self,
        x,
        x_time,
        atomic_numbers,
        edge_distance,
        edge_vec,
        edge_index,
        cutoff_index,
        edge_sh_add,
        edge_sh,
        node_vec,
    ):

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        x_edge = self.dist_block(
            edge_distance,
            atomic_numbers[edge_index[0]],
            atomic_numbers[edge_index[1]],
        )


        # x_glovec = self.dist_block_node(
        #     node_vec,
        #     atomic_numbers,
        # )
        # x_glovec = node_sh 
        x_glovec = None

        node_num = len(x)
        x_new = torch.zeros(
            node_num,
            self.sphharm_list[0].sphere_basis,
            self.sphere_channels_reduce,
            dtype=x.dtype,
            device=x.device,
        )

        if self.sphere_channels != self.sphere_channels_reduce:
            x_down = self.downsample(x.view(-1, self.sphere_channels))
        else:
            x_down = x
        x_down = x_down.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels_reduce
        )

        node_sh_inv = None
        # moe1 = self.moe1(x_time)
        # x_input = moe1[:, 0:1, :] * x_down + torch.bmm(edge_sh_add[0].unsqueeze(-1), moe1[:, 1:2, :]) 
        # x_input = x_down
        
        x_input = self.layer_norm(x_down)

        for i, interaction in enumerate(self.message_blocks):
            start_idx = cutoff_index[i]
            end_idx = cutoff_index[i + 1]

            inv_edge = None

            eq_edge = edge_sh_add[i].unsqueeze(-1)
            # .expand(node_num, self.sphharm_list[i].sphere_basis, self.sphere_channels_reduce)
            
            # moe2 = self.moe2(x_time).unsqueeze(1)
            
            # x_input = moe1[:, 0:1, :] * x_down[:, 0 : self.sphharm_list[i].sphere_basis, :] + torch.bmm(eq_edge, moe1[:, 1:2, :]) 

            x_message = interaction(
                x_input[:, 0 : self.sphharm_list[i].sphere_basis, :],
                x_time,
                x_glovec,
                x_edge[start_idx:end_idx],
                edge_index[:, start_idx:end_idx],
                edge_vec[start_idx:end_idx],
                edge_sh[start_idx:end_idx],
                inv_edge,
                eq_edge,
                node_sh_inv,
                node_self=None,
            )

            # Sum all incoming edges to the target nodes
            x_new[:, 0 : self.sphharm_list[i].sphere_basis, :].index_add_(
                0, edge_index[1, start_idx:end_idx], x_message.to(x_new.dtype)
            )

        if self.use_grid:
            # Feed in the spherical functions from the previous time step
            x_grid = self.sphharm_list[0].ToGrid_inv(
                x_down, self.sphere_channels_reduce
            )
            x_grid = torch.cat(
                [
                    x_grid,
                    self.sphharm_list[0].ToGrid_inv(
                        x_new, self.sphere_channels_reduce
                    ),
                ],
                dim=-1,
            )
            x_grid = x_grid.permute(0, 3, 1, 2)
            x_grid = self.act(self.fc1_sphere(x_grid))
            x_grid = self.act(self.fc2_sphere(x_grid))
            x_attn = self.GAP(x_grid).view(node_num, -1)
            x_attn = self.SEattention(x_attn)
            x_grid *= x_attn.view(node_num, -1, 1, 1)
            x_grid = self.fc3_sphere(x_grid)
            # x_new_time = x_grid.reshape(node_num, -1, self.sphere_channels_reduce)
            x_grid = x_grid.permute(0, 2, 3, 1)
            x_new = self.sphharm_list[0].FromGrid(
                x_grid, self.sphere_channels_reduce
            )           

        if self.sphere_channels != self.sphere_channels_reduce:
            x_new = x_new.view(-1, self.sphere_channels_reduce)
            x_new = self.upsample(x_new)
        x_new = x_new.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels
        )

        inv_edge = torch.zeros(
            node_num,
            self.num_basis_functions,
            dtype=x.dtype,
            device=x.device,
        )
        inv_edge.index_add_(
            0, edge_index[1, cutoff_index[0]:cutoff_index[1]], x_edge[cutoff_index[0]:cutoff_index[1]].to(inv_edge.dtype)
        )

        # inv_edge = self.act(self.fc_edge(inv_edge))
        if self.time_flag == 1:
            # x_new_time = self.act(self.fc_edge(inv_edge) + x_time)
            x_new_time = self.act(self.fc_edge(inv_edge) * self.inv_3 + x_time)
        else:
            x_new_time = x_time
        # x_new_time = x_time
        return x_new, x_new_time

class MOE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, branch, mode='self'):
        super(MOE, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, input_dim // 4)
        self.fc2 = nn.Linear(input_dim // 4, branch * output_dim)
        self.branch = branch
        self.mode = mode
        if self.mode == 'self':
            self.mode
        else:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        num = x.shape[0]
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        if self.mode == 'self':
            out = out.view(num, self.branch, -1)
            return torch.softmax(out, 1)
        elif self.mode == 'message':
            out = out.view(num, self.branch, -1)
            return self.sigmoid(out)
        else:
            raise NotImplementedError

class MessageBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels_reduce,
        sphere_channels_l0,
        hidden_channels,
        num_basis_functions,
        sphharm,
        act,
        sphharm_node=None,
        time_flag=None,
    ):
        super(MessageBlock, self).__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.hidden_channels_reduce = hidden_channels // 4
        self.sphere_channels_reduce = sphere_channels_reduce
        self.sphere_channels_l0 = sphere_channels_l0
        self.sphharm = sphharm
        # self.sphharm_node = sphharm_node
        self.branch = 5
        self.fc1_dist = nn.Linear(num_basis_functions, self.hidden_channels)
        self.fc2_dist = nn.Linear(num_basis_functions, self.hidden_channels)
        self.gate = MOE(num_basis_functions + 2 * self.sphere_channels_l0, self.sphere_channels_reduce, branch=self.branch, mode='message')
        self.m = torch.Tensor(data=[0])
        for i in range(1, self.sphharm.lmax + 1):
            self.m = torch.cat([self.m, torch.Tensor(data=[i ** 2 + i - 1, i ** 2 + i, i ** 2 + i + 1])])

        self.lmax = int(np.sqrt(self.sphharm.sphere_basis)) - 1
        self.lmax_reduce = (self.lmax + 1) // 2
        self.lmax_reduce_twice = self.lmax_reduce
        self.sphere_bias = (self.lmax + 1) ** 2
        self.sphere_bias_reduce = (self.lmax_reduce + 1) ** 2

        self.l_middle = 2
        self.middle_bias = (self.l_middle + 1) ** 2

        self.sphere_bias_reducetwice = self.sphere_bias_reduce

        # self.layer_norm = EquivariantLayerNormV2()
        self.cg1 = cg_interaction(self.lmax, self.lmax, self.l_middle, self.lmax, channel=self.sphere_channels_reduce)
        # self.cg2 = cg_interaction(self.lmax, self.lmax, self.lmax, channel=self.sphere_channels_reduce)

        self.node_interaction1 = node_interaction(sphharm, self.lmax_reduce, self.hidden_channels, sphere_channels_reduce, m=self.m, concate=False)

        # self.node_interaction2 = node_interaction(sphharm, self.lmax_reduce, self.hidden_channels_reduce, sphere_channels_reduce, m=self.m, concate=False)

        self.node_interaction2 = node_interaction(sphharm, self.lmax_reduce, self.hidden_channels, sphere_channels_reduce, m=self.m, concate=False)

        # self.node_depthwise = o3.elementwisetensorproduct()
        self.scale = 1
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)

        # self.linear = o3.Linear()
        
    def forward(
        self,
        x,
        x_time,
        x_glovec,
        f_edge,
        edge_index,
        edge_vec,
        edge_sh,
        inv_edge,
        eq_edge,
        node_sh_inv,
        node_self,
    ):
        ###############################################################
        # Compute messages
        ###############################################################
        edge_num = f_edge.shape[0]
        x = x * self.inv_sqrt_3
        # x = self.layer_norm(x)
        s_l0, t_l0 = x_time[edge_index[0, :]], x_time[edge_index[1, :]]
        x_edge = self.act(self.fc1_dist(f_edge))
        x_edge_c = self.act(self.fc2_dist(f_edge))
        x_source, x_target = x[edge_index[0, :]], x[edge_index[1, :]]
        edge_with_ne = torch.cat([f_edge, s_l0, t_l0], dim=1)
        gating_logits = self.gate(edge_with_ne)
        # message_l0 = self.act(self.fc_l0(edge_with_ne))

        # cross message
        # equivariant branch: cg tensor product,
        out_message = torch.zeros_like(x_source)

        cg_lowdegree, cg_all = self.cg1(x_source + x_target, edge_sh)
        out_message[:, :self.middle_bias, :] += gating_logits[:, 3:4, :] * cg_lowdegree
        out_message += torch.bmm(cg_all.unsqueeze(-1), gating_logits[:, 4:5, :])


        # approximate equivariant branch
        x_msg_source = self.sphharm.Rotate(x_source)
        x_msg_target = self.sphharm.Rotate(x_target)
        m = self.m.to(device=edge_index.device, dtype=edge_index.dtype)
        s_reduce = x_source[:, m, :].reshape(edge_num, -1)
        t_reduce = x_target[:, m, :].reshape(edge_num, -1)
        # x_message = torch.cat([x_msg_source, x_msg_target], dim=1)
        # o_message = torch.cat([s_reduce, t_reduce], dim=1)
        x_message = x_msg_source + x_msg_target
        o_message = s_reduce + t_reduce

        non_message = self.node_interaction1(x_msg_source, x_edge_c, edge_num, s_reduce) * gating_logits[:, 0:1, :] + self.node_interaction1(x_msg_target, x_edge_c, edge_num, t_reduce) * gating_logits[:, 1:2, :] + gating_logits[:, 2:3, :] * self.node_interaction2(x_message, x_edge, edge_num, o_message)

        if self.scale != 1:
            non_message = self.scale * non_message

        out_message[:, seg:, :] += self.sphharm.RotateInv(non_message)

        return out_message

class node_interaction(torch.nn.Module):
    def __init__(self, sphharm_node, lmax, hidden_channel, sphere_channels_reduce, m, concate=False):
        super(node_interaction, self).__init__()
        self.sphharm_node = sphharm_node
        self.lmax = lmax
        self.sphere_channels_reduce = sphere_channels_reduce
        self.hidden_channel = hidden_channel
        self.basis_num = (self.lmax + 1) ** 2
        self.m = m

        if concate:
            self.node_fc1 = nn.Linear(2 * self.sphharm_node.sphere_basis_reduce * self.sphere_channels_reduce, self.hidden_channel)
        else:
            self.node_fc1 = nn.Linear(self.sphharm_node.sphere_basis_reduce * self.sphere_channels_reduce, self.hidden_channel)
        self.node_fc2 = nn.Linear(self.hidden_channel, (self.sphharm_node.sphere_basis_reduce - seg_reduce) * self.sphere_channels_reduce)

        # TODO sph act
        self.act = nn.SiLU()

    def forward(self, node_embed, x_edge_c, edge_num, ori):
        node_embed = torch.cat([node_embed, ori], dim=0)
        node_embed = self.act(self.node_fc1(node_embed))
        node_embed = (
            node_embed.view(
                2, -1, self.hidden_channel
            )
        ) * x_edge_c.view(1, -1, self.hidden_channel)
        node_embed = node_embed.view(-1, self.hidden_channel)
        node_embed = self.act(self.node_fc2(node_embed))
        node_embed = node_embed.view(2 * edge_num, -1, self.sphere_channels_reduce)
        node_embed = node_embed[:edge_num] + node_embed[edge_num:]
        return node_embed

class cg_interaction(torch.nn.Module):
    def __init__(self, l1, l2, l_middel, l_out, channel=128, right=None, fc=True, specific_irreps=None):
        super(cg_interaction, self).__init__()

        self.l1 = l1
        self.l2 = l2
        self.middle = l_middel
        self.l_out = l_out
        self.l1_bias = (self.l1 + 1) ** 2
        self.l2_bias = (self.l2 + 1) ** 2
        self.middle_bias = (self.middle + 1) ** 2
        self.lout_bias = (self.l_out + 1) ** 2
        self.act = nn.SiLU()
        self.channel = channel

        if specific_irreps is not None:
            irreps_in1, irreps_in2, irreps_out = specific_irreps
        else:
            irreps_in1 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l1))).sort().irreps.simplify()                
            irreps_in2 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l2))).sort().irreps.simplify()
            irreps_midd_0 = (channel * o3.Irreps(str(o3.Irreps.spherical_harmonics(self.middle)))).sort().irreps.simplify()
            irreps_midd = irreps_midd_0[1:]
            irreps_out_0 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l_out))).sort().irreps.simplify()
            irreps_out = irreps_out_0[1:]
        
        self.cg = FullyConnectedTensorProductRescaleSwishGate(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_midd_0,
            irreps_norm=irreps_midd_0)
        # EquivariantLayerNormV2(irreps_in1)
        self.cg_all = FullyConnectedTensorProductRescaleSwishGate(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out_0,
            irreps_norm=irreps_out_0)
        self.index_l = []
        self.index_c = []
        for i in range(self.middle + 1):
            self.index_l.append(torch.arange(2 * i + 1) + i ** 2)
            self.index_c.append(torch.arange((2 * i + 1) * self.channel) + self.channel * (i ** 2))

    def forward(self, x_o, y_o, inv_feature=None):
       
        if len(x_o.shape) == 3:
            edge_num, basis_num_x, channel = x_o.shape
            x = x_o.mean(2)
        else:
            edge_num, basis_num_x = x_o.shape
            x = x_o

        if len(y_o.shape) == 3:
            y = y_o[:, :self.l2_bias, :].mean(2)
        else:
            y = y_o[:, :self.l2_bias]
        cg_lowdegree = self.cg(x, y)
        cg_all = self.cg_all(x, y)
        cg_lowdegree = self.reshape(cg_lowdegree)
        return cg_lowdegree, cg_all

    def reshape(self, x):
        edge_num, num = x.shape
        out = torch.zeros((edge_num, self.middle_bias, self.channel), device=x.device)
        for i in range(self.middle + 1):
            out[:, self.index_l[i], :] += x[:, self.index_c[i]].view(edge_num, self.channel, -1).transpose(1, 2)

        return out

    def cg_inv(self, z):
        edge_num, channel, basis_num_z = z.shape
        cg_inv = self.cg.right(self.fixed_y.to(z.device)).transpose(0, 1)
        cg_inv = cg_inv.unsqueeze(0).repeat(edge_num, 1, 1)
        return z + torch.linalg.solve(cg_inv, torch.mean(z, dim=1)).unsqueeze(1)
    
class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out, irreps_norm, 
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None, inv_feature=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    
class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out, irreps_norm, 
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.norm = EquivariantLayerNormV2(irreps_norm)
        
        
    def forward(self, x, y, batch, inv_feature, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = torch.cat([inv_feature, out], dim=1)
        out = self.norm(out, batch=batch)
        return out

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=True)
    return tp   

class DistanceBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis_functions,
        distance_expansion,
        max_num_elements,
        act,
    ):
        super(DistanceBlock, self).__init__()
        self.in_channels = in_channels
        self.distance_expansion = distance_expansion
        self.act = act
        self.num_basis_functions = num_basis_functions
        self.max_num_elements = max_num_elements
        self.num_edge_channels = self.num_basis_functions

        self.fc1_dist = nn.Linear(self.in_channels, self.num_basis_functions)

        self.source_embedding = nn.Embedding(
            self.max_num_elements, self.num_basis_functions
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements, self.num_basis_functions
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.fc1_edge_attr = nn.Linear(
            self.num_edge_channels,
            self.num_edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        x_edge = self.act(source_embedding + target_embedding + x_dist)
        x_edge = self.act(self.fc1_edge_attr(x_edge))

        return x_edge
