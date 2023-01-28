"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from IPython import embed
import logging
import time

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
from ocpmodels.models.hdgnn.spherical_harmonics import SphericalHarmonicsHelper, SphericalHarmonicsHelper_4
# from ocpmodels.models.hdgnn.spherical_harmonics_4 import SphericalHarmonicsHelper_4

try:
    import e3nn
    from e3nn import o3, io
except ImportError:
    pass

@registry.register_model("hdgnn")
class HDGNN(BaseModel):

    def __init__(
        self,
        # num_atoms,  # not used
        # bond_feat_dim,  # not used
        # num_targets,  # not used
        use_pbc=False,
        regress_forces=False,
        otf_graph=False,
        max_num_neighbors=60,
        cutoff=8.0,
        max_num_elements=90,
        num_interactions=4,
        lmax=6,
        mmax=1,
        num_resolutions=1,
        sphere_channels=16,
        sphere_channels_reduce=16,
        hidden_channels=16,
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

        self.regress_forces = False
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
        self.max_num_neighbors = self.max_neighbors = max_num_neighbors
        self.num_basis_functions = num_basis_functions
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax = lmax
        self.mmax = mmax
        self.basis_width_scalar = basis_width_scalar
        self.sphere_basis = (self.lmax + 1) ** 2
        self.use_grid = use_grid
        self.distance_function = distance_function

        # variables used for display purposes
        self.counter = 0

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

        self.lmax_reduce = (self.lmax_list[i] + 1) // 2
        self.sphharm_node = SphericalHarmonicsHelper(
                    self.lmax_reduce,
                    self.mmax,
                    num_taps,
                    num_bands,
                )
        # self.sphharm_node = None

        # Weights for message initialization
        # self.sphere_embedding = nn.Embedding(
        #     self.max_num_elements, (self.lmax_list[0] + 1) * (self.lmax_list[0] + 1) * self.sphere_channels
        # )
        # self.sphere_embedding = nn.Embedding(
        #     self.max_num_elements, (self.lmax_list[0] + 1) * (self.lmax_list[0] + 1) * self.sphere_channels
        # )
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        self.edge_blocks = nn.ModuleList()
        for i in range(self.num_interactions):
            if i == 0:
                time_flag = 0
            else:
                time_flag = 1
            block = EdgeBlock(
                self.num_resolutions,
                self.sphere_channels_reduce,
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

        self.energy_res1 = nn.Linear(self.sphere_channels * self.num_interactions, self.sphere_channels)
        self.energy_res2 = nn.Linear(self.sphere_channels, 1)

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
        atomic_numbers = data.z.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        edge_index = data.edge_index[[1, 0],:]
        j, i = edge_index
        edge_distance_vec = data.pos[j] - data.pos[i]

        edge_distance = edge_distance_vec.norm(dim=-1)

        # (
        #     edge_index,
        #     edge_distance,
        #     edge_distance_vec,
        #     cell_offsets,
        #     _,  # cell offset distances
        #     neighbors,
        # ) = self.generate_graph(data)

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

        center = torch.mean(pos, dim=0, keepdim=True)
        node_vec = pos - center
        node_neigh_vec = node_vec[edge_index[0, :]]

        node_rot_mat = self._init_node_rotinv_mat(
            node_vec
        )
        node_vec = torch.norm(node_vec, dim=1, p=2)

        # edge_rot_mat = node_rot_mat[edge_index[1, :]] @ edge_rot_mat

        # TODO norm the atom information
        node_sh = o3.spherical_harmonics(torch.arange(0, self.lmax_reduce + 1).tolist(), node_vec, False).to(edge_distance.device)

        # node_neigh_sh = o3.spherical_harmonics(torch.arange(0, self.lmax_reduce + 1).tolist(), node_neigh_vec, False).to(edge_distance.device)

        # node_neigh_sh = node_neigh_sh.unsqueeze(1).repeat(1, self.sphere_channels, 1)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.sphharm_list[i].InitWignerDMatrix(
                edge_rot_mat[cutoff_index[i] : cutoff_index[i + 1]],
            )

        self.sphharm_node.InitWignerDMatrix(
                node_rot_mat
            )

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
        x[:, 0, :] = self.sphere_embedding(atomic_numbers)

        # x += self.sphere_embedding(atomic_numbers).view(num_atoms, self.sphere_basis, self.sphere_channels)

        # sphere_points = CalcSpherePoints(
        #     self.sphere_channels, x.device
        # ).detach()
        # sphharm_weights = o3.spherical_harmonics(
        #     torch.arange(0, self.lmax + 1).tolist(), sphere_points, False
        # ).detach()

        # x = x + sphharm_weights.transpose(0, 1).unsqueeze(0)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        ans = torch.tensor([], device=x.device)
        x_time = None
        for i, interaction in enumerate(self.edge_blocks):
            if i > 0:
                h, h_time = interaction(
                    x, x_time, atomic_numbers, edge_distance, edge_distance_vec, edge_index, cutoff_index, node_sh, node_vec)
                x = x + h
                x_time = x_time + h_time
                ans = torch.cat([ans, x_time.mean(dim=1)], dim=1)
            else:
                x, x_time = interaction(
                    x, x_time, atomic_numbers, edge_distance, edge_distance_vec, edge_index, cutoff_index, node_sh, node_vec)
                ans = torch.cat([ans, x_time.mean(dim=1)], dim=1)

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
        node_res = self.act(self.energy_res1(ans))
        node_res = self.energy_res2(node_res)
        node_energy = node_energy.view(-1, self.sphere_channels)
        node_energy = self.act(self.energy_fc1(node_energy))
        node_energy = self.act(self.energy_fc2(node_energy))
        node_energy = self.energy_fc3(node_energy)
        node_energy = node_energy.view(-1, self.num_sphere_samples, 1)
        node_energy = (torch.sum(node_energy, dim=1) / self.num_sphere_samples) + node_res
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

        # Edge features
        self.dist_block = DistanceBlock(
            self.num_gaussians,
            self.num_basis_functions,
            self.distance_expansion,
            self.max_num_elements,
            self.act,
        )

        # Node vec feature
        self.dist_block_node = DistanceBlock2(
            self.num_gaussians,
            self.num_basis_functions,
            self.distance_expansion,
            self.max_num_elements,
            self.sphere_channels,
            self.act,
        )

        # self.node_interaction = node_interaction(self.sphharm_node, self.lmax, self.sphere_channels)
        self.y_sh_reduce = o3.spherical_harmonics(torch.arange(0, 2).tolist(), torch.Tensor(data=[0, 1, 0]), False)
        # self.lmax_reduce = (self.lmax + 1) // 2 
        self.cg_self = cg_interaction(self.lmax, 1, self.lmax_reduce, right=self.y_sh_reduce)

        # Create a message block for each cutoff
        self.message_blocks = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = MessageBlock(
                self.sphere_channels_reduce,
                int(self.hidden_channels_list[i]),
                self.num_basis_functions,
                self.sphharm_list[i],
                self.act,
                self.sphharm_node,
                time_flag,
            )
            self.message_blocks.append(block)

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
            self.fc1_sphere = nn.Linear(
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
            )

            self.fc2_sphere = nn.Linear(
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                2 * self.sphere_channels_reduce,
            )

            self.fc3_sphere = nn.Linear(
                2 * self.sphere_channels_reduce, self.sphere_channels_reduce
            )


            # ours
            # self.fc1_sphere = nn.Linear(
            #     49 * 2 * self.sphere_channels_reduce,
            #     self.sphere_channels_reduce
            # )

            # self.fc2_sphere = nn.Linear(
            #     self.sphere_channels_reduce,
            #     self.sphere_channels_reduce

            # )

            # self.fc3_sphere = nn.Linear(
            #     self.sphere_channels_reduce, 49 * self.sphere_channels_reduce
            # )
            # # self, l1, l2, l_out, self_fixed=False, specific_irreps=None
            # self.cg_sh = cg_interaction(self.lmax, self.lmax, self.lmax)
            # self.cg_re = cg_interaction(self.lmax, self.lmax, self.lmax)
            # self.cg_down = cg_interaction(self.lmax, 1, self.lmax)
            # self.cg_new = cg_interaction(self.lmax, 1, self.lmax)

        # dynamic sh
        # self.sh_basis = nn.Parameter(data=torch.randn(12, 3))


        # self.sh_basis = CalcSpherePoints(
        #     16, 'cuda'
        # ).detach()
        # self.attention = sh_attention(self.sphere_channels, self.sphere_channels, self.lmax_reduce)

    def radial_function(self, x, y):
        b, l, c  = x.shape
        attention = self.attention(x)
        _, m, n = attention.shape
        l_reduce = y.shape[-1]
        radial = torch.zeros(
            b,
            m,
            n,
            l_reduce,
            dtype=x.dtype,
            device=x.device,
        )
        attention = attention.unsqueeze(-1)
        k = torch.ones_like(attention).to(device=x.device)
        lmax = int(math.sqrt(l_reduce)) - 1
        for i in range(lmax + 1):
            radial[:, :, :, (i + 1)**2 - (2 * i + 1) : (i + 1)**2] += k
            k = k * attention
        return torch.einsum('cmns,ns->csm', radial, y)

    def forward(
        self,
        x,
        x_time,
        atomic_numbers,
        edge_distance,
        edge_vec,
        edge_index,
        cutoff_index,
        node_sh,
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
        x_glovec = node_vec.unsqueeze(-1)

        node_num = len(x)
        x_new = torch.zeros(
            node_num,
            self.sphharm_list[0].sphere_basis,
            self.sphere_channels_reduce,
            dtype=x.dtype,
            device=x.device,
        )
        
        # x_new = self.sphharm_list[0].FromGrid(
            #     x_grid, self.sphere_channels_reduce
            # )

        if self.sphere_channels != self.sphere_channels_reduce:
            x_down = self.downsample(x.view(-1, self.sphere_channels))
        else:
            x_down = x
        x_down = x_down.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels_reduce
        )

        node_sh_inv = node_sh
        # node_sh_inv = self.node_interaction(x_down, node_sh)
        # y = self.y_sh_reduce.to(device=x.device).unsqueeze(0).repeat(node_num, 1)
        # node_self = self.cg_self(x_down)
        
        # dynamic 3D
        # node_self = o3.spherical_harmonics(torch.arange(0, self.lmax_reduce + 1).tolist(), self.attention(x_down) @ self.sh_basis, False).to(x.device)
        # node_self = node_self.transpose(1, 2)
        # node_self += (self.attention(x_down) @ self.sh_basis).transpose(1, 2)
        # node_self = node_self / 2

        # dynamic l**2 D
        # sh_basis = o3.spherical_harmonics(torch.arange(0, self.lmax_reduce + 1).tolist(), self.sh_basis, False).to(x.device)
        # node_self = self.radial_function(x_down, sh_basis)
        # node_self += self.attention(x_down)

        for i, interaction in enumerate(self.message_blocks):
            start_idx = cutoff_index[i]
            end_idx = cutoff_index[i + 1]
            x_message = interaction(
                x_down[:, 0 : self.sphharm_list[i].sphere_basis, :],
                x_time,
                x_glovec,
                x_edge[start_idx:end_idx],
                edge_index[:, start_idx:end_idx],
                edge_vec[start_idx:end_idx],
                node_sh_inv,
                node_self=None,
            )

            # Sum all incoming edges to the target nodes
            x_new[:, 0 : self.sphharm_list[i].sphere_basis, :].index_add_(
                0, edge_index[1, start_idx:end_idx], x_message.to(x_new.dtype)
            )

        if self.use_grid:
            # Feed in the spherical functions from the previous time step
            x_grid = self.sphharm_list[0].ToGrid(
                x_down, self.sphere_channels_reduce
            )
            x_grid = torch.cat(
                [
                    x_grid,
                    self.sphharm_list[0].ToGrid(
                        x_new, self.sphere_channels_reduce
                    ),
                ],
                dim=1,
            )
            x_grid = self.act(self.fc1_sphere(x_grid))
            x_grid = self.act(self.fc2_sphere(x_grid))
            x_grid = self.fc3_sphere(x_grid)
            x_new_time = x_grid.reshape(node_num, -1, self.sphere_channels_reduce)
            x_new = self.sphharm_list[0].FromGrid(
                x_grid, self.sphere_channels_reduce
            )

            

        if self.sphere_channels != self.sphere_channels_reduce:
            x_new = x_new.view(-1, self.sphere_channels_reduce)
            x_new = self.upsample(x_new)
        x_new = x_new.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels
        )

        return x_new, x_new_time

class sh_attention(torch.nn.Module):
    def __init__(
        self,
        input_channel,
        out_channel,
        lmax, 
        reduce_ratio = 4,
        sample = 16,
    ):
        super(sh_attention, self).__init__()
        self.out_channel = out_channel
        self.fc1 = nn.Linear(input_channel, input_channel // reduce_ratio)
        self.fc2 = nn.Linear(input_channel // reduce_ratio, sample * out_channel)
        self.act_mid = nn.ReLU(inplace=True)
        self.lmax = lmax
        # self.act = torch.tanh()
        # self.fc = nn.Linear(input_channel, sample * out_channel)
        sh_basis = CalcSpherePoints(
            sample, 'cuda'
        ).detach()
        self.sh_basis = o3.spherical_harmonics(torch.arange(0, self.lmax + 1).tolist(), sh_basis, False)

    def cal_attn(self, x):
        b, l, c = x.shape
        out = x.mean(dim=1)
        # out = self.act_mid(self.fc1(out))
        # return torch.sigmoid(self.fc2(out)).view(b, c, -1)
        # return torch.softmax(self.fc2(out).view(b, c, -1), dim=2)
        out = self.fc2(self.fc1(out))
        # out = self.act_mid(self.fc(out)).view(b, c, -1)
        return out.view(b, c, -1)

    def forward(self, x):
        b, l, c  = x.shape
        attention = self.cal_attn(x)
        _, m, n = attention.shape
        y = self.sh_basis.to(device=x.device)
        l_reduce = y.shape[-1]
        # radial = torch.zeros(
        #     b,
        #     m,
        #     n,
        #     l_reduce,
        #     dtype=x.dtype,
        #     device=x.device,
        # )
        # attention = attention.unsqueeze(-1)
        # k = torch.ones_like(attention).to(device=x.device)
        # lmax = int(math.sqrt(l_reduce)) - 1
        # for i in range(lmax + 1):
        #     radial[:, :, :, (i + 1)**2 - (2 * i + 1) : (i + 1)**2] += attention
        #     # k = k * attention
        # return torch.einsum('cmns,ns->csm', radial, y)
        return torch.einsum('cmn,ns->csm', attention, y)



class MessageBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels_reduce,
        hidden_channels,
        num_basis_functions,
        sphharm,
        act,
        sphharm_node,
        time_flag,
    ):
        super(MessageBlock, self).__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.hidden_channels_reduce = hidden_channels // 4
        self.sphere_channels_reduce = sphere_channels_reduce
        self.sphharm = sphharm
        self.sphharm_node = sphharm_node
        time_flag = 0
        if time_flag:
            self.fc1_dist = nn.Linear(num_basis_functions + 2 * self.hidden_channels, self.hidden_channels)
        else:
            self.fc1_dist = nn.Linear(num_basis_functions, self.hidden_channels)

        # self.fc2_dist = nn.Linear(num_basis_functions, self.hidden_channels)

        self.lmax = int(np.sqrt(self.sphharm.sphere_basis)) - 1
        self.lmax_reduce = (self.lmax + 1) // 2
        # self.lmax_reduce_twice = (self.lmax_reduce + 1) // 2 
        self.lmax_reduce_twice = self.lmax_reduce
        self.sphere_bias = (self.lmax + 1) ** 2
        self.sphere_bias_reduce = (self.lmax_reduce + 1) ** 2
        # self.sphere_bias_reducetwice = (self.lmax_reduce_twice + 1) * (self.lmax_reduce_twice + 1)

        self.sphere_bias_reducetwice = self.sphere_bias_reduce

        # Network for each edge to compute edge messages
        # (4 * self.sphere_bias + self.sphere_bias_reducetwice) * self.hidden_channels
        self.fc1_edge_proj = nn.Linear(
            # self.sphere_bias * self.hidden_channels * 2,
            self.sphharm.sphere_basis_reduce * self.sphere_channels_reduce,
            self.hidden_channels,
        )

        self.fc2_edge = nn.Linear(
            self.hidden_channels,
            self.sphharm.sphere_basis_reduce * self.sphere_channels_reduce,
        )

        # self.cg_self = cg_interaction(self.lmax, 1, self.lmax_reduce, right=self.y_sh_reduce)
        self.cg_node1 = cg_interaction2(self.lmax, self.lmax, self.lmax)
        # self.cg_node2 = cg_interaction2(self.lmax, self.lmax_reduce, self.lmax_reduce)
        # self.edge_sh = BasisGenerator(1, direction=True)

        self.node_interaction1 = node_interaction(self.sphharm_node, self.lmax_reduce, self.hidden_channels)
        self.node_interaction2 = node_interaction(self.sphharm_node, self.lmax_reduce, self.hidden_channels)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)
        
    def forward(
        self,
        x,
        x_time,
        x_glovec,
        x_edge,
        edge_index,
        edge_vec,
        node_sh_inv,
        node_self,
    ):

        ###############################################################
        # Compute messages
        ###############################################################
        edge_num = x_edge.shape[0]
        x_time = None
        if x_time is not None:
            # x_time = torch.norm(x, dim=1, p=2)
            x_time = torch.mean(x_time, dim=1) / 10
            x_time_s = x_time[edge_index[0, :]]
            x_time_t = x_time[edge_index[1, :]]
            x_edge = torch.cat([x_edge, x_time_s, x_time_t], dim=1)

        x_edge = self.act(self.fc1_dist(x_edge))
        x_source, x_node_feature_s = x[edge_index[0, :]], x_glovec[edge_index[0, :]]
        # node_self[edge_index[0, :]].transpose(1, 2)
        x_target, x_node_feature_t = x[edge_index[1, :]], x_glovec[edge_index[1, :]]
        # , node_self[edge_index[1, :]].transpose(1, 2)

        # x_msg_source = self.sphharm.Rotate_reduce(x_source).transpose(1, 2)
        # x_msg_target = self.sphharm.Rotate_reduce(x_target).transpose(1, 2)
        
        # # cross
        # node_self_s = self.sphharm.Rotate_reduce(node_self_s).transpose(1, 2)
        # node_self_t = self.sphharm.Rotate_reduce(node_self_t).transpose(1, 2)
        # node_self_s = x_msg_source[:, :, :16]
        # node_self_t = x_msg_target[:, :, :16]
        # node_sh_inv = node_sh_inv.unsqueeze(2)
        # node_sh_inv_s = self.sphharm.Rotate_reduce(node_sh_inv[edge_index[0, :]]).transpose(1, 2)
        # node_sh_inv_t = self.sphharm.Rotate_reduce(node_sh_inv[edge_index[1, :]]).transpose(1, 2)
        # node_sh_inv_t, node_sh_inv_s = None, None
        sh_cg = self.cg_node1(x_source, x_target)
        sh_source = self.node_interaction1(x_source, x_node_feature_t, edge_index, 1)
        sh_target = self.node_interaction2(x_target, x_node_feature_s, edge_index, 0)
        c = sh_cg
        c[:, :16, :] += (sh_source + sh_target)


        x_msg_source = self.sphharm.Rotate(x_source)
        x_msg_target = self.sphharm.Rotate(x_target)
        c_msg = self.sphharm.Rotate(c)

        # x_message = torch.cat([x_msg_source, x_msg_target, c_msg], dim=1)
        x_message = x_msg_source + x_msg_target + c_msg
        x_message = self.act(self.fc1_edge_proj(x_message))
        x_message = (
            x_message.view(
                -1, self.sphharm.num_y_rotations, self.hidden_channels
            )
        ) * x_edge.view(-1, 1, self.hidden_channels)

        x_message = x_message.view(-1, self.hidden_channels)

        # x_message = self.act(self.fc1_edge(x_message))
        x_message = self.act(self.fc2_edge(x_message))

        # Combine the rotated versions of the messages

        # Rotate the spherical harmonic basis functions back to global coordinate frame
        x_message = x_message.view(-1, self.sphere_channels_reduce)
        x_message = self.sphharm.CombineYRotations(x_message)
        x_message = self.sphharm.RotateInv(x_message) * self.inv_sqrt_3
        # x_message = torch.cat([x_message, x_source, x_target], dim=2).reshape(edge_num, -1)
        # # x_message = ((x_message + x_source + x_target)/3).reshape(edge_num, -1)

        # x_message = self.fc3_edge(x_message).reshape(edge_num, -1, self.hidden_channels)
        # x_message = x_message.view(edge_num, -1, self.hidden_channels)
        # x_message = self.sphharm.RotateInv(x_message).transpose(1, 2)
        # x_message = self.cg_self_inv.cg_inv(x_message).transpose(1, 2)

        return x_message

class node_interaction(torch.nn.Module):
    def __init__(self, sphharm_node, lmax, hidden_channel):
        super(node_interaction, self).__init__()
        self.sphharm_node = sphharm_node
        self.lmax = lmax
        # self.lmax_reduce = lmax_reduce
        self.hidden_channel = hidden_channel
        self.basis_num = (self.lmax + 1) ** 2

        self.node_fc1 = nn.Linear(self.basis_num * (hidden_channel + 1), hidden_channel)
        # self.node_fc2 = nn.Linear(self.hidden_channel, self.hidden_channel)
        self.node_fc3 = nn.Linear(self.hidden_channel, self.basis_num * hidden_channel)

        # TODO sph act
        self.act = nn.SiLU()
        # self.node_cg = cg_interaction(self.lmax, self.lmax, self.lmax)

    def forward(self, node_embed, node_dis_feature, edge_index, flag):
        b, l, c = node_embed.shape
        node_embed = self.sphharm_node.Rotate_edgeInv(edge_index[flag, :], node_embed[:, :self.basis_num, :])

        node_dis_feature = (node_embed.mean(dim=2) * node_dis_feature).unsqueeze(-1)
        node_embed.view(b, -1)
        node_embed = torch.cat([node_embed, node_dis_feature], dim=2).view(b, -1)

        node_embed = self.act(self.node_fc1(node_embed))
        # node_embed = self.act(self.node_fc2(node_embed))
        node_embed = self.act(self.node_fc3(node_embed)).reshape(b, self.basis_num, c)
        node_embed = self.sphharm_node.Rotate_edge(edge_index[flag, :], node_embed)
        return node_embed

# class node_interaction(torch.nn.Module):
#     def __init__(self, sphharm_node, lmax, hidden_channel):
#         super(node_interaction, self).__init__()
#         self.sphharm_node = sphharm_node
#         self.lmax = lmax
#         # self.lmax_reduce = lmax_reduce
#         self.hidden_channel = hidden_channel
#         self.basis_num = (self.lmax + 1) ** 2

#         self.node_fc1 = nn.Linear(self.basis_num * (hidden_channel + 1), hidden_channel)
#         self.node_fc2 = nn.Linear(self.hidden_channel, self.hidden_channel)
#         self.node_fc3 = nn.Linear(self.hidden_channel, self.basis_num * hidden_channel)

#         # TODO sph act
#         self.act = nn.SiLU()

#     def forward(self, node_embed, node_sh):
#         b, l, c = node_embed.shape
#         node_embed = self.sphharm_node.RotateInv(node_embed).view(b, -1)

#         # TODO learning Basis

#         # TODO channel wise fc
#         node_embed = torch.cat([node_embed, node_sh], dim=1)
#         node_embed = self.act(self.node_fc1(node_embed))
#         node_embed = self.act(self.node_fc2(node_embed))
#         node_embed = self.node_fc3(node_embed)

#         node_embed = self.sphharm_node.Rotate(node_embed.reshape(b, l, c))
#         return node_embed

class cg_interaction2(torch.nn.Module):
    def __init__(self, l1, l2, l_out, right=None, fc=True, specific_irreps=None):
        super(cg_interaction2, self).__init__()

        self.l1 = l1
        self.l2 = l2
        self.l_out = l_out
        self.l1_bias = (self.l1 + 1) * (self.l1 + 1)
        self.l2_bias = (self.l2 + 1) * (self.l2 + 1)
        self.lout_bias = (self.l_out + 1) * (self.l_out + 1)
        self.act = nn.SiLU()

        if specific_irreps is not None:
            irreps_in1, irreps_in2, irreps_out = specific_irreps
        else:
            irreps_in1 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l1))).sort().irreps.simplify()                
            irreps_in2 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l2))).sort().irreps.simplify()
            irreps_midd = o3.Irreps(str(o3.Irreps.spherical_harmonics(4))).sort().irreps.simplify()
            irreps_out = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l_out))).sort().irreps.simplify()

        self.cg1 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_midd,
            shared_weights=True, normalization='component', compile_right=True)
        
        self.cg21 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_midd,
            irreps_out=irreps_out,
            shared_weights=True, normalization='component', compile_right=True)

        self.cg22 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in2,
            irreps_in2=irreps_midd,
            irreps_out=irreps_out,
            shared_weights=True, normalization='component')
        
        # self.fc = fc
        # if fc:
        #     self.fc_x = nn.Linear(self.l1_bias, self.lout_bias)
        #     if right is None:
        #         self.fc_y = nn.Linear(self.l2_bias, self.lout_bias)

    def forward(self, x, y=None):
       
        edge_num, basis_num_x, channel = x.shape
        basis_num_y = y.shape[1]
        
        x_sc = x
        y_sc = y
        x = x.mean(2)
        y = y.mean(2)
        # out = torch.einsum('bik,bi->bk', self.cg1.right(y), x)
        out = self.cg1(x, y)
        out = x_sc + y_sc + self.cg21(x, out).unsqueeze(-1) + self.cg22(y, out).unsqueeze(-1)

        return out

    def cg_inv(self, z):
        edge_num, channel, basis_num_z = z.shape
        cg_inv = self.cg.right(self.fixed_y.to(z.device)).transpose(0, 1)
        cg_inv = cg_inv.unsqueeze(0).repeat(edge_num, 1, 1)
        return z + torch.linalg.solve(cg_inv, torch.mean(z, dim=1)).unsqueeze(1)

class cg_interaction(torch.nn.Module):
    def __init__(self, l1, l2, l_out, right=None, fc=False, specific_irreps=None):
        super(cg_interaction, self).__init__()

        self.l1 = l1
        self.l2 = l2
        self.l_out = l_out
        self.l1_bias = (self.l1 + 1) * (self.l1 + 1)
        self.l2_bias = (self.l2 + 1) * (self.l2 + 1)
        self.lout_bias = (self.l_out + 1) * (self.l_out + 1)
        self.act = nn.SiLU()

        if specific_irreps is not None:
            irreps_in1, irreps_in2, irreps_out = specific_irreps
        else:
            irreps_in1 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l1))).sort().irreps.simplify()                
            irreps_in2 = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l2))).sort().irreps.simplify() 
            irreps_out = o3.Irreps(str(o3.Irreps.spherical_harmonics(self.l_out))).sort().irreps.simplify()

        self.cg = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            shared_weights=True, normalization='component')

        self.fc = fc
        if fc:
            self.fc_x = nn.Linear(self.l1_bias, self.lout_bias)
            if right is None:
                self.fc_y = nn.Linear(self.l2_bias, self.lout_bias)

        self.right = right

    def forward(self, x, y=None):
        if self.right is not None:
            # edge_num, basis_num_x, channel = x.shape
            # out = torch.einsum('ji,bjc->bic', self.cg.right(self.right.to(x.device)), x)
            # if self.fc:
            #     x = x.transpose(1, 2).reshape(-1, basis_num_x)
            #     out += self.fc_x(x).reshape(edge_num, channel, -1).transpose(1, 2)
            #     # out += self.fc_y(y).reshape(edge_num, channel, -1)
            # return out
            edge_num, basis_num_x, channel = x.shape
            # out = torch.einsum('ji,bjc->bic', self.cg.right(self.right.to(x.device)), x)
            y = self.right.to(x.device).view(1, 1, -1).repeat(edge_num, channel, 1)
            x = x.transpose(1, 2)
            out = self.cg(x, y)
            if self.fc:
                x = x.reshape(-1, basis_num_x)
                out += self.fc_x(x).reshape(edge_num, channel, -1)
                # out += self.fc_y(y).reshape(edge_num, channel, -1)
            return out.transpose(1, 2)
        edge_num, channel, basis_num_x = x.shape
        basis_num_y = y.shape[-1]
        out = self.cg(x, y)
        x = x.reshape(-1, basis_num_x)
        y = y.reshape(-1, basis_num_y)
        if self.fc:
            out += self.fc_x(x).reshape(-1, channel, self.lout_bias)
            out += self.fc_y(y).reshape(-1, channel, self.lout_bias)
        return out.reshape(edge_num, channel, -1)

    def cg_inv(self, z):
        edge_num, channel, basis_num_z = z.shape
        cg_inv = self.cg.right(self.fixed_y.to(z.device)).transpose(0, 1)
        cg_inv = cg_inv.unsqueeze(0).repeat(edge_num, 1, 1)
        return z + torch.linalg.solve(cg_inv, torch.mean(z, dim=1)).unsqueeze(1)

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

class DistanceBlock2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis_functions,
        distance_expansion,
        max_num_elements,
        hidden_dim,
        act,
    ):
        super(DistanceBlock2, self).__init__()
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
        # self.target_embedding = nn.Embedding(
        #     self.max_num_elements, self.num_basis_functions
        # )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        # nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.fc1_edge_attr = nn.Linear(
            self.num_edge_channels,
            hidden_dim,
        )

    def forward(self, edge_distance, source_element):
        edge_distance = torch.norm(edge_distance, dim=1, p=2)
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)
        source_embedding = self.source_embedding(source_element)

        x_edge = self.act(source_embedding + x_dist)
        x_edge = self.act(self.fc1_edge_attr(x_edge))

        return x_edge
