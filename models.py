import flax
import flax.linen
import jax.numpy as jnp
import jraph

import e3nn_jax as e3nn
from nequip_jax import NEQUIPLayerFlax, filter_layers  # e3nn implementation of NEQUIP

from atom_graphs import get_relative_vectors

class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs, cutoff=2.0):
        num_nodes = graphs.nodes["positions"].shape[0]
        senders = graphs.senders
        receivers = graphs.receivers
        species = graphs.nodes["species"]

        vectors = get_relative_vectors(
            senders,
            receivers,
            graphs.n_edge,
            positions=graphs.nodes["positions"],
            cells=graphs.globals["cells"],
            shifts=graphs.edges["shifts"],
        )

        # We divide the relative vectors by the cutoff
        # because NEQUIPLayerFlax assumes a cutoff of 1.0
        vectors = vectors / cutoff

        # Embedding: since we have a single atom type, we don't need embedding
        # The node features are just ones and the species indices are all zeros
        node_feats = flax.linen.Embed(num_embeddings=5, features=32)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        # Apply 3 Nequip layers with different internal representations
        for irreps in [
            "32x0e + 8x1o + 8x2e",
            "32x0e + 8x1o + 8x2e",
            "32x0e",
        ]: 
            layer = NEQUIPLayerFlax(
                avg_num_neighbors=20.0,  # average number of neighbors to normalize by
                output_irreps=irreps,
            )
            node_feats = layer(vectors, node_feats, species, senders, receivers)

        # Self-Interaction layers
        node_feats = e3nn.flax.Linear("16x0e")(node_feats)
        node_feats = e3nn.flax.Linear("2e")(node_feats)

        return node_feats
    

class NEQUIP(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        positions = graph.nodes["positions"]
        species = graph.nodes["species"]

        vectors = e3nn.IrrepsArray(
            "1o", positions[graph.receivers] - positions[graph.senders]
        )
        node_feats = flax.linen.Embed(num_embeddings=5, features=32)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        layers_irreps = ["16x0e + 16x1o + 16x1e"] * 2 + ["32x0e"]
        layers_irreps = filter_layers(layers_irreps, max_ell=3)
        for irreps in layers_irreps:
            layer = NEQUIPLayerFlax(
                avg_num_neighbors=1.0,
                output_irreps=irreps,
                max_ell=3,
            )
            node_feats = layer(
                vectors,
                node_feats,
                species,
                graph.senders,
                graph.receivers,
            )

        return node_feats

    # graph = dummy_graph()

    # model = NEQUIP()
    # w = model.init(jax.random.PRNGKey(0), graph)

    # apply = jax.jit(model.apply)
    # apply(w, graph)
    # apply(w, graph)

class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs, cutoff=2.0):
        num_nodes = graphs.nodes["positions"].shape[0]
        senders = graphs.senders
        receivers = graphs.receivers
        species = graphs.nodes["species"]

        vectors = get_relative_vectors(
            senders,
            receivers,
            graphs.n_edge,
            positions=graphs.nodes["positions"],
            cells=graphs.globals["cells"],
            shifts=graphs.edges["shifts"],
        )

        # We divide the relative vectors by the cutoff
        # because NEQUIPLayerFlax assumes a cutoff of 1.0
        vectors = vectors / cutoff

        # Embedding: since we have a single atom type, we don't need embedding
        # The node features are just ones and the species indices are all zeros
        node_feats = flax.linen.Embed(num_embeddings=5, features=32)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        # Apply 3 Nequip layers with different internal representations
        for irreps in [
            "64x0e + 64x0o + 48x1o + 48x1e +36x2o + 36x2e + 24x3o + 24x3e",
            "64x0e + 64x0o + 48x1o + 48x1e +36x2o + 36x2e + 24x3o + 24x3e",
            "64x0e + 64x0o + 48x1o + 48x1e +36x2o + 36x2e + 24x3o + 24x3e",
        ]: 
            layer = NEQUIPLayerFlax(
                avg_num_neighbors=20.0,  # average number of neighbors to normalize by
                output_irreps=irreps,
            )
            node_feats = layer(vectors, node_feats, species, senders, receivers)

        # Self-Interaction layers
        # node_feats = e3nn.flax.Linear("16x0e + 8x1o + 8x2e")(node_feats)
        # node_feats = e3nn.flax.Linear("1x2e")(node_feats)

        node_feats = jnp.mean(node_feats, axis=0)

        hyperpol_tensor_matrix = e3nn.reduced_tensor_product_basis("ijk=kij=kji", i="1x1o")

        hyperpol_tensor_L1 = e3nn.flax.Linear(irreps_out = hyperpol_tensor_matrix.irreps[0])(node_feats).array
        hyperpol_tensor_L3 = e3nn.flax.Linear(irreps_out = hyperpol_tensor_matrix.irreps[1])(node_feats).array

        hyperpol_tensor_fromL1 = jnp.einsum('ijks,as->aijk', hyperpol_tensor_matrix.chunks[0].squeeze(),hyperpol_tensor_L1)
        hyperpol_tensor_fromL3 = jnp.einsum('ijks,as->aijk', hyperpol_tensor_matrix.chunks[1].squeeze(),hyperpol_tensor_L3)

        hyperpol_tensor = hyperpol_tensor_fromL1 + hyperpol_tensor_fromL3
        
        return hyperpol_tensor