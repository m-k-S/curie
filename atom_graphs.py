import os

import jax.numpy as jnp
import jraph
import e3nn_jax as e3nn
from e3nn_jax import radius_graph
from mp_api.client import MPRester
from matscipy.neighbours import neighbour_list  # fast neighbour list implementation

# Load the API key from the environment variable
api_key = os.getenv('MP_API_KEY')

def get_structure_from_mp(material_ids):
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=material_ids,
            fields=["structure", "formula_pretty"]
        )
    return docs

# TODO: Decide between this graph structure and the other one
def structure_to_graph(structure, cutoff_radius=5.0) -> jraph.GraphsTuple:
    # Extract atomic positions and atomic numbers
    pos = jnp.array([site.coords for site in structure], dtype=jnp.float32)
    atomic_numbers = jnp.array([site.specie.Z for site in structure], dtype=jnp.int32)

    # Create the graph
    senders, receivers = radius_graph(pos, cutoff_radius)

    return jraph.GraphsTuple(
        nodes=jnp.column_stack([pos, atomic_numbers.reshape(-1, 1)]),  # [num_nodes, 4] (x, y, z, Z)
        edges=None,
        globals=jnp.array([len(structure)], dtype=jnp.int32),  # number of atoms as a global feature
        senders=senders,  # [num_edges]
        receivers=receivers,  # [num_edges]
        n_node=jnp.array([len(structure)]),  # [num_graphs]
        n_edge=jnp.array([len(senders)]),  # [num_graphs]
    )

def compute_edges(positions, cell, cutoff):
    """Compute edges of the graph from positions and cell."""
    # cutoff in Angstrom
    receivers, senders, senders_unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=jnp.array([True, True, True]),
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )

    num_edges = senders.shape[0]
    assert senders.shape == (num_edges,)
    assert receivers.shape == (num_edges,)
    assert senders_unit_shifts.shape == (num_edges, 3)
    return senders, receivers, senders_unit_shifts

def encode_species(species):
    """Encode atomic species as integers."""
    species = jnp.array(species)
    unique_species = jnp.unique(species)
    encoded_species = jnp.searchsorted(unique_species, species)
    return encoded_species

def create_graph(positions, cell, dielectric, species, cutoff=8.0):
    """Create a graph from positions, cell, and dielectric tensor."""

    # species is a list of atomic numbers

    senders, receivers, senders_unit_shifts = compute_edges(positions, cell, cutoff)

    # In a jraph.GraphsTuple object, nodes, edges, and globals can be any
    # pytree. We will use dicts of arrays.
    # What matters is that the first axis of each array has length equal to
    # the number of nodes, edges, or graphs.
    num_nodes = positions.shape[0]
    num_edges = senders.shape[0]

    species = encode_species(species)

    graph = jraph.GraphsTuple(
        # positions are per-node features:
        nodes=dict(positions=positions, species=species),
        # Unit shifts are per-edge features:
        edges=dict(shifts=senders_unit_shifts),
        # energy and cell are per-graph features:
        globals=dict(dielectric=dielectric, cells=cell[None, :, :]),
        # The rest of the fields describe the connectivity and size of the graph.
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([num_edges]),
    )
    return graph

def get_relative_vectors(senders, receivers, n_edge, positions, cells, shifts):
    """Compute the relative vectors between the senders and receivers."""
    num_nodes = positions.shape[0]
    num_edges = senders.shape[0]
    num_graphs = n_edge.shape[0]

    assert positions.shape == (num_nodes, 3)
    assert cells.shape == (num_graphs, 3, 3)
    assert senders.shape == (num_edges,)
    assert receivers.shape == (num_edges,)
    assert shifts.shape == (num_edges, 3)

    # We need to repeat the cells for each edge.
    cells = jnp.repeat(cells, n_edge, axis=0, total_repeat_length=num_edges)

    # Compute the two ends of each edge.
    positions_receivers = positions[receivers]
    positions_senders = positions[senders] + jnp.einsum("ei,eij->ej", shifts, cells)

    vectors = e3nn.IrrepsArray("1o", positions_receivers - positions_senders)
    return vectors

