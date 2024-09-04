import requests
import json
import time
import os

from tqdm import tqdm
import jax.numpy as jnp

from mp_api.client import MPRester
from emmet.core.summary import HasProps

api_key = os.getenv('MP_API_KEY')

def get_materials_with_dielectric_data():
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(has_props = [HasProps.dielectric], fields=["material_id", "formula_pretty"])
    return docs

def get_structures_and_dielectric_tensors(material_ids, download=True, truncated=False):
    with MPRester(api_key) as mpr:
        # Get structures for all material_ids
        docs = mpr.materials.summary.search(
            material_ids=material_ids,
            fields=["material_id", "structure"]
        )

    materials_data = []

    for doc in tqdm(docs):
        material_id = doc.material_id
        structure = doc.structure

        # Extract atom coordinates, atomic species, and lattice parameters as jax arrays
        coords = jnp.array([site.coords for site in structure])
        species = jnp.array([site.specie.number for site in structure])
        lattice = jnp.array(structure.lattice.matrix)

        # Get dielectric tensor from URL
        url = f"https://legacy.materialsproject.org/phonons/{material_id}/dielectric_tensor?download=true"
        response = requests.get(url)

        if response.status_code == 200:
            dielectric_data = json.loads(response.text)
            
            # Convert dielectric tensors to jax arrays
            try:
                epsilon_electronic = jnp.array(dielectric_data['epsilon_electronic'])
                epsilon_ionic = jnp.array(dielectric_data['epsilon_ionic'])
            except KeyError:
                print(f"No dielectric data found for {material_id}. Skipping...")
                continue

            # Combine all data in a dictionary
            material_data = {
                'material_id': material_id,
                'coordinates': coords,
                'species': species,
                'lattice': lattice,
                'epsilon_electronic': epsilon_electronic,
                'epsilon_ionic': epsilon_ionic
            }

            materials_data.append(material_data)

            if truncated:
                if len(materials_data) == 10:
                    return materials_data

            if download:
                if not os.path.exists('data'):
                    os.makedirs('data')
                # Create the filename using the material_id
                filename = f"data/{material_id}_data.json"

                # Save the data for this material to a separate JSON file
                with open(filename, 'w') as f:
                    json.dump({k: v.tolist() if isinstance(v, jnp.ndarray) else v 
                               for k, v in material_data.items()}, f, indent=2)

                print(f"Downloaded and saved data for {material_id}")
        else:
            print(f"Error downloading dielectric data for {material_id}: {response.status_code}")

    return materials_data
