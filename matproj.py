from mp_api.client import MPRester
import os

# Load the API key from the environment variable
api_key = os.getenv('MP_API_KEY')

def get_structure_from_mp(material_id):
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=["structure", "formula_pretty"]
        )
    doc = docs[0]
    return doc.structure, doc.formula_pretty
