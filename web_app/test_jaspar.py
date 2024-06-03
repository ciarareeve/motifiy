import requests

def fetch_jaspar_motifs(species):
    base_url = "https://jaspar.elixir.no/api/v1/matrix/"
    params = {
        "tax_group": species,
        "page_size": 100,  # Adjust page size if needed
        "release": "2022",
        "version": "latest"
    }

    all_motifs = []
    while base_url:
        print(f"Requesting URL: {base_url}")
        response = requests.get(base_url, params=params)
        print(f"Response status code: {response.status_code}")
        data = response.json()
        all_motifs.extend(data['results'])
        base_url = data['next']  # Get the next page URL
        params = {}  # Clear params after the first request to avoid redundancy
        print(f"Processing {len(data['results'])} motifs")

    return all_motifs

def test_fetch_jaspar_motifs_for_single_species():
    print("Starting test script")
    species = "vertebrates"
    print(f"Testing JASPAR API call for species: {species}")
    motifs_list = fetch_jaspar_motifs(species)
    print(f"Total motifs fetched: {len(motifs_list)}")

test_fetch_jaspar_motifs_for_single_species()