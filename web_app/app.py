from flask import Flask, request, render_template, url_for
from Bio import motifs
from Bio.Seq import Seq
from Bio import SeqIO
import requests
import os
import logging
import uuid
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import time
import json
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)

CACHE_FILE = 'motifs_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_cache(data):
    with open(CACHE_FILE, 'w') as file:
        json.dump(data, file)

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), reraise=True)
def fetch_with_retry(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_jaspar_motifs(species, motif_name=None):
    cache = load_cache()
    if species in cache:
        if isinstance(cache[species], list):  # Check if the entry is a list
            cache[species] = {}  # Initialize as a dictionary
        if motif_name and motif_name in cache[species]:
            logging.info(f"Loading motifs from cache for species: {species} and motif: {motif_name}")
            return cache[species][motif_name]
        elif not motif_name:
            logging.info(f"Loading all motifs from cache for species: {species}")
            return cache[species]

    start_time = time.time()
    base_url = "https://jaspar.elixir.no/api/v1/matrix/"
    url = f"{base_url}?tax_group={species}&page_size=1000&release=2022&version=latest"
    motifs_list = []
    urls = []

    while url:
        response = fetch_with_retry(url)
        for motif in response['results']:
            if motif_name is None or motif['name'] == motif_name:
                urls.append(motif['url'])
                if motif_name and motif['name'] == motif_name:
                    break  # Exit early if we found the desired motif
        if url and not (motif_name and motif['name'] == motif_name):
            url = response.get('next')  # Get the URL for the next page of results
        else:
            url = None

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_with_retry, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                motif_detail_data = future.result()
                if 'pfm' in motif_detail_data:
                    pwm = {
                        'A': motif_detail_data['pfm']['A'],
                        'C': motif_detail_data['pfm']['C'],
                        'G': motif_detail_data['pfm']['G'],
                        'T': motif_detail_data['pfm']['T']
                    }
                    logging.info(f"Fetched PWM for motif {motif_detail_data['matrix_id']}: {pwm}")
                    motifs_list.append((motif_detail_data['matrix_id'], pwm))
                    if motif_name:
                        break  # Exit early if we found the desired motif
            except Exception as e:
                logging.error(f"Error fetching motif details: {e}")

    if species not in cache:
        cache[species] = {}
    if motif_name:
        cache[species][motif_name] = motifs_list
    else:
        cache[species] = motifs_list
    save_cache(cache)

    end_time = time.time()
    logging.info(f"Fetched motifs in {end_time - start_time:.2f} seconds")
    return motifs_list

def scan_sequence(sequence, known_motifs):
    start_time = time.time()
    matches = []
    logging.info(f"Sequence to scan: {sequence}")

    for motif_id, pwm in known_motifs:
        logging.info(f"Processing motif: {motif_id}")
        
        # Convert the PWM dictionary into a format accepted by Biopython
        counts = {base: pwm[base] for base in 'ACGT'}
        m = motifs.create(counts)
        
        # Create a Position-Specific Scoring Matrix (PSSM) from the motif
        pssm = m.pssm
        
        logging.info(f"PSSM for motif {motif_id}: {pssm}")
        
        # Search for the motif in the sequence without threshold
        for position, score in pssm.search(Seq(sequence), threshold=0.0):
            logging.info(f"Position: {position}, Score: {score}")
            matches.append((motif_id, position, score, m))
    
    end_time = time.time()
    logging.info(f"Scanned sequence in {end_time - start_time:.2f} seconds")
    return matches

def visualize_motif(pwm, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    pwm_df = pd.DataFrame(pwm)
    pwm_df.index.name = 'Position'
    logo = logomaker.Logo(pwm_df)
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved motif logo to {output_file}")

def process_sequence(sequence, species, motif_name=None):
    logging.info(f"Processing sequence for species: {species}")
    start_time = time.time()
    known_motifs = fetch_jaspar_motifs(species, motif_name)
    if not known_motifs:
        logging.error("No motifs found for the specified species.")
        return []

    logging.info("Scanning sequence for motifs...")
    matches = scan_sequence(sequence, known_motifs)
    logging.info(f"Found {len(matches)} motif matches.")

    logos = []
    for match in matches:
        motif_id, position, score, motif = match
        unique_filename = f'static/motif_logo_{motif_id}_{position}_{uuid.uuid4().hex}.png'
        visualize_motif(motif.counts, unique_filename)
        logos.append((motif_id, position, score, unique_filename))
        logging.info(f"Generated logo for motif {motif_id} at position {position}.")
    end_time = time.time()
    logging.info(f"Processed sequence in {end_time - start_time:.2f} seconds")
    return logos

def read_sequences_from_file(filepath):
    """
    Read DNA sequences from a file.

    Args:
        filepath (str): The path to the file containing the DNA sequences.

    Returns:
        list: A list of DNA sequences.
    """
    sequences = []
    file_format = filepath.split('.')[-1].lower()
    if file_format in ['fa', 'fasta', 'fna']:
        for record in SeqIO.parse(filepath, "fasta"):
            sequences.append(str(record.seq))
    elif file_format in ['fastq', 'fq']:
        for record in SeqIO.parse(filepath, "fastq"):
            sequences.append(str(record.seq))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return sequences

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        species = request.form.get('species', 'vertebrates')
        if not species:
            species = 'vertebrates'  # Default species

        if 'sequence' in request.form and request.form['sequence']:
            sequence = request.form['sequence']
        elif 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join('/tmp', file.filename)
            file.save(filepath)
            sequences = read_sequences_from_file(filepath)
            sequence = "".join(sequences)
        else:
            return "No sequence or file provided", 400

        logging.info("Starting motif processing...")
        logos = process_sequence(sequence, species, motif_name="ABF1")
        logging.info("Motif processing completed.")
        return render_template('result.html', logos=logos)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
