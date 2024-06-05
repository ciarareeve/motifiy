from flask import Flask, request, render_template, url_for
import os
import logging
import pandas as pd
import logomaker
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.DEBUG)

# Extend Jinja2 environment
app.jinja_env.globals.update(enumerate=enumerate)

# Degenerate base mapping
degenerate_base_map = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['C', 'G'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T'],
    'S': ['C', 'G'],
    'V': ['A', 'C', 'G']
}

# Dynamically set the MOTIF_DIR to the correct path based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIF_DIR = os.path.join(BASE_DIR, 'motifs')  # Adjust the path as needed

def load_motifs():
    logging.debug(f"Loading motifs from directory: {MOTIF_DIR}")
    motif_files = []
    if not os.path.exists(MOTIF_DIR):
        logging.error(f"Motif directory does not exist: {MOTIF_DIR}")
        return motif_files

    for root, _, files in os.walk(MOTIF_DIR):
        for file in files:
            if file.endswith('.motif'):
                full_path = os.path.join(root, file)
                logging.debug(f"Found motif file: {full_path}")
                motif_files.append(full_path)
    
    logging.debug(f"Total motif files loaded: {len(motif_files)}")
    return motif_files

def expand_degenerate_sequence(sequence):
    expanded_sequences = ['']
    for base in sequence:
        if base not in degenerate_base_map:
            logging.warning(f"Skipping unknown base: {base}")
            continue
        expanded_sequences = [prefix + suffix for prefix in expanded_sequences for suffix in degenerate_base_map[base]]
    return expanded_sequences

def parse_motif_file(filepath):
    logging.debug(f"Parsing motif file: {filepath}")
    with open(filepath) as f:
        lines = f.readlines()
        first_line = lines[0].strip()
        if first_line.startswith('>'):
            parts = first_line.split()
            sequence = parts[0][1:]  # Motif sequence identifier
            pwm = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) == 4:
                    pwm.append([float(p) for p in parts])
            return sequence, pwm
    return None, None

def generate_logo(pwm, motif_id):
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    logo = logomaker.Logo(pwm_df)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    logo.ax.set_ylabel('Probability')
    
    # Save the logo to a file
    filename = f"{motif_id}_{uuid.uuid4().hex}.png"
    logo_filename = os.path.join('static', 'motif_logos', filename)
    plt.savefig(logo_filename)
    plt.close()
    logging.debug(f"Logo saved to {logo_filename}")
    return logo_filename

def find_matching_motifs(user_sequence):
    logging.debug("Finding matching motifs")
    motif_files = load_motifs()
    matching_motifs = []
    expanded_user_sequences = set(expand_degenerate_sequence(user_sequence))
    logging.debug(f"Expanded user sequences: {expanded_user_sequences}")
    
    seen_motifs = set()  # To track already processed motifs
    
    for filepath in motif_files:
        sequence, pwm = parse_motif_file(filepath)
        if sequence:
            expanded_motif_sequences = set(expand_degenerate_sequence(sequence))
            logging.debug(f"Checking motif: {sequence}, Expanded: {expanded_motif_sequences}")
            
            if sequence in seen_motifs:
                continue
            
            # Check if any expanded motif sequence is a subset of any expanded user sequence or vice versa
            for user_seq in expanded_user_sequences:
                for motif_seq in expanded_motif_sequences:
                    if motif_seq in user_seq or user_seq in motif_seq:
                        logo_filename = generate_logo(pwm, os.path.basename(filepath))
                        matching_motifs.append((sequence, pwm, logo_filename))
                        logging.debug(f"Match found: {user_seq} in {motif_seq}")
                        seen_motifs.add(sequence)
                        break
                if any(motif_seq in user_seq or user_seq in motif_seq for motif_seq in expanded_motif_sequences):
                    break
    
    logging.debug(f"Total matching motifs found: {len(matching_motifs)}")
    return matching_motifs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'sequence' in request.form and request.form['sequence']:
            user_sequence = request.form['sequence']
            logging.info("Starting motif processing...")
            matching_motifs = find_matching_motifs(user_sequence)
            logging.info("Motif processing completed.")
            return render_template('result.html', matching_motifs=matching_motifs)
        else:
            return "No sequence provided", 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
