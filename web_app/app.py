from flask import Flask, request, render_template
from motif_finder.motif import find_motifs, load_known_motifs, compare_motifs, visualize_motif, read_sequences_from_file
import os

app = Flask(__name__)
known_motifs = load_known_motifs()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle the index route of the web application.

    If the request method is POST, it checks if a sequence is provided in the form data.
    If a sequence is provided, it finds motifs in the sequence and renders the result template.
    If a file is provided, it reads the sequences from the file, finds motifs, and renders the result template.
    If neither a sequence nor a file is provided, it returns an error message.

    Returns:
        If successful, it renders the result template with the motifs, function name, and logo URL.
        If unsuccessful, it returns an error message with a status code of 400.

    """
    if request.method == 'POST':
        if 'sequence' in request.form and request.form['sequence']:
            sequence = request.form['sequence']
            sequences = [sequence]
        elif 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join('/tmp', file.filename)
            file.save(filepath)
            sequences = read_sequences_from_file(filepath)
        else:
            return "No sequence or file provided", 400

        consensus_motif, motif_object = find_motifs(sequences)
        motif_name = compare_motifs(consensus_motif, known_motifs)
        
        # Create and save the motif logo
        visualize_motif(motif_object.counts, 'static/motif_logo.png')
        
        return render_template('result.html', motifs=consensus_motif, function=motif_name, logo_url='static/motif_logo.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
