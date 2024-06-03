
import argparse
import logging
# from tqdm import tqdm
from web_app.app import app
from motif_finder.motif import find_motifs, load_known_motifs, compare_motifs, visualize_motif, read_sequences_from_file

ASCII_ART = """


░▒▓██████████████▓▒░ ░▒▓██████▓▒░▒▓████████▓▒░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓██████▓▒░  ░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     
                                                                               
                                                                               

"""

def run_web_app():
    """
    Runs the web application.
    """
    print(ASCII_ART)
    print("Running web app...")
    app.run(debug=True)

def generate_html_report(consensus_motif, motif_name, pwm, output_file):
    """
    Generates an HTML report with the given consensus motif, motif name, PWM, and saves it to the output file.

    Args:
        consensus_motif (str): The consensus motif.
        motif_name (str): The name of the motif.
        pwm (str): The Position Weight Matrix (PWM).
        output_file (str): The output file path.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Motify Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #282a36;
                color: #f8f8f2;
            }}
            .container {{
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #44475a;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                text-align: center;
                color: #bd93f9;
            }}
            p {{
                color: #f8f8f2;
            }}
            .logo {{
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px;
                color: #ffffff;
                background-color: #bd93f9;
                border-radius: 4px;
                text-decoration: none;
            }}
            a:hover {{
                background-color: #ff79c6;
            }}
            pre {{
                font-family: 'Courier New', Courier, monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motify Result</h1>
            <pre>{ASCII_ART}</pre>
            <p><strong>Consensus Motif:</strong> {consensus_motif}</p>
            <p><strong>Motif Name:</strong> {motif_name}</p>
            <p><strong>Position Weight Matrix (PWM):</strong></p>
            <pre>{pwm}</pre>
            <div class="logo">
                <img src="motif_logo.png" alt="Motif Logo">
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_file, 'w') as file:
        file.write(html_content)

def run_cli(sequence=None, file=None, output_file=None, verbose=False):
    """
    Runs the command-line interface (CLI) with the given sequence or file.

    Args:
        sequence (str, optional): The DNA sequence to find motifs. Defaults to None.
        file (str, optional): The file containing DNA sequences to find motifs. Defaults to None.
        output_file (str, optional): The output file to write results in HTML format. Defaults to None.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
    """
    print(ASCII_ART)
    print(f"Running CLI with sequence: {sequence} or file: {file}")

    if verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        logging.info("Loading known motifs...")
        known_motifs = load_known_motifs()

        if file:
            logging.info(f"Reading sequences from file: {file}")
            sequences = read_sequences_from_file(file)
        else:
            sequences = [sequence]

        logging.info("Finding motifs in the sequences...")
        consensus_motif, motif_object = find_motifs(sequences)
        motif_name = compare_motifs(consensus_motif, known_motifs)
        
        pwm = motif_object.counts
        result = f"Consensus Motif: {consensus_motif}\nMotif Name: {motif_name}\n\nPWM:\n{pwm}"
        print(result)
        
        # Visualize the motif and save the logo
        logging.info("Visualizing the motif and saving the logo...")
        visualize_motif(pwm, 'motif_logo.png')
        
        # Generate the HTML report
        if output_file:
            logging.info(f"Generating HTML report: {output_file}")
            generate_html_report(consensus_motif, motif_name, pwm, output_file)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if verbose:
            logging.exception("Exception occurred")

def generate_static_html_report(output_file):
    import json
    # Static motif data for ABF1
    motif_data = {
        "matrix_id": "MA0265.1",
        "name": "ABF1",
        "pfm": {
            "A": [11.0, 0.0, 0.0, 0.0, 41.0, 22.0, 39.0, 34.0, 40.0, 41.0, 12.0, 0.0, 92.0, 0.0, 37.0, 34.0],
            "C": [41.0, 99.0, 0.0, 0.0, 17.0, 24.0, 16.0, 18.0, 21.0, 0.0, 25.0, 0.0, 8.0, 45.0, 22.0, 18.0],
            "G": [4.0, 0.0, 99.0, 0.0, 22.0, 26.0, 28.0, 20.0, 24.0, 57.0, 9.0, 81.0, 0.0, 9.0, 18.0, 14.0],
            "T": [44.0, 1.0, 0.0, 99.0, 20.0, 28.0, 17.0, 28.0, 15.0, 0.0, 55.0, 19.0, 0.0, 46.0, 23.0, 34.0]
        },
        "sequence_logo": "https://jaspar.elixir.no/static/logos/svg/MA0265.1.svg"
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Motify Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #282a36;
                color: #f8f8f2;
            }}
            .container {{
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #44475a;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                text-align: center;
                color: #bd93f9;
            }}
            p {{
                color: #f8f8f2;
            }}
            .logo {{
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px;
                color: #ffffff;
                background-color: #bd93f9;
                border-radius: 4px;
                text-decoration: none;
            }}
            a:hover {{
                background-color: #ff79c6;
            }}
            pre {{
                font-family: 'Courier New', Courier, monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motify Result</h1>
            <p><strong>Motif Name:</strong> {motif_data['name']}</p>
            <p><strong>Position Weight Matrix (PWM):</strong></p>
            <pre>{json.dumps(motif_data['pfm'], indent=4)}</pre>
            <div class="logo">
                <img src="{motif_data['sequence_logo']}" alt="Motif Logo">
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_file, 'w') as file:
        file.write(html_content)

    print(f"Static report generated: {output_file}")

def run_static_example(output_file):
    generate_static_html_report(output_file)

def main():
    parser = argparse.ArgumentParser(description='Motif Finder Tool')
    parser.add_argument('--web', action='store_true', help='Run the web application')
    parser.add_argument('--sequence', type=str, help='DNA sequence to find motifs')
    parser.add_argument('--file', type=str, help='File containing DNA sequences to find motifs')
    parser.add_argument('--out', type=str, help='Output file to write results in HTML format')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    parser.add_argument('--benchmark-test', action='store_true', help='Generate a static example report')

    args = parser.parse_args()

    if args.web:
        run_web_app()
    elif args.benchmark_test:
        run_static_example(args.out)
    elif args.sequence or args.file:
        run_cli(args.sequence, args.file, args.out, args.verbose)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()