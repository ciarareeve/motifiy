import argparse
from web_app.app import app
from motif_finder.motif import find_motifs, load_known_motifs, compare_motifs, visualize_motif, read_sequences_from_file

def run_web_app():
    """
    Runs the web application.
    """
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motify Result</h1>
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

def run_cli(sequence=None, file=None, output_file=None):
    """
    Runs the command-line interface (CLI) with the given sequence or file.

    Args:
        sequence (str, optional): The DNA sequence to find motifs. Defaults to None.
        file (str, optional): The file containing DNA sequences to find motifs. Defaults to None.
        output_file (str, optional): The output file to write results in HTML format. Defaults to None.
    """
    print(f"Running CLI with sequence: {sequence} or file: {file}")
    known_motifs = load_known_motifs()
    
    if file:
        sequences = read_sequences_from_file(file)
    else:
        sequences = [sequence]

    consensus_motif, motif_object = find_motifs(sequences)
    motif_name = compare_motifs(consensus_motif, known_motifs)
    
    pwm = motif_object.counts
    result = f"Consensus Motif: {consensus_motif}\nMotif Name: {motif_name}\n\nPWM:\n{pwm}"
    print(result)
    
    # Visualize the motif and save the logo
    visualize_motif(pwm, 'motif_logo.png')
    
    # Generate the HTML report
    if output_file:
        generate_html_report(consensus_motif, motif_name, pwm, output_file)

def main():
    """
    The main function that parses command-line arguments and runs the appropriate function.
    """
    parser = argparse.ArgumentParser(description='Motif Finder Tool')
    parser.add_argument('--web', action='store_true', help='Run the web application')
    parser.add_argument('--sequence', type=str, help='DNA sequence to find motifs')
    parser.add_argument('--file', type=str, help='File containing DNA sequences to find motifs')
    parser.add_argument('--out', type=str, help='Output file to write results (HTML format)')

    args = parser.parse_args()

    if args.web:
        run_web_app()
    elif args.sequence or args.file:
        run_cli(args.sequence, args.file, args.out)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
