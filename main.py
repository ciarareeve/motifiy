import argparse
import logging
import os
from app import app, find_matching_motifs, generate_logo

ASCII_ART = """
                                                                                                                                                 
"""

def read_sequences_from_file(file_path):
    """
    Reads sequences from a given file.

    Args:
        file_path (str): The path to the file containing sequences.

    Returns:
        list: A list of tuples containing the sequence name and the sequence.
    """
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sequence_name = ""
        sequence = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append((sequence_name, sequence))
                sequence_name = line[1:]  # Remove the '>' character
                sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append((sequence_name, sequence))
    return sequences

def run_web_app():
    """
    Runs the web application.
    """
    print(ASCII_ART)
    print("Running web app...")
    app.run(debug=True)

def generate_html_report(matching_motifs, output_file):
    """
    Generates an HTML report with the matching motifs and saves it to the output file.

    Args:
        matching_motifs (list): List of tuples containing motif sequence, PWM, and logo filename.
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
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 10px;
                border: 1px solid #ddd;
                text-align: center;
            }}
            th {{
                background-color: #bd93f9;
                color: #f8f8f2;
            }}
            tr:nth-child(even) {{
                background-color: #44475a;
            }}
            .logo {{
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }}
            .logo img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motify Result</h1>
            <pre>{ASCII_ART}</pre>
    """
    for sequence, pwm, logo_filename in matching_motifs:
        html_content += f"""
        <div class="motif">
            <h2>Motif: {sequence}</h2>
            <div class="logo">
                <img src="{logo_filename}" alt="Motif Logo">
            </div>
            <table>
                <tr>
                    <th>Position</th>
                    <th>A</th>
                    <th>C</th>
                    <th>G</th>
                    <th>T</th>
                </tr>
        """
        for idx, row in enumerate(pwm):
            html_content += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                </tr>
            """
        html_content += """
            </table>
        </div>
        """
    html_content += """
        </div>
    </body>
    </html>
    """
    with open(output_file, 'w') as file:
        file.write(html_content)

def run_cli(sequence, file, output_file, verbose=False):
    """
    Runs the command-line interface (CLI) with the given sequence or file.

    Args:
        sequence (str): The DNA sequence to find motifs.
        file (str): The file containing DNA sequences to find motifs.
        output_file (str): The output file to write results (HTML format).
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
    """
    print(ASCII_ART)
    if sequence:
        print(f"Running CLI with sequence: {sequence}")
    elif file:
        print(f"Running CLI with file: {file}")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        matching_motifs = []
        if file:
            sequences = read_sequences_from_file(file)
            for name, seq in sequences:
                matching_motifs.extend(find_matching_motifs(seq))
        else:
            matching_motifs = find_matching_motifs(sequence)
        
        # Remove duplicates
        unique_matching_motifs = []
        seen_sequences = set()
        for motif in matching_motifs:
            if motif[0] not in seen_sequences:
                unique_matching_motifs.append(motif)
                seen_sequences.add(motif[0])
        
        print(f"Found {len(unique_matching_motifs)} unique matching motifs")

        if output_file:
            generate_html_report(unique_matching_motifs, output_file)
            logging.debug(f"HTML report generated: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        logging.exception("Exception occurred")

def main():
    """
    The main function that parses command-line arguments and runs the appropriate function.
    """
    parser = argparse.ArgumentParser(description='Motif Finder Tool')
    parser.add_argument('--web', action='store_true', help='Run the web application')
    parser.add_argument('--sequence', type=str, help='DNA sequence to find motifs')
    parser.add_argument('--file', type=str, help='File containing DNA sequences to find motifs')
    parser.add_argument('--out', type=str, help='Output file to write results (HTML format)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')

    args = parser.parse_args()

    if args.web:
        run_web_app()
    elif args.sequence or args.file:
        run_cli(args.sequence, args.file, args.out, args.verbose)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()