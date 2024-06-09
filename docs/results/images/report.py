import os

# Paths
images_dir = '../results/images'
attributions_images = ""
cluster_images = ""

# Original sequence numbers
og_seq = [41906, 7297, 1640, 48599, 18025, 16050, 14629, 9145, 48266, 6718, 44349, 48541, 35742, 5698, 38699, 27652, 2083, 1953, 6141, 14329, 15248, 33119, 39454, 1740, 36782]

# Generate HTML for attributions images
for seq in og_seq:
    attributions_images += f'''
    <div class="logo-container">
        <h3>Sequence {seq}</h3>
        <img src="attributions_sequence_{seq}.png" alt="Attributions for Sequence {seq}">
    </div>
    '''

# Number of clusters (manually determined)
num_clusters = 3

# Generate HTML for cluster images
for cluster in range(num_clusters):
    cluster_images += f'''
    <div class="logo-container">
        <h3>Cluster {cluster}</h3>
        <img src="sequence_logo_cluster_{cluster}.png" alt="Sequence Logo for Cluster {cluster}">
    </div>
    '''

# Read the template HTML file
template_file_path = os.path.join(os.path.dirname(__file__), 'motif_report.html')
with open(template_file_path, 'r') as file:
    template_html = file.read()

# Replace placeholders with generated HTML
output_html = template_html.replace("{{ attributions_images }}", attributions_images).replace("{{ cluster_images }}", cluster_images)

# Write the final HTML to a new file
output_file_path = os.path.join(os.path.dirname(__file__), 'motify_result_output.html')
with open(output_file_path, 'w') as file:
    file.write(output_html)

print("HTML file generated successfully.")
