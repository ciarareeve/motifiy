import os
import pandas as pd
import logomaker
import matplotlib.pyplot as plt

# Define the PWM for the test
pwm = {
    "A": [11.0, 0.0, 0.0, 0.0, 41.0, 22.0, 39.0, 34.0, 40.0, 41.0, 12.0, 0.0, 92.0, 0.0, 37.0, 34.0],
    "C": [41.0, 99.0, 0.0, 0.0, 17.0, 24.0, 16.0, 18.0, 21.0, 0.0, 25.0, 0.0, 8.0, 45.0, 22.0, 18.0],
    "G": [4.0, 0.0, 99.0, 0.0, 22.0, 26.0, 28.0, 20.0, 24.0, 57.0, 9.0, 81.0, 0.0, 9.0, 18.0, 14.0],
    "T": [44.0, 1.0, 0.0, 99.0, 20.0, 28.0, 17.0, 28.0, 15.0, 0.0, 55.0, 19.0, 0.0, 46.0, 23.0, 34.0]
}

# Create the output directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Generate the logo
pwm_df = pd.DataFrame(pwm)
pwm_df.index.name = "Position"
logo = logomaker.Logo(pwm_df)
plt.savefig("static/test_logo.png")
plt.close()

# Create the HTML content
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Motify Result</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
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
        p, pre {{
            color: #f8f8f2;
        }}
        .logo {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        .logo img {{
            max-width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Motify Result</h1>
        <p><strong>Consensus Motif:</strong> ABF1</p>
        <p><strong>Motif Name:</strong> MA0265.1</p>
        <p><strong>Position Weight Matrix (PWM):</strong></p>
        <pre>{pwm}</pre>
        <div class="logo">
            <img src="static/test_logo.png" alt="Motif Logo">
        </div>
    </div>
</body>
</html>
""".format(pwm=pwm)

# Write the HTML content to a file
with open("static/static_result_test.html", "w") as file:
    file.write(html_content)
