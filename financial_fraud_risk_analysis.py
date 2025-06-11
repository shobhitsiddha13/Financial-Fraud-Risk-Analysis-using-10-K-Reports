
# Financial Fraud Risk Analysis using 10-K Reports

import requests
from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

nltk.download('punkt')

# Step 1: Fetch 10-K filing (Example: Apple's 2022 10-K)
url = 'https://www.sec.gov/Archives/edgar/data/320193/000032019323000010/a10-k2022.htm'
response = requests.get(url)
html_content = response.text

# Step 2: Extract and clean text
soup = BeautifulSoup(html_content, 'html.parser')
text = soup.get_text()
cleaned_text = re.sub(r'\s+', ' ', text)

# Step 3: Tokenization
tokens = nltk.word_tokenize(cleaned_text)

# Step 4: Extract financial metrics using regex
revenue = re.findall(r'Revenue[^\$]*\$?([\d,]+)', cleaned_text)
debt = re.findall(r'Debt[^\$]*\$?([\d,]+)', cleaned_text)

print("Extracted Revenue:", revenue[:3])
print("Extracted Debt:", debt[:3])

# Sample structured data (mocked for analysis)
data = {
    'Year': [2020, 2021, 2022],
    'Revenue': [265000, 294000, 365000],
    'Net Income': [57000, 63000, 94000],
    'Debt': [100000, 95000, 130000]
}

df = pd.DataFrame(data)

# Step 5: Calculate financial ratios
df['YoY Revenue Growth'] = df['Revenue'].pct_change()
df['Debt-to-Revenue'] = df['Debt'] / df['Revenue']

# Step 6: Anomaly Detection
model = IsolationForest(contamination=0.1)
df['Anomaly'] = model.fit_predict(df[['Revenue', 'Net Income', 'Debt']])

# Step 7: Visualization
plt.figure(figsize=(10,5))
plt.plot(df['Year'], df['Revenue'], label='Revenue')
plt.plot(df['Year'], df['Net Income'], label='Net Income')
plt.plot(df['Year'], df['Debt'], label='Debt')
plt.title('Financial Metrics Over Years')
plt.xlabel('Year')
plt.ylabel('Amount (in Millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/mnt/data/financial_trend_plot.png')
plt.show()

# Save DataFrame to CSV
df.to_csv('/mnt/data/financial_fraud_analysis.csv', index=False)

print("Analysis complete. Plot and CSV saved.")
