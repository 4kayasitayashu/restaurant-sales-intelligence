\# 🍽️ Restaurant Sales Intelligence \& Demand Forecasting



An end-to-end \*\*Data Science and Machine Learning project\*\* for analyzing restaurant sales, understanding business performance, and forecasting future demand.



🔗 \*\*Live Demo:\*\* https://restaurant-sales-intelligence-13.streamlit.app/



\---



\## 📌 Project Overview



This project analyzes restaurant transaction data to answer questions like:



\* How much revenue is being generated?

\* Which products and categories are performing well?

\* Which outlet performs better?

\* When is customer demand highest?

\* How many orders can be expected in the future?

\* What business actions can be taken based on the analysis?



The project combines \*\*Data Analytics, Machine Learning, Demand Forecasting and an Interactive Streamlit Dashboard\*\* into one system.



\---



\## ✨ Main Features



\* 📊 Sales \& Revenue Analysis

\* 🛍️ Product Performance Analysis

\* 📂 Category Analysis

\* 🏪 Outlet Performance Analysis

\* ⏰ Day \& Hour Demand Analysis

\* 📈 Demand Forecasting

\* 🤖 Machine Learning Model Comparison

\* 🔮 14-Day Future Demand Forecast

\* 💡 Business Insights \& Operational Signals

\* 📊 Interactive Streamlit Dashboard

\* ☁️ Streamlit Cloud Deployment



\---



\## 🧠 Machine Learning



The project uses historical daily order data to predict future restaurant demand.



Different models were compared:



\* Linear Regression

\* Random Forest

\* HistGradientBoosting

\* Weekly Naive Baseline



The final selected model was \*\*HistGradientBoosting\*\*, based on the lowest error on the chronological test period.



\### Model Performance



| Model                 |   MAE |  RMSE |   WAPE |

| --------------------- | ----: | ----: | -----: |

| HistGradientBoosting  |  6.45 |  8.31 | 10.31% |

| Random Forest         |  6.72 |  8.52 | 10.74% |

| Weekly Naive Baseline |  8.23 | 11.02 | 13.16% |

| Linear Regression     | 10.80 | 13.06 | 17.27% |



The test period used was \*\*2025-08-08 to 2025-09-30\*\*.



\---



\## 🔮 Future Forecast



The trained model is used to generate a \*\*14-day demand forecast\*\* for each restaurant outlet.



The forecast can help understand expected order volume and support operational planning such as staffing and capacity planning.



\---



\## 📊 Dashboard



The Streamlit dashboard contains multiple sections:



\### Executive Overview



Shows overall restaurant performance including:



\* Revenue

\* Orders

\* Average Order Value

\* Total Items

\* Monthly performance

\* Outlet performance

\* Forecast snapshot



\### Sales Analytics



Provides:



\* Day-of-week analysis

\* Hourly demand

\* Payment method analysis

\* Time-period analysis



\### Product Intelligence



Shows:



\* Top-selling products

\* Product performance

\* Category performance

\* Revenue contribution



\### Outlet Intelligence



Compares restaurant outlets based on:



\* Revenue

\* Orders

\* Average Order Value



\### Demand Forecast



Displays:



\* Future demand by outlet

\* Forecast trends

\* Forecast KPIs

\* ML model performance



\### Business Insights



Converts analytical results into simple operational signals and recommendations.



\---



\# 📁 Project Structure



```text

restaurant-sales-intelligence/

│

├── dashboard/

│   └── app.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── restaurant\_demand\_model.joblib

│

├── reports/

│   ├── product\_performance.csv

│   ├── category\_performance.csv

│   ├── future\_demand\_forecast.csv

│   ├── forecast\_model\_comparison.csv

│   └── business\_insights.csv

│

├── src/

│   ├── analysis/

│   └── ml\_models/

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\# 💻 How to Run Locally



\## 1. Clone the Repository



```bash

git clone https://github.com/4kayasitayashu/restaurant-sales-intelligence.git

cd restaurant-sales-intelligence

```



\## 2. Create Virtual Environment



\### Windows



```bash

python -m venv .venv

.venv\\Scripts\\activate

```



\## 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\## 4. Run the Dashboard



```bash

streamlit run dashboard/app.py

```



The dashboard will normally open at:



```text

http://localhost:8501

```



\---



\# ☁️ How to Deploy on Your Own Streamlit Account



If you want to deploy your own copy of this project:



\## Step 1 — Create a GitHub Repository



Create a new repository on your GitHub account.



Example:



```text

restaurant-sales-intelligence

```



\## Step 2 — Upload the Project



Inside the project folder:



```bash

git init

git add .

git commit -m "Initial project upload"

```



Connect your GitHub repository:



```bash

git remote add origin https://github.com/YOUR\_USERNAME/restaurant-sales-intelligence.git

git branch -M main

git push -u origin main

```



Replace `YOUR\_USERNAME` with your GitHub username.



\## Step 3 — Open Streamlit Community Cloud



Go to:



https://share.streamlit.io/



Sign in using your GitHub account.



\## Step 4 — Create a New App



Select:



```text

Repository: YOUR\_USERNAME/restaurant-sales-intelligence

Branch: main

Main file path: dashboard/app.py

```



Then click \*\*Deploy\*\*.



Streamlit will automatically install the dependencies listed in:



```text

requirements.txt

```



and deploy the application.



\---



\# 🔄 Updating the Deployed Application



After making changes to the project:



```bash

git add .

git commit -m "Update dashboard"

git push

```



Streamlit Cloud will detect the new GitHub commit and update the deployed application.



\---



\# ⚠️ Dataset Note



The project uses a real-world F\&B transaction dataset.



The original raw dataset is kept outside the GitHub repository and is excluded through `.gitignore`.



The dashboard works using the processed datasets and generated reports stored inside:



```text

data/processed/

reports/

```



Therefore, the deployed dashboard does not need the original raw CSV files to run.



\---



\# 🛠️ Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Plotly

\* Streamlit

\* Git \& GitHub

\* Machine Learning

\* Time-Series / Demand Forecasting



\---



\# 🎯 Project Goal



The main goal of this project is to convert historical restaurant sales data into \*\*useful business insights and future demand predictions\*\*.



In simple terms:



> \*\*Understand the past → Analyze the present → Predict the future → Support better business decisions.\*\*



\---



\# 👨‍💻 Author



\*\*Yash Srivastav\*\*



GitHub: https://github.com/4kayasitayashu



\---



\## 🚀 Live Application



\*\*Restaurant Sales Intelligence Dashboard\*\*



https://restaurant-sales-intelligence-13.streamlit.app/



