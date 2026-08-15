# India Tourism AI

**A Hybrid ML Recommendation System for the Indian Tourism Sector**

India Tourism AI is a full-stack, hybrid machine-learning recommendation system deployed as an interactive Streamlit web application. Designed to solve information overload and choice paralysis in travel planning, this system intelligently matches travellers to 103 iconic destinations across 20 Indian states. It leverages a comprehensive dataset of user profiles, historical travel logs, Google ratings, tourism news, and textual reviews.

---

## 🚀 Key Features

* **Hybrid Recommendation Engine:** Calculates personalized destination rankings using a weighted scoring model: TF-IDF content similarity (25%), Truncated SVD collaborative filtering (20%), preference matching (15%), Google ratings (15%), popularity (10%), average user ratings (10%), and VADER-style sentiment (5%).


* **Custom NLP Sentiment Analysis:** Utilizes a domain-adapted, lexicon-based VADER-style sentiment module customized with 15 positive and 14 negative keywords specific to Indian tourism, achieving 99.6% accuracy against star-rating labels.


* **Conversational AI Assistant:** Features a built-in AI travel chat assistant powered by Google Gemini 2.0 Flash. The LLM is contextually injected with live dataset statistics, top-rated places, and real-time news signals for grounded responses.


* **Interactive Multi-Tab Interface:**
* **Recommend:** Generates ranked destination cards with explainable "Similar to..." drill-downs.


* **Explore:** A filterable, live-searchable grid of all 103 destinations.


* **Insights:** A dataset analytics dashboard featuring eight interactive Plotly charts.


* **Model Evaluation:** On-demand performance metrics, RMSE/MAE curves, and feature correlation heatmaps.


* **AI Chat:** Multi-turn session management with the Gemini AI assistant.


---

## 🛠️ Technology Stack

* **Frontend Framework:** Streamlit (>= 1.35.0)


* **Data Processing:** Pandas (>= 2.0.0), NumPy (>= 1.24.0), SciPy (>= 1.11.0)


* **Machine Learning:** Scikit-learn (>= 1.3.0) for `TfidfVectorizer`, `TruncatedSVD`, `cosine_similarity`, and `MinMaxScaler`

* **Data Visualization:** Plotly (>= 5.18.0)


* **LLM Integration:** Google Generative AI (`google-generativeai` >= 0.7.0)

---

## 📊 Dataset & Model Performance

The engine operates on a highly sparse matrix (98.5% sparsity) containing 999 travellers and 999 reviews. Despite this sparsity, the hybrid approach maintains robust performance:

* **SVD Collaborative Filtering (k=50):** RMSE of 1.8009 and MAE of 1.3239.


* **Content-Based Filtering:** Precision@5 score of 65.63%.


* **Sentiment Module:** 99.60% predictive accuracy.


---

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/india-tourism-ai.git
cd india-tourism-ai

```


2. **Install the required dependencies:**
Ensure you are using Python 3.8+ and install the requirements defined in the project report:


```bash
pip install streamlit>=1.35.0 pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 scipy>=1.11.0 plotly>=5.18.0 google-generativeai>=0.7.0

```


3. **Configure the Gemini API Key:**
To enable the AI Chat functionality, you will need a free Google Gemini API key. You can either set it as an environment variable or input it directly into the application sidebar.


```bash
export GEMINI_API_KEY="your_api_key_here"

```


4. **Run the Streamlit application:**
Launch the app via the main Python script.


```bash
streamlit run india_tourism_app.py

```
