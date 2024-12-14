# Video Recommendation System

A smart video recommendation system built with Streamlit that provides personalized video suggestions based on user preferences and viewing history. The system leverages both content-based and collaborative filtering techniques to deliver tailored recommendations through an interactive web interface.

## Demo

[![Watch the Demo](https://img.youtube.com/vi/NXxmOfZ8Mn4/0.jpg)](https://youtu.be/NXxmOfZ8Mn4)

Click the image above or [watch the demo here](https://youtu.be/NXxmOfZ8Mn4).


## Project Structure

```
video-recommender/
├── .gitignore
├── README.md
├── requirements.txt
├── video_recommender.py
├── api_fetched_data.py
├── api_logs/
│   ├── csv/
│   │   ├── summary.csv
│   │   ├── user.csv
│   │   ├── rate.csv
│   │   ├── like.csv
│   │   ├── inspire.csv
│   │   └── view.csv
│   ├── json/
│   │   ├── summary.json
│   │   ├── user.json
│   │   ├── rate.json
│   │   ├── like.json
│   │   ├── inspire.json
│   │   └── view.json
│   └── logs/
│       ├── api_fetch_20241212_145610.log
│       ├── api_fetch_20241212_145700.log
└── myenv/

```

## Features

- **Personalized Recommendations**
  - Content-based filtering using video metadata (genre, description, keywords)
  - Collaborative filtering based on user-item interaction data
  - Cold start handling for new users
  - Mood-based recommendations (Relaxed, Neutral, Energetic, Creative ,Focused)

- **Interactive User Interface**
  - Video cards with detailed information
  - Title, genre, and description display
  - Average rating with star visualization
  - Keywords displayed as tags

- **System Performance Metrics**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Coverage analysis
  - Diversity measurements
  - Visual analytics with rating correlation plots

## Data Pipeline

- **Data Loading and Processing**
  - Fetching Data from API
  - Reads metadata and user interaction data from CSV files
  - Cleans and prepares data for analysis
  - Extracts video metadata from the post_summary column

- **Recommendation Generation**
  - TF-IDF vectorization for content features
  - User-item interaction matrix computation
  - Cosine similarity calculations
  - Hybrid recommendation combining multiple approaches

## Prerequisites

- Python 3.8+
- Required libraries (specified in requirements.txt):
  ```
  streamlit==1.31.0
  pandas==2.1.4
  numpy==1.26.3
  scikit-learn==1.4.0
  scipy==1.11.4
  plotly==5.18.0
  python-dotenv==1.0.0
  ```

## Installation

1. Clone the repository
```bash
git clone https://github.com/[your-username]/video-recommender.git
cd video-recommender
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

1. Ensure all required CSV files are in the correct location:
   - Place summary.csv, user.csv, and rate.csv in ./api_logs/csv/ directory

2. Launch the Streamlit application:
```bash
streamlit run video_recommender.py
```

3. Access the application in your web browser (typically at http://localhost:8501)

## System Evaluation

The system includes a dedicated evaluation tab that provides:
- Accuracy metrics (MAE, RMSE)
- System coverage analysis
- Recommendation diversity assessment
- Visual correlation plots between actual and predicted ratings

## Future Enhancements

- Integration of deep learning models
- Real-time user interaction tracking
- Multi-language support
- Enhanced visualization capabilities
- Advanced filtering options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
