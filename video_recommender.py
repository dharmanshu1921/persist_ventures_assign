import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import ast
import random
from datetime import datetime

class VideoRecommender:
    def __init__(self):
        self.summary = None
        self.user = None
        self.rate = None
        self.user_item_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.item_similarity = None
        self.mood_categories = {
            'Relaxed': ['Educational', 'Documentary', 'Nature', 'Meditation'],
            'Neutral': ['Comedy', 'Drama', 'Lifestyle', 'Technology'],
            'Energetic': ['Action', 'Sports', 'Music', 'Adventure']
        }
        
    def load_data(self):
        self.summary = pd.read_csv('./api_logs/csv/summary.csv')
        self.user = pd.read_csv('./api_logs/csv/user.csv')
        self.rate = pd.read_csv('./api_logs/csv/rate.csv')
        self.prepare_data()
        
    def prepare_data(self):
        required_columns = ['id', 'title', 'rating_count', 'average_rating', 'post_summary']
        self.summary = self.summary[required_columns]
        self.summary.dropna(subset=['title'], inplace=True)
        self.summary['rating_count'].fillna(0, inplace=True)
        self.summary['average_rating'].fillna(0, inplace=True)
        self.summary['post_summary'].fillna("No Summary", inplace=True)
        
        self.summary['genre'] = self.summary['post_summary'].apply(self.extract_genre)
        self.summary['description'] = self.summary['post_summary'].apply(self.extract_description)
        self.summary['keywords'] = self.summary['post_summary'].apply(self.extract_fields)
        
        self.summary['combined_features'] = self.summary.apply(
            lambda row: " ".join(
                filter(None, [str(row['genre']), str(row['description']), str(row['keywords'])])
            ),
            axis=1
        )
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.summary['combined_features'] = self.summary['combined_features'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.summary['combined_features'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.summary.index, index=self.summary['title']).drop_duplicates()
        
        self.rate.columns = self.rate.columns.str.strip().str.lower()
        self.user_item_matrix = self.rate.pivot(
            index='user_id', 
            columns='post_id', 
            values='rating_percent'
        ).fillna(0)
        interaction_matrix = csr_matrix(self.user_item_matrix)
        self.item_similarity = cosine_similarity(interaction_matrix.T)

    @staticmethod
    def extract_genre(post_summary):
        try:
            summary_dict = ast.literal_eval(post_summary) if isinstance(post_summary, str) else post_summary
            return summary_dict.get('genre', None)
        except (ValueError, SyntaxError, AttributeError):
            return None

    @staticmethod
    def extract_description(post_summary):
        try:
            summary_dict = ast.literal_eval(post_summary) if isinstance(post_summary, str) else post_summary
            return summary_dict.get('description', None)
        except (ValueError, SyntaxError, AttributeError):
            return None

    @staticmethod
    def extract_fields(post_summary):
        try:
            post_summary_dict = ast.literal_eval(post_summary) if isinstance(post_summary, str) else post_summary
            keywords = [
                keyword['keyword'] for keyword in post_summary_dict.get('keywords', [])
                if keyword['weight'] > 5
            ]
            return keywords
        except (ValueError, SyntaxError, AttributeError):
            return None

    def get_video_details(self, title):
        video = self.summary[self.summary['title'] == title].iloc[0]
        return {
            'title': video['title'],
            'genre': self.extract_genre(video['post_summary']),
            'description': self.extract_description(video['post_summary']),
            'rating': video['average_rating'],
            'rating_count': video['rating_count'],
            'keywords': self.extract_fields(video['post_summary'])
        }

    def get_recommendations(self, title="", user_id=None, mood=None, is_new_user=False, n_recommendations=10):
        if is_new_user:
            return self.get_mood_based_recommendations(mood, n_recommendations)
        
        content_recs = self.get_content_recommendations(title) if title else pd.Series([])
        
        if user_id is not None:
            collab_recs = self.get_collaborative_recommendations(user_id)
            final_recommendations = pd.concat([
                pd.Series(content_recs),
                pd.Series(collab_recs)
            ]).drop_duplicates().head(n_recommendations)
        else:
            final_recommendations = content_recs.head(n_recommendations)
        
        rec_details = []
        for rec_title in final_recommendations:
            try:
                details = self.get_video_details(rec_title)
                rec_details.append(details)
            except:
                continue
                
        return rec_details

    def get_mood_based_recommendations(self, mood, n_recommendations=10):
        if not mood:
            return []
            
        preferred_genres = self.mood_categories.get(mood, [])
        
        mood_filtered = self.summary[
            self.summary['genre'].apply(
                lambda x: any(genre in str(x).lower() for genre in map(str.lower, preferred_genres))
            )
        ]
        
        top_recommendations = mood_filtered.nlargest(
            n_recommendations,
            ['average_rating', 'rating_count']
        )
        
        return [self.get_video_details(title) for title in top_recommendations['title']]

    def get_content_recommendations(self, title):
        if title not in self.indices:
            return pd.Series([])
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.summary['title'].iloc[movie_indices]

    def get_collaborative_recommendations(self, user_id):
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            scores = self.item_similarity.dot(user_ratings)
            recommended_video_ids = self.user_item_matrix.columns[
                (scores > 0) & (user_ratings == 0)
            ]
            return self.summary[self.summary['id'].isin(recommended_video_ids)]['title']
        return pd.Series([])

    def evaluate_system(self, test_size=0.2):
        rate_df = self.rate.copy()
        train_data, test_data = train_test_split(
            rate_df, test_size=test_size, random_state=42
        )
        
        actual_ratings = []
        predicted_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            post_id = row['post_id']
            actual_rating = row['rating_percent']
            
            try:
                pred_rating = self.predict_rating(user_id, post_id)
                if pred_rating is not None:
                    actual_ratings.append(actual_rating)
                    predicted_ratings.append(pred_rating)
            except:
                continue
        
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        
        total_possible = len(self.user_item_matrix.index) * len(self.user_item_matrix.columns)
        actual_predictions = sum(~np.isnan(self.user_item_matrix.values.flatten()))
        coverage = actual_predictions / total_possible
        
        unique_recommendations = len(set(predicted_ratings))
        diversity = unique_recommendations / len(self.user_item_matrix.columns)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'diversity': diversity,
            'actual_ratings': actual_ratings,
            'predicted_ratings': predicted_ratings
        }

    def predict_rating(self, user_id, post_id):
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            if post_id < len(self.item_similarity):
                similar_items = self.item_similarity[post_id]
                pred = np.sum(similar_items * user_ratings) / (np.sum(np.abs(similar_items)) + 1e-8)
                return max(0, min(100, pred))
        return None

def create_video_card(video_details):
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(
                "https://via.placeholder.com/150x100.png?text=Video",
                use_container_width =True
            )
            
        with col2:
            st.subheader(video_details['title'])
            st.write(f"Genre: {video_details['genre']}")
            if video_details['description']:
                st.write(video_details['description'][:200] + "..." if len(video_details['description']) > 200 else video_details['description'])
            
            rating = float(video_details['rating']) if video_details['rating'] else 0
            stars = "⭐" * int(rating/20)  
            st.write(f"Rating: {stars} ({rating:.1f}/100)")
            
            if video_details['keywords']:
                st.write("Tags:", end=" ")
                for keyword in video_details['keywords'][:5]:
                    st.markdown(f'<span style="background-color: blue; padding: 2px 8px; border-radius: 10px; margin-right: 5px;">{keyword}</span>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Video Recommendation System",
        page_icon="🎥",
        layout="wide"
    )

    st.markdown("""
        <style>
        .stApp {
            background-color: black;
        }
        .video-container {
            background-color: black;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px black;
        }
        .metric-card {
            background-color: black;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px black;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎥 Smart Video Recommender")
    
    recommender = VideoRecommender()
    recommender.load_data()
    
    tab1, tab2 = st.tabs(["🎬 Recommendations", "📊 System Evaluation"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("""
            ### 🎯 Get Personalized Recommendations
            Choose your user type to get started!
            """)
            
            user_type = st.radio(
                "Select User Type",
                ["New User", "Existing User"]
            )
            
            user_id = None
            selected_title = None
            mood = None
            
            if user_type == "Existing User":
                user_id_search = st.text_input("🔍 Search User ID")
                if user_id_search:
                    available_users = recommender.user_item_matrix.index.tolist()
                    matching_users = [str(uid) for uid in available_users if str(uid).startswith(user_id_search)]
                    if matching_users:
                        user_id = st.selectbox("Select User ID", matching_users)
                
                title_search = st.text_input("🔍 Search Video Title")
                if title_search:
                    available_titles = recommender.summary['title'].tolist()
                    matching_titles = [title for title in available_titles if title_search.lower() in title.lower()]
                    if matching_titles:
                        selected_title = st.selectbox("Select Video Title", matching_titles)
                
            else:  
                st.markdown("### 🎭 Select Your Mood")
                mood = st.radio(
                    "What kind of content are you interested in?",
                    ['Relaxed', 'Neutral', 'Energetic'],
                    help="We'll recommend content based on your mood!"
                )
                
                st.markdown(f"""
                #### Recommended Categories for {mood} mood:
                {', '.join(recommender.mood_categories[mood])}
                """)
            
            if st.button("🔍 Get Recommendations", key="rec_button"):
                with col1:
                    if user_type == "New User" and mood:
                        st.markdown(f"### 🎬 Recommended for {mood} Mood")
                        recommendations = recommender.get_recommendations(
                            mood=mood,
                            is_new_user=True
                        )
                    elif user_type == "Existing User" and (selected_title or user_id):
                        st.markdown("### 🎬 Personalized Recommendations")
                        recommendations = recommender.get_recommendations(
                            title=selected_title if selected_title else "",
                            user_id=user_id
                        )
                    else:
                        st.warning("Please provide the required information.")
                        recommendations = []
                    
                    if recommendations:
                        for video in recommendations:
                            with st.container():
                                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                create_video_card(video)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("No recommendations found. Try different inputs!")
    
    with tab2:
        st.header("📊 System Performance Metrics")
        
        if st.button("📈 Calculate Metrics"):
            with st.spinner("Analyzing system performance..."):
                metrics = recommender.evaluate_system()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("MAE", f"{metrics['mae']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Coverage", f"{metrics['coverage']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Diversity", f"{metrics['diversity']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("📈 Prediction Accuracy")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=metrics['actual_ratings'],
                    y=metrics['predicted_ratings'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='rgb(49, 130, 189)',
                        opacity=0.6
                    ),
                    name='Predictions'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[min(metrics['actual_ratings']), max(metrics['actual_ratings'])],
                    y=[min(metrics['actual_ratings']), max(metrics['actual_ratings'])],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Ratings',
                    xaxis_title='Actual Ratings',
                    yaxis_title='Predicted Ratings',
                    template='plotly_white',
                    height=500,
                    width=800
                )
                
                st.plotly_chart(fig, use_container_width=True)

def run_app():
    main()

if __name__ == "__main__":
    run_app()