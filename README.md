# League of Legends Win Prediction Using Machine Learning and AI
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/3a0f5ae9-4f38-45b5-87af-788933a01cc0" />

## Project Overview
This bachelor's dissertation implements a comprehensive system for League of Legends match analysis and win prediction using machine learning techniques. The system provides real-time coaching advice based on objective control metrics and leverages both solo queue and professional match data for training accurate prediction models.

## Key Features
- **Data Integration Pipeline**: Combines and cleans solo queue and professional match data
- **Win Probability Prediction**: Calculates win probability based on game state and objective control
- **Machine Learning Models**: Implements and compares multiple models for win prediction
- **Objective Weight Analysis**: Statistically determines the impact of various game objectives on win rate
- **Live Game Analysis**: Connects to the League of Legends client to provide real-time coaching
- **Web Interface**: User-friendly Gradio application for match analysis and coaching

## Technical Components

### Data Processing
- Data cleaning and normalization of match statistics
- Feature engineering focused on game objectives (dragons, barons, turrets, etc.)
- Integration of professional play data for improved model performance

### Machine Learning Implementation
- Comparative analysis of multiple models including:
  - Neural Networks
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Machines
- Feature importance analysis for model interpretability
- Hyperparameter tuning for optimal performance

### Statistical Analysis
- Objective weight calculation using logistic regression
- Phase-specific weight adjustments (early, mid, late game)
- Time-factor analysis for contextual probability adjustments

### Live Game Integration
- League of Legends Live Client Data API integration
- Real-time game state extraction and analysis
- Dynamic coaching advice generation

## Technologies Used
- **Python**: Core programming language
- **pandas & NumPy**: Data processing and analysis
- **scikit-learn**: Machine learning model implementation
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualization components
- **Gradio**: Web interface development
- **Requests**: API interaction
- **XGBoost**: Gradient boosting implementation

## Installation and Usage

### Prerequisites
- Python 3.8+
- League of Legends client (for live analysis)
- Riot API key (for player lookup)

### Setup
1. Clone the repository/ entire code
2. Run the application on Jupyter (Google Colab doesn't work with the League of Legends Live Client Data API because of how cloud computing environments operate.):

### Web Interface Usage
The application provides several tabs:
1. **Player Search**: Look up player statistics
2. **Live Game Coaching**: Get real-time advice during matches
3. **Match Analysis**: Analyze completed games
4. **About**: Information about the system

## Findings
- Inhibitors have the highest impact on win probability (4.08% per inhibitor)
- Turrets are the second most important objective (2.04% per turret)
- Baron control significantly affects win rate (1.47% per baron)
- Elder dragons have approximately twice the impact of regular dragons
- Gold difference becomes more influential in the late game

## Future Work
- Integration with more comprehensive datasets
- Expansion to include champion-specific recommendations
- Improved player behavior analysis
- Enhanced visualization of game states
- META!!

## Acknowledgements
- Riot Games API for match data access
- Oracle's Elixir for professional match statistics
- League of Legends Live Client Data API for in-game analysis

