import json
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time
import pickle
import gc
import requests
import gradio as gr
import warnings
import ssl
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# Set up logging and directory structure
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Disable SSL verification for local connections (Live Client Data API)
ssl._create_default_https_context = ssl._create_unverified_context

def log_message(message):
    """Log message to file and print to console"""
    with open('logs/training_log.txt', 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")
    print(message)

###########################################
# PART 1: DATA PREPARATION AND CLEANING
###########################################

def combine_soloq_pro_data():
    """
    Combine solo queue and professional match data with cleaning to remove non-ranked matches
    """
    # Load the datasets
    pro_df = pd.read_csv("Improve League win rate using MLandAI/2025_LoL_esports_match_data_from_OraclesElixir.csv")
    solo_df = pd.read_csv("Improve League win rate using MLandAI/soloq_match_stats.csv")
    
    # ---- STEP 1: Clean solo queue data to remove non-ranked matches ----
    original_solo_count = len(solo_df)
    print(f"Original solo queue data: {original_solo_count} records")
    
    # Filter out ME1 matches (non-ranked games) and other non-ranked identifiers
    solo_df = solo_df[~solo_df['Match ID'].astype(str).str.contains('ME1')]
    solo_df = solo_df[~solo_df['Match ID'].astype(str).str.contains('CUSTOM', case=False)]
    solo_df = solo_df[~solo_df['Match ID'].astype(str).str.contains('ARAM', case=False)]
    solo_df = solo_df[~solo_df['Match ID'].astype(str).str.contains('TFT', case=False)]
    
    # Report on solo queue filtering results
    removed_count = original_solo_count - len(solo_df)
    print(f"Removed {removed_count} non-ranked matches ({removed_count/original_solo_count:.1%} of solo queue data)")
    print(f"Retained {len(solo_df)} valid solo queue matches")
    
    # Reset index after filtering
    solo_df = solo_df.reset_index(drop=True)
    
    # ---- STEP 2: Filter professional data for target leagues ----
    target_leagues = ['LEC', 'LCK', 'LTA']
    filtered_pro_df = pro_df[pro_df['league'].isin(target_leagues)]

    print("\nFiltered Pro Play Counts by League:")
    print(filtered_pro_df['league'].value_counts())

    # Save filtered pro data
    filtered_pro_df.to_csv('Improve League win rate using MLandAI/filtered_2025_proplay.csv', index=False)
    print(f"Filtered pro play data saved. Total matches: {len(filtered_pro_df)}")

    # ---- STEP 3: Process professional data to match solo queue format ----
    # Process player rows: select participantid between 1 and 10 (i.e., individual players)
    player_df = filtered_pro_df[filtered_pro_df['participantid'].between(1, 10)].copy()

    # Map game id and region appropriately
    player_df['Match ID'] = player_df['gameid']

    # Map leagues to regions: LCK -> KR, LTA -> AMERICA, LEC -> EUROPE
    region_mapping = {'LCK': 'KR', 'LTA': 'AMERICAS', 'LEC': 'EUROPE'}
    player_df['Region'] = player_df['league'].map(region_mapping)

    # Convert gamelength from seconds to minutes
    player_df['Game Duration'] = player_df['gamelength'] / 60

    # Role mapping: change utility to support; ensure consistency by converting to uppercase
    role_mapping = {'top': 'TOP', 'jng': 'JUNGLE', 'mid': 'MIDDLE', 'bot': 'BOTTOM', 'sup': 'UTILITY'}
    player_df['Role'] = player_df['position'].map(role_mapping)

    # Compute KDA, ensuring deaths are at least 1
    player_df['KDA'] = (player_df['kills'] + player_df['assists']) / player_df['deaths'].clip(lower=1)

    # Map other performance metrics
    player_df['Gold Per Minute'] = player_df['earned gpm']
    player_df['Damage Per Minute'] = player_df['dpm']
    player_df['Vision Per Minute'] = player_df['vspm']

    # Determine side (Blue/Red) based on participant ID
    player_df['Side'] = player_df['participantid'].apply(lambda x: 'Blue' if x <= 5 else 'Red')

    # ---- STEP 4: Extract and compute objective data for professional matches ----
    # Extract team-level objectives for Blue team (participantid == 100)
    blue_obj = filtered_pro_df[filtered_pro_df['participantid'] == 100][['gameid', 'towers', 'dragons', 'barons']].copy()
    blue_obj = blue_obj.rename(columns={'towers': 'blue_towers', 'dragons': 'blue_dragons', 'barons': 'blue_barons'})

    # Extract team-level objectives for Red team (participantid == 200)
    red_obj = filtered_pro_df[filtered_pro_df['participantid'] == 200][['gameid', 'opp_towers', 'opp_dragons', 'opp_barons']].copy()
    red_obj = red_obj.rename(columns={'opp_towers': 'red_towers', 'opp_dragons': 'red_dragons', 'opp_barons': 'red_barons'})

    # Merge team objectives on gameid
    team_obj = pd.merge(blue_obj, red_obj, on='gameid', how='inner')

    # Merge team objectives into the pro player dataframe on gameid
    player_df = player_df.merge(team_obj, left_on='gameid', right_on='gameid', how='left')

    # Assign team-level objective values to individual players based on their 'side'
    player_df['Turret Takedowns'] = player_df.apply(lambda row: row['blue_towers'] if row['Side'].strip().lower() == 'blue'
                                                    else row['red_towers'], axis=1)
    player_df['Dragon Takedowns'] = player_df.apply(lambda row: row['blue_dragons'] if row['Side'].strip().lower() == 'blue'
                                                    else row['red_dragons'], axis=1)
    player_df['Baron Takedowns'] = player_df.apply(lambda row: row['blue_barons'] if row['Side'].strip().lower() == 'blue'
                                                    else row['red_barons'], axis=1)

    # Compute Gold Difference from team totals
    blue_gold = filtered_pro_df[filtered_pro_df['participantid'] == 100][['gameid', 'totalgold']].rename(columns={'totalgold': 'blue_totalgold'})
    red_gold = filtered_pro_df[filtered_pro_df['participantid'] == 200][['gameid', 'totalgold']].rename(columns={'totalgold': 'red_totalgold'})
    team_gold = blue_gold.merge(red_gold, on='gameid', how='inner')
    team_gold['Gold Difference'] = team_gold['blue_totalgold'] - team_gold['red_totalgold']
    player_df = player_df.merge(team_gold[['gameid', 'Gold Difference']], on='gameid', how='left')

    # Compute other differences using available columns from pro data
    player_df['Turret Difference'] = player_df.apply(
        lambda row: row['towers'] - row['opp_towers'] if row['Side'].strip().lower() == 'blue'
                    else row['opp_towers'] - row['towers'], axis=1
    )
    player_df['Dragon Difference'] = player_df.apply(
        lambda row: row['dragons'] - row['opp_dragons'] if row['Side'].strip().lower() == 'blue'
                    else row['opp_dragons'] - row['dragons'], axis=1
    )
    player_df['Baron Difference'] = player_df.apply(
        lambda row: row['barons'] - row['opp_barons'] if row['Side'].strip().lower() == 'blue'
                    else row['opp_barons'] - row['barons'], axis=1
    )

    # Set Win from result column
    player_df['Win'] = player_df['result']

    # Set Items Bought and Legendary Items to empty lists (as strings)
    player_df['Items Bought'] = '[]'
    player_df['Legendary Items'] = '[]'

    # ---- STEP 5: Handle missing values ----
    # For numerical columns, fill with mean
    numerical_cols = player_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        player_df[col] = player_df[col].fillna(player_df[col].mean())

    # For categorical columns, fill with most frequent value
    categorical_cols = player_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Check if mode is not empty before filling
        if player_df[col].mode().shape[0] > 0:
            player_df[col] = player_df[col].fillna(player_df[col].mode()[0])
        else:
            # Handle case where mode is empty (e.g., all NaN values)
            player_df[col] = player_df[col].fillna('Unknown')

    # ---- STEP 6: Select and standardize columns ----
    # Select columns that match the solo queue data attributes
    solo_columns = [
        'Match ID', 'Region', 'Game Duration', 'Role', 'KDA', 'Gold Per Minute', 'Damage Per Minute',
        'Vision Per Minute', 'Turret Takedowns', 'Dragon Takedowns', 'Baron Takedowns', 'Win', 'Items Bought',
        'Legendary Items', 'Gold Difference', 'Turret Difference', 'Dragon Difference', 'Baron Difference'
    ]
    pro_player_df = player_df[solo_columns]

    # Ensure region label for NA is set; fill missing regions with 'AMERICAS'
    solo_df['Region'] = solo_df['Region'].fillna('AMERICAS')

    # If roles in solo_df are not uppercase, convert them to uppercase for consistency
    solo_df['Role'] = solo_df['Role'].str.upper()

    # ---- STEP 7: Combine datasets and save ----
    # Combine the pro play and solo queue data
    combined_df = pd.concat([solo_df, pro_player_df], ignore_index=True)

    # Save the combined dataset
    combined_df.to_csv('Improve League win rate using MLandAI/Complete_match_data(solo+pro).csv', index=False)
    print(f"\nCombined data saved. Total records: {len(combined_df)}")
    print(f"  - Solo queue matches: {len(solo_df)}")
    print(f"  - Professional matches: {len(pro_player_df)}")

    return combined_df

###########################################
# PART 2: OBJECTIVE WEIGHT CALCULATION
###########################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import eli5
from eli5.sklearn import PermutationImportance

def calculate_objective_weights(data_path):
    """
    Calculate precise weights for objectives by analyzing match data

    Parameters:
    data_path (str): Path to the match data CSV file

    Returns:
    dict: Dictionary of objective weights and analysis results
    """
    print(f"Loading match data from {data_path}...")
    # Load your match dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} matches.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Select relevant objective-related features
    objective_features = [
        'Dragon Takedowns', 'Dragon Difference', 'Baron Takedowns', 'Baron Difference',
        'Turret Takedowns', 'Turret Difference', 'Gold Difference', 'Game Duration'
    ]

    # Make sure all required columns exist
    missing_cols = [col for col in objective_features if col not in df.columns]
    if missing_cols:
        print(f"Missing columns in dataset: {missing_cols}")
        print("Available columns:", df.columns.tolist())

        # Try to find alternative column names
        alt_features = []
        for feature in objective_features:
            found = False
            for col in df.columns:
                if feature.lower().replace(' ', '_') in col.lower().replace(' ', '_'):
                    alt_features.append(col)
                    found = True
                    print(f"Using '{col}' for '{feature}'")
                    break
            if not found:
                print(f"No alternative found for '{feature}'")

        if alt_features:
            objective_features = alt_features
        else:
            return None

    # Prepare feature matrix and target
    X = df[objective_features]
    y = df['Win']  # Assuming 'Win' is a binary column (0 or 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Extract coefficients as raw weights
    raw_weights = {}
    for feature, coef in zip(objective_features, model.coef_[0]):
        raw_weights[feature] = coef

    # Normalize coefficients to get percentage impacts
    abs_coefs = np.abs(model.coef_[0])
    total_impact = np.sum(abs_coefs)

    normalized_weights = {}
    for feature, coef in zip(objective_features, model.coef_[0]):
        normalized_weights[feature] = abs(coef) / total_impact

    # Calculate transformed weights for practical use (as percentages)
    practical_weights = {}
    for feature, weight in normalized_weights.items():
        practical_weights[feature] = weight * 100  # Convert to percentage

    # Use permutation importance for validation
    print("\nCalculating permutation importance...")
    perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
    perm_importance = {}
    for feature, imp in zip(objective_features, perm.feature_importances_):
        perm_importance[feature] = imp

    # Try to use SHAP values for deeper analysis
    try:
        print("\nCalculating SHAP values...")
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

        # Calculate SHAP-based importance
        shap_importance = {}
        for i, feature in enumerate(objective_features):
            shap_importance[feature] = np.mean(np.abs(shap_values[:, i]))

        # Normalize SHAP values
        total_shap = sum(shap_importance.values())
        shap_weights = {k: v/total_shap * 100 for k, v in shap_importance.items()}
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        shap_importance = None
        shap_weights = None

    # Analyze game time influence on objective weights
    print("\nAnalyzing game time influence...")

    # Create game phase column
    df['GamePhase'] = pd.cut(
        df['Game Duration'],
        bins=[0, 15, 25, float('inf')],
        labels=['Early', 'Mid', 'Late']
    )

    # Calculate phase-specific weights
    phase_weights = {}
    for phase in ['Early', 'Mid', 'Late']:
        phase_df = df[df['GamePhase'] == phase]
        if len(phase_df) > 100:  # Ensure enough samples
            X_phase = phase_df[objective_features]
            y_phase = phase_df['Win']

            phase_model = LogisticRegression(max_iter=1000)
            phase_model.fit(X_phase, y_phase)

            phase_weights[phase] = {}
            for feature, coef in zip(objective_features, phase_model.coef_[0]):
                phase_weights[phase][feature] = coef

    # Prepare plots
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(practical_weights.keys()), y=list(practical_weights.values()))
    plt.title('Normalized Feature Weights (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('objective_weights.png')

    if shap_importance:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(shap_weights.keys()), y=list(shap_weights.values()))
        plt.title('SHAP-based Feature Weights (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('shap_weights.png')

    # Compile results
    results = {
        'model_accuracy': accuracy,
        'model_auc': auc,
        'raw_weights': raw_weights,
        'normalized_weights': normalized_weights,
        'practical_weights': practical_weights,
        'permutation_importance': perm_importance,
        'shap_importance': shap_importance,
        'shap_weights': shap_weights,
        'phase_weights': phase_weights
    }

    # Format weights for final win probability formula
    formatted_weights = {}
    for key, value in practical_weights.items():
        formatted_key = key.lower().replace(' ', '_')
        if 'dragon' in formatted_key:
            formatted_weights['dragon_weight'] = value / 100
        elif 'baron' in formatted_key:
            formatted_weights['baron_weight'] = value / 100
        elif 'turret' in formatted_key:
            formatted_weights['turret_weight'] = value / 100
        elif 'herald' in formatted_key:
            formatted_weights['herald_weight'] = value / 100
        elif 'gold' in formatted_key:
            formatted_weights['gold_per_1k_weight'] = value / 1000  # Adjust for per 1k gold

    # Apply special adjustments based on expert knowledge
    if 'elder_dragon_weight' not in formatted_weights:
        # Elder dragons often have higher importance than regular dragons
        if 'dragon_weight' in formatted_weights:
            formatted_weights['elder_dragon_weight'] = formatted_weights['dragon_weight'] * 2

    if 'inhibitor_weight' not in formatted_weights:
        # Estimate inhibitor weight if not directly measured
        if 'turret_weight' in formatted_weights:
            formatted_weights['inhibitor_weight'] = formatted_weights['turret_weight'] * 2

    results['formatted_weights'] = formatted_weights

    # Print final weights for win probability formula
    print("\nFinal weights for win probability formula:")
    for key, value in formatted_weights.items():
        print(f"{key}: {value:.4f}")

    return results

def generate_win_probability_formula(weights):
    """
    Generate a formula for win probability based on calculated weights

    Parameters:
    weights (dict): Dictionary of calculated weights

    Returns:
    str: Python code for win probability calculation
    """
    if not weights or 'formatted_weights' not in weights:
        return "# No weights available to generate formula"

    fw = weights['formatted_weights']

    formula = """
def predict_win_probability_from_objectives(game_state):
    \"\"\"
    Predict win probability based on objective control metrics using statistically derived weights

    Parameters:
    game_state (dict): A dictionary containing game state information including
                      objective differences and game time

    Returns:
    float: Win probability (0.0 to 1.0)
    \"\"\"
    # Base win probability (50-50 at start)
    win_prob = 0.5

    # Weights derived from statistical analysis of %(num_matches)d matches
    weights = {
        'dragon_diff': %(dragon_weight).4f,  # Each dragon adds this to win probability
        'elder_dragon': %(elder_dragon_weight).4f,  # Elder dragon impact
        'baron_diff': %(baron_weight).4f,    # Each baron impact
        'herald_diff': %(herald_weight).4f,   # Each herald impact
        'turret_diff': %(turret_weight).4f,   # Each turret impact
        'inhibitor_diff': %(inhibitor_weight).4f, # Each inhibitor impact
        'gold_diff_per_1k': %(gold_per_1k_weight).4f # Each 1000 gold difference impact
    }
    """ % {
        'num_matches': 10000,  # Placeholder for your actual match count
        'dragon_weight': fw.get('dragon_weight', 0.08),
        'elder_dragon_weight': fw.get('elder_dragon_weight', 0.15),
        'baron_weight': fw.get('baron_weight', 0.12),
        'herald_weight': fw.get('herald_weight', 0.05),
        'turret_weight': fw.get('turret_weight', 0.03),
        'inhibitor_weight': fw.get('inhibitor_weight', 0.07),
        'gold_per_1k_weight': fw.get('gold_per_1k_weight', 0.02)
    }

    # Add game phase dependent factors
    formula += """
      # Adjust for dragon control
      dragon_diff = game_state.get('dragon_diff', 0)
      win_prob += dragon_diff * weights['dragon_diff']

      # Special case for Elder Dragon
      if game_state.get('has_elder_dragon', False):
          win_prob += weights['elder_dragon']

      # Adjust for Baron control
      baron_diff = game_state.get('baron_diff', 0)
      win_prob += baron_diff * weights['baron_diff']

      # Adjust for Herald control
      herald_diff = game_state.get('herald_diff', 0)
      win_prob += herald_diff * weights['herald_diff']

      # Adjust for turret control
      turret_diff = game_state.get('turret_diff', 0)
      win_prob += turret_diff * weights['turret_diff']

      # Adjust for inhibitor control
      inhibitor_diff = game_state.get('inhibitor_diff', 0)
      win_prob += inhibitor_diff * weights['inhibitor_diff']

      # Adjust for gold difference
      gold_diff = game_state.get('gold_diff', 0)
      gold_diff_k = gold_diff / 1000  # Convert to thousands
      win_prob += gold_diff_k * weights['gold_diff_per_1k']

      # Game time factor - early advantages mean less than late game advantages
      game_time = game_state.get('game_time', 15)  # Default to mid-game if not provided

    if game_time < 15:
        # Early game - objectives matter less
        time_factor = 0.7
    elif game_time < 25:
        # Mid game - objectives matter more
        time_factor = 1.0
    else:
        # Late game - objectives matter most
        time_factor = 1.3

    # Apply time factor to the deviation from 50%
    win_prob = 0.5 + (win_prob - 0.5) * time_factor

    # Clamp probability between 0 and 1
    win_prob = max(0.0, min(1.0, win_prob))

    return win_prob
    """

    return formula

# Example usage
if __name__ == "__main__":
    data_path = "processed_match_data.csv"

    # Calculate weights
    weights = calculate_objective_weights(data_path)

    if weights:
        # Generate formula
        formula = generate_win_probability_formula(weights)

        # Save formula to file
        with open("win_probability_formula.py", "w") as f:
            f.write(formula)

        print("\nFormula generated and saved to win_probability_formula.py")

###########################################
# PART 3: MACHINE LEARNING MODEL
###########################################

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to complete training and evaluation of models
def train_evaluate_models(processed_df):
    # Prepare features and target
    drop_columns = ['Match ID', 'Items Bought', 'Legendary Items']
    drop_columns = [col for col in drop_columns if col in processed_df.columns]

    X = processed_df.drop(['Win'] + drop_columns, axis=1)
    y = processed_df['Win']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define features by type
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features) if categorical_features else ('cat', 'passthrough', [])
        ])

    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png', dpi=300)
        plt.show()

        # ROC Curve (for models that support predict_proba)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png', dpi=300)
            plt.show()
        except:
            print(f"ROC curve cannot be computed for {name}")

    # Compare model performance
    plt.figure(figsize=(12, 6))
    models_df = pd.DataFrame({'Model': list(results.keys()), 'Accuracy': list(results.values())})
    models_df = models_df.sort_values('Accuracy', ascending=False)
    sns.barplot(x='Accuracy', y='Model', data=models_df)
    plt.title('Model Comparison')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

    # Select the best model
    best_model_name = models_df.iloc[0]['Model']
    print(f"Best model: {best_model_name} with accuracy {models_df.iloc[0]['Accuracy']:.4f}")

    # Hyperparameter tuning for the best model
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

    # Define parameter grid based on the best model
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__penalty': ['l1', 'l2']
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    elif best_model_name == 'SVM':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1]
        }
    elif best_model_name == 'KNN':
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]
        }
    elif best_model_name == 'Decision Tree':
        param_grid = {
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'AdaBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1]
        }
    elif best_model_name == 'Neural Network':
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }

    # Create a pipeline with the best model
    best_model = models[best_model_name]
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate the tuned model
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned model accuracy on test data: {tuned_accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance analysis for the tuned model
    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        # Get the feature names after preprocessing
        categorical_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        if categorical_features:
            cat_feature_names = categorical_encoder.get_feature_names_out(categorical_features)
            all_feature_names = numerical_features + list(cat_feature_names)
        else:
            all_feature_names = numerical_features

        # Get feature importances
        feature_importances = best_pipeline.named_steps['classifier'].feature_importances_

        # Create DataFrame for easier visualization
        feature_importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig('top_features_importance.png', dpi=300)
        plt.show()

        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))

    # Save the best model
    joblib.dump(best_pipeline, 'lol_win_prediction_model.pkl')
    print("Best model saved as 'lol_win_prediction_model.pkl'")

    return best_pipeline


###########################################
# PART 4: DATA PREPROCESSING
###########################################

def preprocess_data(df):
    """Preprocess data for neural network"""
    log_message("Preprocessing data...")
    
    # Drop unnecessary columns
    drop_columns = ['Match ID', 'Items Bought', 'Legendary Items']
    drop_columns = [col for col in drop_columns if col in df.columns]
    
    # Split features and target
    X = df.drop(['Win'] + drop_columns, axis=1)
    y = df['Win']
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Get categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Process numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])
    
    # Process categorical features if any
    if categorical_features:
        try:
            # For newer scikit-learn versions
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # For older scikit-learn versions
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
        X_train_cat = encoder.fit_transform(X_train[categorical_features])
        X_test_cat = encoder.transform(X_test[categorical_features])
        
        # Combine numerical and categorical features
        X_train_processed = np.hstack((X_train_num, X_train_cat))
        X_test_processed = np.hstack((X_test_num, X_test_cat))
        
        # Update feature names to include encoded categorical features
        encoded_cat_features = []
        for i, feature in enumerate(categorical_features):
            feature_values = encoder.categories_[i]
            encoded_cat_features.extend([f"{feature}_{value}" for value in feature_values])
        
        feature_names = numerical_features + encoded_cat_features
    else:
        X_train_processed = X_train_num
        X_test_processed = X_test_num
    
    # Create preprocessor object for future use
    preprocessor = {
        'scaler': scaler,
        'encoder': encoder if categorical_features else None,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'feature_names': feature_names
    }
    
    log_message(f"Preprocessing complete. X_train shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, feature_names, preprocessor

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['losses'])
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(history['train_accuracies']) * 10, 10), history['train_accuracies'], label='Train')
    if 'val_accuracies' in history and history['val_accuracies']:
        plt.plot(range(0, len(history['val_accuracies']) * 10, 10), history['val_accuracies'], label='Validation')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

###########################################
# PART 5: WIN PREDICTION MODEL TRAINING
###########################################

def lol_win_prediction_workflow(data_path):
    """End-to-end workflow for LoL win prediction using machine learning"""
    start_time = time.time()
    log_message("=== League of Legends Win Prediction with Machine Learning ===")
    
    # Load data
    log_message(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        log_message(f"Loaded {len(df)} records from existing data file.")
    except Exception as e:
        log_message(f"Error loading data: {e}")
        return None, None
    
    # Calculate objective weights
    weights = calculate_objective_weights(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data(df)
    
    # Save preprocessed data
    with open('models/preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'preprocessor': preprocessor,
            'objective_weights': weights
        }, f)
    log_message("Preprocessed data saved to models/preprocessed_data.pkl")
    
    # Train and compare machine learning models
    best_model = train_evaluate_models('processed_match_data.csv')
    
    # Save the best model
    best_model_path = f'models/lol_win_prediction_model_{Neural.lower().replace(" ", "_")}.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    log_message(f"Best model Neural Network saved to {best_model_path}")
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return best_model, results, weights

###########################################
# PART 6: OBJECTIVE-BASED WIN PREDICTION
###########################################
def load_weights():
    """
    Load statistically derived weights for objective importance
    
    Returns:
    dict: Dictionary of weights for different objectives
    """
    try:
        # Try to load weights from file if available
        with open('models/objective_weights.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        # Default weights if file not found
        return {
            'dragon_weight': 0.0072,
            'elder_dragon_weight': 0.0144,
            'baron_weight': 0.0147,
            'herald_weight': 0.0072,
            'turret_weight': 0.0204,
            'inhibitor_weight': 0.0408,
            'gold_diff_per_1k_weight': 0.0001
        }
# Win probability prediction function
def predict_win_probability_from_objectives(game_state, weights=None):
    """
    Predict win probability based on objective control metrics using statistically derived weights
    """
    # Base win probability (50-50 at start)
    win_prob = 0.5
    
    # Use provided weights or default to statistically derived weights
    if weights is None:
        weights = load_weights()
    
    # Adjust for dragon control
    dragon_diff = game_state.get('dragon_diff', 0)
    win_prob += dragon_diff * weights.get('dragon_weight', 0.0072)
    
    # Special case for Elder Dragon
    if game_state.get('has_elder_dragon', False):
        win_prob += weights.get('elder_dragon_weight', 0.0144)
    
    # Adjust for Baron control
    baron_diff = game_state.get('baron_diff', 0)
    win_prob += baron_diff * weights.get('baron_weight', 0.0147)
    
    # Adjust for Herald control
    herald_diff = game_state.get('herald_diff', 0)
    win_prob += herald_diff * weights.get('herald_weight', 0.0072)
    
    # Adjust for turret control
    turret_diff = game_state.get('turret_diff', 0)
    win_prob += turret_diff * weights.get('turret_weight', 0.0204)
    
    # Adjust for inhibitor control
    inhibitor_diff = game_state.get('inhibitor_diff', 0)
    win_prob += inhibitor_diff * weights.get('inhibitor_weight', 0.0408)
    
    # Adjust for gold difference
    gold_diff = game_state.get('gold_diff', 0)
    gold_diff_k = gold_diff / 1000  # Convert to thousands
    win_prob += gold_diff_k * weights.get('gold_diff_per_1k_weight', 0.0001)
    
    # Game time factor - early advantages mean less than late game advantages
    game_time = game_state.get('game_time', 15)  # Default to mid-game if not provided
    
    if game_time < 15:
        # Early game - objectives matter less
        time_factor = 0.7
    elif game_time < 25:
        # Mid game - objectives matter more
        time_factor = 1.0
    else:
        # Late game - objectives matter most
        time_factor = 1.3
    
    # Apply time factor to the deviation from 50%
    win_prob = 0.5 + (win_prob - 0.5) * time_factor
    
    # Clamp probability between 0 and 1
    win_prob = max(0.0, min(1.0, win_prob))
    
    return win_prob

def generate_objective_based_advice(game_state, win_probability, weights=None):
    """
    Generate advice based on objective control and current win probability with precise weights
    
    Parameters:
    game_state (dict): Game state information with objective metrics
    win_probability (float): Current win probability
    weights (dict): Optional dictionary of objective weights
    
    Returns:
    list: List of advice strings
    """
    # Use provided weights or default to statistically derived weights
    if weights is None:
        # Default weights if not provided
        weights = {
            'dragon_weight': 0.0072,
            'elder_dragon_weight': 0.0144,
            'baron_weight': 0.0147,
            'herald_weight': 0.0072,
            'turret_weight': 0.0204,
            'inhibitor_weight': 0.0408,
            'gold_diff_per_1k_weight': 0.0000
        }
    
    advice = []
    
    # Game time-based priorities
    game_time = game_state.get('game_time', 0)
    
    # Focus on high-impact objectives based on our weights
    if game_time < 15:  # Early game
        if game_state.get('turret_diff', 0) <= 0:
            advice.append(f"Focus on securing the first turret ({weights['turret_weight']*100:.2f}% win probability impact per turret).")
        
        if game_state.get('herald_diff', 0) <= 0 and game_time >= 8:
            advice.append("Prioritize Rift Herald to help secure turrets - use it for the first turret or mid turret if possible.")
        
        if game_state.get('dragon_diff', 0) < 0:
            advice.append(f"Contest dragons when safe, but prioritize turrets if forced to choose (turrets have {weights['turret_weight']/weights['dragon_weight']:.1f}× higher impact).")
    
    elif game_time < 25:  # Mid game
        if game_state.get('inhibitor_diff', 0) <= 0:
            advice.append(f"Focus on securing inhibitors when possible ({weights['inhibitor_weight']*100:.2f}% win probability impact per inhibitor).")
        
        if game_state.get('baron_diff', 0) <= 0 and game_time >= 20:
            advice.append(f"Set up vision for Baron ({weights['baron_weight']*100:.2f}% win probability impact) - use it to pressure inhibitors for maximum advantage.")
        
        if game_state.get('dragon_diff', 0) < -1:
            advice.append("Don't let the enemy stack too many dragons - contest when your team has advantage.")
    
    else:  # Late game
        if game_state.get('inhibitor_diff', 0) <= 0:
            advice.append(f"Securing inhibitors should be your highest priority ({weights['inhibitor_weight']*100:.2f}% win impact, the most influential objective).")
        
        if game_state.get('baron_diff', 0) <= 0:
            advice.append("Control Baron to help secure inhibitors - the combined Baron+inhibitor advantage is very strong.")
        
        if game_state.get('dragon_diff', 0) >= 3 and not game_state.get('has_elder_dragon', False):
            advice.append("Secure Dragon Soul - the combined effect of multiple dragons becomes significant.")
        
        if game_state.get('has_elder_dragon', False):
            advice.append(f"With Elder Dragon ({weights['elder_dragon_weight']*100:.2f}% win impact), group and force fights to maximize the execute potential.")
    
    # Win probability based advice
    if win_probability > 0.7:  # High win probability
        advice.append(f"You're significantly ahead - prioritize inhibitors ({weights['inhibitor_weight']*100:.2f}% impact) to extend your lead safely.")
        advice.append("Take objectives methodically without forcing risky fights - your advantage comes from objective control.")
    
    elif win_probability < 0.3:  # Low win probability
        advice.append(f"Look for high-impact inhibitor trades ({weights['inhibitor_weight']*100:.2f}% impact) even if you lose something of lower value.")
        advice.append(f"Set up vision to contest Baron - preventing enemy Baron ({weights['baron_weight']*100:.2f}% impact) is crucial when behind.")
        advice.append(f"Focus on defending your inhibitors at all costs - losing multiple inhibitors ({weights['inhibitor_weight']*100:.2f}% each) is extremely difficult to overcome.")
    
    else:  # Even game
        advice.append("In an even game, vision control around high-impact objectives (Baron, inhibitor turrets) is critical.")
        advice.append(f"If forced to choose, prioritize turrets ({weights['turret_weight']*100:.2f}% impact) over dragons ({weights['dragon_weight']*100:.2f}% impact) in the current game state.")
        advice.append("Look for picks before major objectives to create a numbers advantage.")
    
    # Select 3 most relevant pieces of advice to avoid overwhelming the player
    return advice[:3]
###########################################
# PART 7: LIVE GAME ANALYSIS
###########################################

def extract_objective_data_from_live_game(live_game_data):
    """
    Extract objective control metrics from Live Client Data API response
    
    Parameters:
    live_game_data (dict): Response from League of Legends Live Client Data API
    
    Returns:
    dict: Game state information with objective metrics
    """
    game_state = {}
    
    try:
        # Get active player team
        active_player = live_game_data.get('activePlayer', {})
        all_players = live_game_data.get('allPlayers', [])
        events = live_game_data.get('events', {}).get('Events', [])
        
        # Find active player in all players list to determine team
        active_player_name = active_player.get('summonerName', '')
        active_player_team = None
        
        for player in all_players:
            if player.get('summonerName', '') == active_player_name:
                active_player_team = player.get('team', '')
                break
        
        if not active_player_team:
            return {}
        
        # Separate players by team
        team_players = [p for p in all_players if p.get('team', '') == active_player_team]
        enemy_players = [p for p in all_players if p.get('team', '') != active_player_team]
        
        # Count team kills
        team_kills = sum([p.get('scores', {}).get('kills', 0) for p in team_players])
        enemy_kills = sum([p.get('scores', {}).get('kills', 0) for p in enemy_players])
        game_state['kill_diff'] = team_kills - enemy_kills
        
        # Game time
        game_state['game_time'] = live_game_data.get('gameData', {}).get('gameTime', 0) / 60  # Convert to minutes
        
        # Parse events for objectives
        team_dragons = 0
        enemy_dragons = 0
        team_barons = 0
        enemy_barons = 0
        team_heralds = 0
        enemy_heralds = 0
        team_turrets = 0
        enemy_turrets = 0
        team_inhibitors = 0
        enemy_inhibitors = 0
        has_elder_dragon = False
        
        for event in events:
            event_name = event.get('EventName', '')
            
            if 'DragonKill' in event_name:
                killer_team = event.get('KillerTeam', '')
                dragon_type = event.get('DragonType', '')
                
                if killer_team == active_player_team:
                    team_dragons += 1
                    if 'Elder' in dragon_type:
                        has_elder_dragon = True
                else:
                    enemy_dragons += 1
            
            elif 'BaronKill' in event_name:
                killer_team = event.get('KillerTeam', '')
                if killer_team == active_player_team:
                    team_barons += 1
                else:
                    enemy_barons += 1
            
            elif 'HeraldKill' in event_name:
                killer_team = event.get('KillerTeam', '')
                if killer_team == active_player_team:
                    team_heralds += 1
                else:
                    enemy_heralds += 1
            
            elif 'TurretKilled' in event_name:
                killer_team = event.get('KillerTeam', '')
                if killer_team == active_player_team:
                    team_turrets += 1
                else:
                    enemy_turrets += 1
            
            elif 'InhibKilled' in event_name:
                killer_team = event.get('KillerTeam', '')
                if killer_team == active_player_team:
                    team_inhibitors += 1
                else:
                    enemy_inhibitors += 1
        
        # Set game state with collected data
        game_state['dragon_diff'] = team_dragons - enemy_dragons
        game_state['baron_diff'] = team_barons - enemy_barons
        game_state['herald_diff'] = team_heralds - enemy_heralds
        game_state['turret_diff'] = team_turrets - enemy_turrets
        game_state['inhibitor_diff'] = team_inhibitors - enemy_inhibitors
        game_state['has_elder_dragon'] = has_elder_dragon
        
        # Try to find gold in player stats
        team_gold = sum([p.get('scores', {}).get('totalGold', 0) for p in team_players])
        enemy_gold = sum([p.get('scores', {}).get('totalGold', 0) for p in enemy_players])
        game_state['gold_diff'] = team_gold - enemy_gold
        
        return game_state
    
    except Exception as e:
        print(f"Error extracting objective data: {e}")
        return {}

def get_objective_analysis(live_game_data, weights=None):
    """
    Complete objective analysis pipeline for a live game with statistically precise weights
    
    Parameters:
    live_game_data (dict): Response from League of Legends Live Client Data API
    weights (dict): Optional dictionary of objective weights
    
    Returns:
    dict: Analysis results including win probability and advice
    """
    # Extract objective metrics from live game data
    game_state = extract_objective_data_from_live_game(live_game_data)
    
    if not game_state:
        return {
            "error": "Could not extract objective data from game",
            "win_probability": 0.5,
            "advice": ["Could not generate objective-based advice"]
        }
    
    # Calculate win probability using precise weights
    win_probability = predict_win_probability_from_objectives(game_state, weights)
    
    # Generate advice based on precise weights
    advice = generate_objective_based_advice(game_state, win_probability, weights)
    
    # Calculate individual objective contributions to win probability
    objective_contributions = {}
    if weights:
        objective_contributions = {
            "dragons": game_state.get('dragon_diff', 0) * weights.get('dragon_weight', 0.0072),
            "elder_dragon": weights.get('elder_dragon_weight', 0.0144) if game_state.get('has_elder_dragon', False) else 0,
            "barons": game_state.get('baron_diff', 0) * weights.get('baron_weight', 0.0147),
            "heralds": game_state.get('herald_diff', 0) * weights.get('herald_weight', 0.0072),
            "turrets": game_state.get('turret_diff', 0) * weights.get('turret_weight', 0.0204),
            "inhibitors": game_state.get('inhibitor_diff', 0) * weights.get('inhibitor_weight', 0.0408)
        }
    
    # Format objectives for display
    objectives = {
        "dragons": (game_state.get('dragon_diff', 0) + game_state.get('enemy_dragons', 0), 
                   game_state.get('enemy_dragons', 0)),
        "barons": (game_state.get('baron_diff', 0) + game_state.get('enemy_barons', 0), 
                  game_state.get('enemy_barons', 0)),
        "heralds": (game_state.get('herald_diff', 0) + game_state.get('enemy_heralds', 0), 
                   game_state.get('enemy_heralds', 0)),
        "turrets": (game_state.get('turret_diff', 0) + game_state.get('enemy_turrets', 0), 
                   game_state.get('enemy_turrets', 0)),
        "inhibitors": (game_state.get('inhibitor_diff', 0) + game_state.get('enemy_inhibitors', 0), 
                      game_state.get('enemy_inhibitors', 0)),
        "has_elder_dragon": game_state.get('has_elder_dragon', False)
    }
    
    # Return comprehensive analysis
    return {
        "win_probability": win_probability,
        "win_percentage": f"{win_probability * 100:.1f}%",
        "game_state": game_state,
        "objectives": objectives,
        "objective_contributions": objective_contributions,
        "advice": advice,
        "game_phase": "Early Game" if game_state.get('game_time', 0) < 15 else 
                      "Mid Game" if game_state.get('game_time', 0) < 25 else "Late Game"
    }


###########################################
# PART 8: WEB APPLICATION
###########################################

def get_champion_roles():
    """Get mapping of champions to their roles"""
    return {
        # ADC/Marksman champions
        'Aphelios': 'BOTTOM', 'Ashe': 'BOTTOM', 'Caitlyn': 'BOTTOM', 'Corki': 'BOTTOM', 'Draven': 'BOTTOM',
        'Ezreal': 'BOTTOM', 'Jhin': 'BOTTOM', 'Jinx': 'BOTTOM', 'Kaisa': 'BOTTOM', 'Kalista': 'BOTTOM', 
        'KogMaw': 'BOTTOM', 'Lucian': 'BOTTOM', 'MissFortune': 'BOTTOM', 'Mel': 'BOTTOM', 'Nilah': 'BOTTOM',
        'Samira': 'BOTTOM', 'Senna': 'BOTTOM', 'Seraphine': 'BOTTOM', 'Smolder': 'BOTTOM', 'Sivir': 'BOTTOM', 
        'Tristana': 'BOTTOM', 'Twitch': 'BOTTOM', 'Varus': 'BOTTOM', 'Vayne': 'BOTTOM', 'Xayah': 'BOTTOM', 
        'Zeri': 'BOTTOM',
        
        # Support champions
        'Alistar': 'UTILITY', 'Bard': 'UTILITY', 'Blitzcrank': 'UTILITY', 'Brand': 'UTILITY',
        'Braum': 'UTILITY', 'Janna': 'UTILITY', 'Karma': 'UTILITY', 'Leona': 'UTILITY',
        'Lulu': 'UTILITY', 'Lux': 'UTILITY', 'Mel': 'UTILITY', 'Morgana': 'UTILITY', 'Nami': 'UTILITY',
        'Nautilus': 'UTILITY', 'Neeko': 'UTILITY', 'Pyke': 'UTILITY', 'Rakan': 'UTILITY', 'Rell': 'UTILITY',
        'Renata': 'UTILITY', 'Seraphine': 'UTILITY', 'Sona': 'UTILITY', 'Soraka': 'UTILITY', 'Taric': 'UTILITY',
        'Thresh': 'UTILITY', 'Xerath': 'UTILITY', 'Yuumi': 'UTILITY', 'Zilean': 'UTILITY', 'Zyra': 'UTILITY',
        'Milio': 'UTILITY', 'Hwei': 'UTILITY',
        
        # Mid champions
        'Ahri': 'MIDDLE', 'Akali': 'MIDDLE', 'Akshan': 'MIDDLE', 'Anivia': 'MIDDLE', 'Annie': 'MIDDLE',
        'AurelionSol': 'MIDDLE', 'Aurora': 'MIDDLE', 'Azir': 'MIDDLE', 'Brand': 'MIDDLE', 'Cassiopeia': 'MIDDLE', 
        'Diana': 'MIDDLE', 'Ekko': 'MIDDLE', 'Fizz': 'MIDDLE', 'Galio': 'MIDDLE', 'Heimerdinger': 'MIDDLE','Irelia': 'MIDDLE',
        'Jayce': 'MIDDLE', 'Kassadin': 'MIDDLE', 'Katarina': 'MIDDLE', 'Leblanc': 'MIDDLE', 'Lissandra': 'MIDDLE', 'Malzahar': 'MIDDLE', 
        'Mel': 'MIDDLE', 'Naafiri': 'MIDDLE', 'Orianna': 'MIDDLE', 'Panthom': 'MIDDLE', 'Qiyana': 'MIDDLE', 
        'Ryze': 'MIDDLE', 'Smolder': 'MIDDLE', 'Swain': 'MIDDLE', 'Sylas': 'MIDDLE', 'Syndra': 'MIDDLE',
        'Taliyah': 'MIDDLE', 'Talon': 'MIDDLE', 'Tristana': 'MIDDLE', 'TwistedFate': 'MIDDLE', 'Veigar': 'MIDDLE', 
        'Vex': 'MIDDLE', 'Viktor': 'MIDDLE', 'Vladimir': 'MIDDLE', 'Xerath': 'MIDDLE', 'Yasuo': 'MIDDLE', 
        'Yone': 'MIDDLE', 'Zed': 'MIDDLE', 'Zoe': 'MIDDLE', 'Neeko': 'MIDDLE', 
        
        # Jungle champions
        'Amumu': 'JUNGLE', 'Belveth': 'JUNGLE', 'Brand': 'JUNGLE', 'Briar': 'JUNGLE', 'Diana': 'JUNGLE', 
        'Ekko': 'JUNGLE', 'Elise': 'JUNGLE', 'Evelynn': 'JUNGLE', 'Fiddlesticks': 'JUNGLE', 'Gragas': 'JUNGLE', 
        'Graves': 'JUNGLE', 'Hecarim': 'JUNGLE', 'Ivern': 'JUNGLE', 'JarvanIV': 'JUNGLE', 'Karthus': 'JUNGLE',
        'Kayn': 'JUNGLE', 'Khazix': 'JUNGLE', 'Kindred': 'JUNGLE', 'LeeSin': 'JUNGLE', 'Lillia': 'JUNGLE',
        'MasterYi': 'JUNGLE', 'Naafiri': 'JUNGLE', 'Nidalee': 'JUNGLE', 'Nocturne': 'JUNGLE', 'Nunu': 'JUNGLE', 'Olaf': 'JUNGLE',
        'Poppy': 'JUNGLE', 'Rammus': 'JUNGLE', 'RekSai': 'JUNGLE', 'Rengar': 'JUNGLE', 'Sejuani': 'JUNGLE',
        'Shaco': 'JUNGLE', 'Shyvana': 'JUNGLE', 'Skarner': 'JUNGLE', 'Talon': 'JUNGLE', 'Trundle': 'JUNGLE', 
        'Udyr': 'JUNGLE', 'Vi': 'JUNGLE', 'Viego': 'JUNGLE', 'Volibear': 'JUNGLE', 'Warwick': 'JUNGLE', 
        'XinZhao': 'JUNGLE', 'Zac': 'JUNGLE', 'Zyra': 'JUNGLE',
        
        # Top champions
        'Aatrox': 'TOP', 'Akshan': 'TOP', 'Aurora': 'TOP', 'Ambessa': 'TOP', 'Camille': 'TOP', 'Chogath': 'TOP', 'Darius': 'TOP',
        'DrMundo': 'TOP', 'Fiora': 'TOP', 'Galio': 'TOP', 'Gangplank': 'TOP', 'Garen': 'TOP', 'Gnar': 'TOP', 
        'Gragas': 'TOP', 'Gwen': 'TOP', 'Heimerdinger': 'TOP', 'Illaoi': 'TOP', 'Irelia': 'TOP', 'Jax': 'TOP', 
        'Jayce': 'TOP', 'Kayle': 'TOP', 'Kennen': 'TOP', 'Kled': 'TOP', 'KSante': 'TOP', 'Malphite': 'TOP',
        'Maokai': 'TOP', 'Mordekaiser': 'TOP', 'Nasus': 'TOP', 'Ornn': 'TOP', 'Pantheon': 'TOP',
        'Quinn': 'TOP', 'Renekton': 'TOP', 'Riven': 'TOP', 'Rumble': 'TOP', 'Sett': 'TOP',
        'Shen': 'TOP', 'Singed': 'TOP', 'Sion': 'TOP', 'Swain': 'TOP', 'Teemo': 'TOP', 'Tryndamere': 'TOP',
        'TwistedFate': 'TOP', 'Urgot': 'TOP', 'Vladimir': 'TOP', 'Volibear': 'TOP', 
        'Wukong': 'TOP', 'Yorick': 'TOP', 'Yasuo': 'TOP', 'Yone': 'TOP'
    }

def create_win_prediction_app():
    """Create Gradio interface for League of Legends win prediction and coaching"""
    # Constants
    API_KEY_INFO = "Enter your Riot API key. If you don't have one, get it from https://developer.riotgames.com/"
    REGIONS = {
        "BR": "br1",
        "EUNE": "eun1",
        "EUW": "euw1",
        "JP": "jp1",
        "KR": "kr",
        "LAN": "la1",
        "LAS": "la2",
        "NA": "na1",
        "OCE": "oc1",
        "TR": "tr1",
        "RU": "ru"
    }
    
    # Load model and objective weights if available
    try:
        with open('lol_win_prediction_model.pkl', 'rb') as f:
            data = pickle.load(f)
            weights = data.get('objective_weights', None)
    except Exception as e:
        print(f"Error loading weights: {e}")
        weights = None
    
    # API Interaction Functions
    def get_summoner_by_riot_id(game_name, tag_line, region, api_key):
        """Get summoner information from Riot ID"""
        try:
            # Get account info
            account_url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}?api_key={api_key}"
            account_response = requests.get(account_url)
            
            if account_response.status_code != 200:
                return None, f"Error: {account_response.status_code} - {account_response.text}"
            
            account_data = account_response.json()
            puuid = account_data.get('puuid')
            
            # Get summoner info using PUUID
            summoner_url = f"https://{REGIONS[region]}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}?api_key={api_key}"
            summoner_response = requests.get(summoner_url)
            
            if summoner_response.status_code != 200:
                return None, f"Error: {summoner_response.status_code} - {summoner_response.text}"
            
            return summoner_response.json(), None
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_ranked_stats(summoner_id, region, api_key):
        """Get ranked stats for a summoner"""
        try:
            ranked_url = f"https://{REGIONS[region]}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}?api_key={api_key}"
            response = requests.get(ranked_url)
            
            if response.status_code != 200:
                return None, f"Error: {response.status_code} - {response.text}"
            
            return response.json(), None
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_match_history(puuid, region, api_key, count=10):
        """Get match history for a player"""
        try:
            # Map region to regional routing value
            region_mapping = {
                "BR": "americas",
                "NA": "americas",
                "LAN": "americas",
                "LAS": "americas",
                "EUNE": "europe",
                "EUW": "europe",
                "TR": "europe",
                "RU": "europe",
                "KR": "asia",
                "JP": "asia",
                "OCE": "sea"
            }
            
            matches_url = f"https://{region_mapping[region]}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}&api_key={api_key}"
            response = requests.get(matches_url)
            
            if response.status_code != 200:
                return None, f"Error: {response.status_code} - {response.text}"
            
            return response.json(), None
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_match_details(match_id, region, api_key):
        """Get details for a specific match"""
        try:
            # Map region to regional routing value
            region_mapping = {
                "BR": "americas",
                "NA": "americas",
                "LAN": "americas",
                "LAS": "americas",
                "EUNE": "europe",
                "EUW": "europe",
                "TR": "europe",
                "RU": "europe",
                "KR": "asia",
                "JP": "asia",
                "OCE": "sea"
            }
            
            match_url = f"https://{region_mapping[region]}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}"
            response = requests.get(match_url)
            
            if response.status_code != 200:
                return None, f"Error: {response.status_code} - {response.text}"
            
            return response.json(), None
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_live_game_data():
        """Get data from the Live Client Data API (during an active game)"""
        try:
            response = requests.get("https://127.0.0.1:2999/liveclientdata/allgamedata", verify=False, timeout=5)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"Error: {response.status_code} - Not in an active game or client API not accessible"
        except requests.exceptions.RequestException:
            return None, "Error connecting to the League client. Make sure you're in an active game."
    
    def check_client_connection():

      try:
        response = requests.get("https://127.0.0.1:2999/liveclientdata/allgamedata", 
                               verify=False, timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:100]}...")  # Print first 100 chars
        return True
      except Exception as e:
        print(f"Error connecting to client: {str(e)}")
        return False
    # Gradio UI Functions
    def search_player(game_name, tag_line, region, api_key):
        """Search for a player and display their stats"""
        if not game_name or not tag_line or not api_key:
            return "Please fill in all fields."
        
        # Get summoner info
        summoner_data, error = get_summoner_by_riot_id(game_name, tag_line, region, api_key)
        if error:
            return error
        
        # Get ranked stats
        ranked_data, error = get_ranked_stats(summoner_data['id'], region, api_key)
        if error:
            return error
        
        # Get match history
        match_ids, error = get_match_history(summoner_data['puuid'], region, api_key, count=5)
        if error:
            return error
        
        # Format ranked data
        ranked_info = "Unranked"
        if ranked_data:
            for queue in ranked_data:
                if queue['queueType'] == 'RANKED_SOLO_5x5':
                    ranked_info = f"{queue['tier']} {queue['rank']} {queue['leaguePoints']} LP\nWins: {queue['wins']} Losses: {queue['losses']} ({round(queue['wins']/(queue['wins'] + queue['losses'])*100)}% WR)"
                    break
        
        # Format recent matches
        recent_matches = ""
        for i, match_id in enumerate(match_ids[:5]):
            match_data, error = get_match_details(match_id, region, api_key)
            if error:
                recent_matches += f"Match {i+1}: Error retrieving data\n"
                continue
            
            # Find player in match
            player_data = None
            for participant in match_data['info']['participants']:
                if participant['puuid'] == summoner_data['puuid']:
                    player_data = participant
                    break
            
            if player_data:
                result = "Victory" if player_data['win'] else "Defeat"
                champion = player_data['championName']
                kills = player_data['kills']
                deaths = player_data['deaths']
                assists = player_data['assists']
                
                recent_matches += f"Match {i+1}: {result} - {champion} - {kills}/{deaths}/{assists} KDA\n"
            else:
                recent_matches += f"Match {i+1}: Player data not found\n"
        
        # Format result
        result = f"""
## Summoner Profile: {game_name}#{tag_line}
**Level:** {summoner_data['summonerLevel']}
**Region:** {region}

### Ranked Status (Solo/Duo)
{ranked_info}

### Recent Matches
{recent_matches}
        """
        
        return result
    
    def check_live_game(api_key):
      """Check if the player is in a live game and provide coaching with win probability graph"""
      try:
          # Set a longer timeout and print debug info
          response = requests.get("https://127.0.0.1:2999/liveclientdata/allgamedata", 
                                verify=False, timeout=20)
          
          print(f"Status code: {response.status_code}")
          print(f"Response received, length: {len(response.text)}")
          
          if response.status_code == 200:
              live_data = response.json()
              
              # Generate coaching advice with our AI coach
              analysis = get_objective_analysis(live_data, weights)
              
              if 'error' in analysis:
                  return analysis['error'], None
              
              # Create win probability graph
              graph = create_win_probability_graph(analysis['game_state'], weights)
              
              # Format the result
              game_time_min = int(analysis['game_state']['game_time'] // 60)
              game_time_sec = int(analysis['game_state']['game_time'] % 60)
              
              # Generate objective contribution section
              contributions = []
              for obj, value in analysis.get('objective_contributions', {}).items():
                  if value != 0:
                      contributions.append(f"- {obj.capitalize()}: {value*100:.2f}%")
              
              contribution_text = "\n".join(contributions) if contributions else "No objective advantages yet."
              
              result = f"""
  ## Live Game Analysis
  **Game Time:** {game_time_min}:{game_time_sec:02d} ({analysis['game_phase'].replace('_', ' ').title()})
  **Current Status:** {analysis['game_state'].get('kill_diff', 0)} kill difference {'(Winning)' if analysis['game_state'].get('kill_diff', 0) > 0 else '(Losing)' if analysis['game_state'].get('kill_diff', 0) < 0 else '(Even)'}

  ### Win Probability
  {analysis['win_percentage']} chance to win

  ### Objective Contributions to Win Probability
  {contribution_text}

  ### Coaching Advice
  """
              
              for i, advice in enumerate(analysis['advice']):
                  result += f"{i+1}. {advice}\n"
              
              return result, graph
          else:
              return f"Error: {response.status_code} - Not in an active game or client API not accessible", None
              
      except requests.exceptions.RequestException as e:      
          return f"Error connecting to the League client: {str(e)}\n\nMake sure:\n1. You're in an active game\n2. You've launched as administrator\n3. Your firewall isn't blocking connections", None
      except requests.exceptions.SSLError as e:
        return f"SSL Error connecting to the League client: {str(e)}\n\nTry running both League and this application as administrator.", None
      except requests.exceptions.ConnectionError as e:
        return f"Connection Error: {str(e)}\n\nMake sure you're in an active game and your firewall isn't blocking connections.", None
      except requests.exceptions.Timeout as e:
        return f"Timeout Error: {str(e)}\n\nThe connection to the League client timed out.", None
      except Exception as e:
        return f"Unexpected Error: {str(e)}", None
    
    def test_league_api_connection():
      """Test connection to the League Live Client Data API"""
      try:
          # Set SSL context to unverified
          import ssl
          ssl._create_default_https_context = ssl._create_unverified_context
          
          response = requests.get(
              "https://127.0.0.1:2999/liveclientdata/allgamedata", 
              verify=False,
              timeout=10
          )
          
          if response.status_code == 200:
              # Try to parse the JSON to confirm it's valid
              data = response.json()
              return True, "Successfully connected to League client API"
          else:
              return False, f"API responded with status code: {response.status_code}"
              
      except Exception as e:
          return False, f"Connection test failed: {str(e)}"

    def get_live_game_data_with_retry(max_retries=3, delay=2):
        """Get data from Live Client Data API with retries"""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    "https://127.0.0.1:2999/liveclientdata/allgamedata", 
                    verify=False, 
                    timeout=10
                )
                response.raise_for_status()
                return response.json(), None
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)  # Wait before retrying
                else:
                    return None, f"Failed to connect after {max_retries} attempts: {str(e)}"
    
    # Win probability graph function
    def create_win_probability_graph(game_state, weights=None):
        """
        Create a win probability graph based on game state and objectives
        """
        if weights is None:
            weights = load_weights()
        
        # Calculate base win probability
        win_prob = 0.5  # Start at 50%
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create data for the plot
        labels = ['Base']
        values = [0.5]
        colors = ['#b0b0b0']  # Gray for base
        descriptions = ['Starting probability: 50%']
        
        # Track the running probability
        running_prob = 0.5
        
        # Add dragon impact
        dragon_diff = game_state.get('dragon_diff', 0)
        if dragon_diff != 0:
            dragon_impact = dragon_diff * weights.get('dragon_weight', 0.0072)
            running_prob += dragon_impact
            
            labels.append('Dragons')
            values.append(running_prob)
            descriptions.append(f"{dragon_diff} dragon{'s' if abs(dragon_diff) > 1 else ''}: {dragon_impact*100:+.1f}%")
            colors.append('#3498db' if dragon_impact > 0 else '#e74c3c')  # Blue positive, red negative
        
        # Add elder dragon impact
        if game_state.get('has_elder_dragon', False):
            elder_impact = weights.get('elder_dragon_weight', 0.0144)
            running_prob += elder_impact
            
            labels.append('Elder')
            values.append(running_prob)
            descriptions.append(f"Elder dragon: {elder_impact*100:+.1f}%")
            colors.append('#9b59b6')  # Purple for elder
        
        # Add baron impact
        baron_diff = game_state.get('baron_diff', 0)
        if baron_diff != 0:
            baron_impact = baron_diff * weights.get('baron_weight', 0.0147)
            running_prob += baron_impact
            
            labels.append('Barons')
            values.append(running_prob)
            descriptions.append(f"{baron_diff} baron{'s' if abs(baron_diff) > 1 else ''}: {baron_impact*100:+.1f}%")
            colors.append('#3498db' if baron_impact > 0 else '#e74c3c')
        
        # Add herald impact
        herald_diff = game_state.get('herald_diff', 0)
        if herald_diff != 0:
            herald_impact = herald_diff * weights.get('herald_weight', 0.0072)
            running_prob += herald_impact
            
            labels.append('Heralds')
            values.append(running_prob)
            descriptions.append(f"{herald_diff} herald{'s' if abs(herald_diff) > 1 else ''}: {herald_impact*100:+.1f}%")
            colors.append('#3498db' if herald_impact > 0 else '#e74c3c')
        
        # Add turret impact
        turret_diff = game_state.get('turret_diff', 0)
        if turret_diff != 0:
            turret_impact = turret_diff * weights.get('turret_weight', 0.0204)
            running_prob += turret_impact
            
            labels.append('Turrets')
            values.append(running_prob)
            descriptions.append(f"{turret_diff} turret{'s' if abs(turret_diff) > 1 else ''}: {turret_impact*100:+.1f}%")
            colors.append('#3498db' if turret_impact > 0 else '#e74c3c')
        
        # Add inhibitor impact
        inhibitor_diff = game_state.get('inhibitor_diff', 0)
        if inhibitor_diff != 0:
            inhibitor_impact = inhibitor_diff * weights.get('inhibitor_weight', 0.0408)
            running_prob += inhibitor_impact
            
            labels.append('Inhibitors')
            values.append(running_prob)
            descriptions.append(f"{inhibitor_diff} inhibitor{'s' if abs(inhibitor_diff) > 1 else ''}: {inhibitor_impact*100:+.1f}%")
            colors.append('#3498db' if inhibitor_impact > 0 else '#e74c3c')
        
        # Add gold impact
        gold_diff = game_state.get('gold_diff', 0)
        if abs(gold_diff) > 1000:  # Only if gold diff is significant
            gold_diff_k = gold_diff / 1000
            gold_impact = gold_diff_k * weights.get('gold_diff_per_1k_weight', 0.0001)
            running_prob += gold_impact
            
            labels.append('Gold')
            values.append(running_prob)
            descriptions.append(f"{gold_diff_k:.1f}k gold: {gold_impact*100:+.1f}%")
            colors.append('#3498db' if gold_impact > 0 else '#e74c3c')
        
        # Apply game time factor
        game_time = game_state.get('game_time', 15)
        if game_time < 15:
            time_factor = 0.7
            factor_desc = "Early game: x0.7"
        elif game_time < 25:
            time_factor = 1.0
            factor_desc = "Mid game: x1.0"
        else:
            time_factor = 1.3
            factor_desc = "Late game: x1.3"
        
        time_adjusted = 0.5 + (running_prob - 0.5) * time_factor
        
        labels.append('Time Factor')
        values.append(time_adjusted)
        descriptions.append(factor_desc)
        colors.append('#f39c12')  # Orange for time factor
        
        # Final clamped probability
        final_prob = max(0.0, min(1.0, time_adjusted))
        
        labels.append('Final')
        values.append(final_prob)
        descriptions.append(f"Final win probability: {final_prob*100:.1f}%")
        colors.append('#2ecc71' if final_prob > 0.5 else '#e74c3c')  # Green if winning, red if losing
        
        # Plot bars
        positions = np.arange(len(labels))
        bars = ax.bar(positions, [v-0.5 for v in values], bottom=0.5, color=colors, width=0.7)
        
        # Add labels and title
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Win Probability')
        ax.set_title(f'Win Probability: {final_prob*100:.1f}%', fontsize=16, fontweight='bold')
        
        # Add win probability line
        ax.plot(positions, values, 'o-', color='black', linewidth=2, markersize=6)
        
        # Add horizontal line at 50%
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # Add text descriptions
        for i, (bar, desc) in enumerate(zip(bars, descriptions)):
            height = bar.get_height() + 0.5
            if height > 0.5:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.01
            
            if i > 0:  # Skip the base probability
                ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                        desc, ha='center', va=va, fontsize=9, rotation=0)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Add winning/losing areas
        ax.axhspan(0.5, 1.0, alpha=0.1, color='green')
        ax.axhspan(0, 0.5, alpha=0.1, color='red')
        
        # Add a title for the game status
        game_phase = "Early Game" if game_time < 15 else "Mid Game" if game_time < 25 else "Late Game"
        game_status = f"{game_phase} | {int(game_time//60):02d}:{int(game_time%60):02d}"
        plt.suptitle(game_status, fontsize=12)
        
        plt.tight_layout()
        
        return fig

    def analyze_match(match_id, region, api_key):
        """Analyze a completed match"""
        match_data, error = get_match_details(match_id, region, api_key)
        if error:
            return error
        
        # Extract match info
        match_info = match_data['info']
        game_mode = match_info.get('gameMode', 'Unknown')
        game_duration = match_info.get('gameDuration', 0) / 60  # convert to minutes
        
        # Team stats
        team1 = match_info['teams'][0]
        team2 = match_info['teams'][1]
        
        team1_win = team1['win']
        team2_win = team2['win']
        
        team1_kills = sum([p['kills'] for p in match_info['participants'] if p['teamId'] == team1['teamId']])
        team2_kills = sum([p['kills'] for p in match_info['participants'] if p['teamId'] == team2['teamId']])
        
        # Get objectives
        team1_objectives = team1['objectives']
        team2_objectives = team2['objectives']
        
        # Format result
        result = f"""
## Match Analysis - {match_id}
**Game Mode:** {game_mode}
**Duration:** {int(game_duration)} minutes

### Team Results
**Blue Team:** {team1_kills} kills - {"Victory" if team1_win else "Defeat"}
**Red Team:** {team2_kills} kills - {"Victory" if team2_win else "Defeat"}

### Objectives
**Blue Team:**
- Towers: {team1_objectives['tower']['kills']}
- Dragons: {team1_objectives['dragon']['kills']}
- Barons: {team1_objectives['baron']['kills']}

**Red Team:**
- Towers: {team2_objectives['tower']['kills']}
- Dragons: {team2_objectives['dragon']['kills']}
- Barons: {team2_objectives['baron']['kills']}

### Player Performance
"""
        
        # Add player stats
        for i, participant in enumerate(match_info['participants']):
            team = "Blue" if participant['teamId'] == team1['teamId'] else "Red"
            champ = participant['championName']
            player_name = participant.get('riotIdGameName', participant.get('summonerName', 'Unknown'))
            tag_line = participant.get('riotIdTagline', '')
            player_id = f"{player_name}#{tag_line}" if tag_line else player_name
            
            kills = participant['kills']
            deaths = participant['deaths']
            assists = participant['assists']
            cs = participant['totalMinionsKilled'] + participant.get('neutralMinionsKilled', 0)
            gold = participant['goldEarned']
            dmg = participant['totalDamageDealtToChampions']
            
            result += f"{i+1}. **{player_id}** ({team}) - {champ}\n"
            result += f"   KDA: {kills}/{deaths}/{assists}, CS: {cs}, Gold: {gold}, Damage: {dmg}\n\n"
        
        return result
    
    # Create interface
    with gr.Blocks(title="League of Legends Win Prediction & AI Coaching") as app:
        gr.Markdown("# LoL Win Prediction & AI Coaching")
        gr.Markdown("This app helps predict win probability and provides AI coaching advice for League of Legends games.")
        
        with gr.Tab("Player Search"):
            gr.Markdown("### Search for a player to view their stats")
            
            with gr.Row():
                with gr.Column():
                    game_name = gr.Textbox(label="Game Name")
                    tag_line = gr.Textbox(label="Tag Line")
                    region = gr.Dropdown(choices=list(REGIONS.keys()), label="Region", value="NA")
                    api_key = gr.Textbox(label="Riot API Key", placeholder=API_KEY_INFO, type="password")
                    search_button = gr.Button("Search Player")
                
                with gr.Column():
                    player_result = gr.Markdown("Player information will appear here")
            
            search_button.click(search_player, inputs=[game_name, tag_line, region, api_key], outputs=player_result)
        
        with gr.Tab("Live Game Coaching"):
          gr.Markdown("### Get coaching advice for your current game")
          gr.Markdown("This tab connects to your League client to provide real-time coaching advice while you're in a game.")
          
          with gr.Row():
              with gr.Column():
                  test_connection_button = gr.Button("Test API Connection")
                  connection_status = gr.Markdown("Connection status unknown")

                  live_api_key = gr.Textbox(label="Riot API Key", placeholder=API_KEY_INFO, type="password")
                  check_game_button = gr.Button("Check Live Game")
              
              with gr.Column():
                  live_game_result = gr.Markdown("Coaching advice will appear here when you're in a game")
                  win_prob_graph = gr.Plot(label="Win Probability Graph")

          test_connection_button.click(
              lambda: test_league_api_connection()[1],
              inputs=[], 
              outputs=connection_status
          )
          check_game_button.click(
              check_live_game, 
              inputs=[live_api_key], 
              outputs=[live_game_result, win_prob_graph]
          )
          
          gr.Markdown("Note: You must be in an active League of Legends game for this feature to work.")
        
        with gr.Tab("Match Analysis"):
            gr.Markdown("### Analyze a completed match")
            
            with gr.Row():
                with gr.Column():
                    match_id = gr.Textbox(label="Match ID")
                    match_region = gr.Dropdown(choices=list(REGIONS.keys()), label="Region", value="NA")
                    match_api_key = gr.Textbox(label="Riot API Key", placeholder=API_KEY_INFO, type="password")
                    analyze_button = gr.Button("Analyze Match")
                
                with gr.Column():
                    match_result = gr.Markdown("Match analysis will appear here")
            
            analyze_button.click(analyze_match, inputs=[match_id, match_region, match_api_key], outputs=match_result)
        
        with gr.Tab("About"):
            gr.Markdown("""
            # About LoL Win Prediction & AI Coaching
            
            This application provides win prediction and AI coaching for League of Legends players. It uses machine learning techniques to analyze game data and provide personalized advice.
            
            ## Features
            
            - **Player Stats**: Look up player statistics including rank, win rate, and recent match history
            - **Live Game Coaching**: Get real-time coaching advice when you're in a game
            - **Match Analysis**: Analyze completed matches to understand what went well and what could be improved
            - **Win Prediction**: Predictive analytics to estimate your chance of winning
            
            ## How It Works
            
            The system uses a neural network trained on over 10,000 matches from both solo queue and professional play. By analyzing patterns in game data, the model can predict win probability with high accuracy.
            
            The AI coaching provides customized advice based on:
            
            - Game phase (early, mid, late game)
            - Your role and champion
            - Current game state and performance metrics
            - Historical win patterns in similar situations
            
            ## Weight Importance
            
            Our statistical analysis identified these objective weights:
            - Inhibitors: 4.08% per inhibitor
            - Turrets: 2.04% per turret
            - Baron: 1.47% per baron
            - Elder Dragon: 1.44% per elder dragon
            - Regular Dragons: 0.72% per dragon
            
            ## Credits
            
            This application was created by combining machine learning with League of Legends data provided through the Riot Games API.
            
            Data sources:
            - Riot Games API
            - Professional match data from Oracle's Elixir
            - League of Legends Live Client Data API
            """)
    
    return app

###########################################
# PART 9: MAIN ENTRY POINT
###########################################

def main():
    """Main function to run the LoL win prediction workflow"""
    import os
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define data path - adjust this to your actual data path
    data_path = "processed_match_data.csv"
    
    print("=== League of Legends Win Prediction & AI Coaching System ===")
    print(f"Loading data from {data_path}")
    
    try:
        '''# Run the workflow
        print("\nRunning win prediction workflow...")
        #retrain
        model = train_evaluate_models(pd.read_csv(data_path))
        
        # Generate the objective weights
        print("\nCalculating objective weights...")
        weights = calculate_objective_weights(data_path)'''
        
        # Create and launch the web app
        print("\nStarting web application...")
        app = create_win_prediction_app()
        app.launch(share=True, debug=True)
        
        print("\nApplication running successfully!")
        
    except Exception as e:
        print(f"Error running the application: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
