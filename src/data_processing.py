from pybaseball import statcast, playerid_lookup
import pandas as pd

def load_pitcher_data(start_date: str, end_date: str):
    # Get Statcast data
    df = statcast(start_dt=start_date, end_dt=end_date, player_type='pitcher')
    
    # Filter relevant pitches
    fastballs = ['FF', 'SI', 'FT']  # Four-seam, sinker, cutter
    df = df[df.pitch_type.isin(fastballs)]
    
    # Feature engineering
    df['release_diff'] = df['release_pos_x'].diff().abs()
    df['velo_delta'] = df['release_speed'] - df.groupby('pitcher')['release_speed'].transform('mean')
    
    # Aggregate per game
    agg_df = df.groupby(['game_date', 'pitcher']).agg(
        avg_velo=('release_speed', 'mean'),
        max_velo=('release_speed', 'max'),
        spin_diff=('release_spin_rate', lambda x: x.max() - x.min()),
        stress_pitches=('pitch_number', 'count')
    ).reset_index()
    
    return agg_df

def create_labels(df, threshold=1.5, window=30):
    """Create velocity drop labels"""
    df = df.sort_values(['pitcher', 'game_date'])
    df['future_velo'] = df.groupby('pitcher')['avg_velo'].shift(-window)
    df['velo_drop'] = (df['avg_velo'] - df['future_velo']) >= threshold
    return df.dropna(subset=['future_velo'])
