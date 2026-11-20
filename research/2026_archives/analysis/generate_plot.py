#!/usr/bin/env python3
"""
BeeMonitor Events + Weather Figure Generator

Weather data source: Weather Underground Personal Weather Station (PWS)
    Station ID: KPABELLE49 (Grove Park, Bellefonte, PA)
    Location: 40.88°N, 77.83°W, Elevation 314m
    URL: https://www.wunderground.com/dashboard/pws/KPABELLE49/table/2024-04-30/2024-04-30/monthly

This PWS is ~10km from the bee hotel site and captured precipitation events
that the airport ASOS station (KUNV) missed.

Usage:
    python plot_beemonitor_weather.py

Inputs:
    - Hardcoded daily events data (from aggregate_events.py output)
    - Hardcoded PWS weather data (from Weather Underground)

Outputs:
    - beemonitor_events_weather.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy import stats


# =============================================================================
# DATA: Daily events from BeeMonitor batch processing
# Source: aggregate_events.py -> daily_summary.csv
# =============================================================================
DAILY_EVENTS = """video_date,n_videos,n_events
2024-04-16,5,20
2024-04-17,10,35
2024-04-18,32,170
2024-04-20,28,211
2024-04-22,46,300
2024-04-23,50,412
2024-04-24,34,438
2024-04-25,42,219
2024-04-26,49,316
2024-04-27,2,4
2024-04-28,60,815
2024-04-29,60,885
2024-04-30,45,741
2024-05-01,57,1123
2024-05-02,21,360
2024-05-06,14,598
2024-05-07,60,1960
2024-05-08,58,1296
2024-05-09,38,235
2024-05-11,27,445
2024-05-12,2,21"""


# =============================================================================
# DATA: Weather from Weather Underground PWS KPABELLE49
# Source: https://www.wunderground.com/dashboard/pws/KPABELLE49
# Downloaded: April-May 2024 monthly history tables
# =============================================================================
PWS_WEATHER = """date,temp_max,temp_avg,temp_min,precip_mm
2024-04-16,24.8,13.7,2.0,0.00
2024-04-17,18.7,12.8,7.5,9.91
2024-04-18,19.9,15.1,9.4,2.31
2024-04-19,12.4,10.4,7.4,0.00
2024-04-20,16.4,10.7,3.2,0.00
2024-04-21,8.2,3.9,-1.7,0.00
2024-04-22,16.4,8.0,-2.5,0.00
2024-04-23,21.1,11.3,-0.1,0.00
2024-04-24,17.1,12.5,7.5,0.71
2024-04-25,14.8,7.8,2.9,0.00
2024-04-26,18.8,10.3,-0.5,0.00
2024-04-27,11.9,9.6,7.8,2.31
2024-04-28,28.9,18.3,9.6,0.30
2024-04-29,30.1,21.5,11.6,0.00
2024-04-30,28.6,19.2,13.3,1.70
2024-05-01,27.4,19.0,9.6,0.30
2024-05-02,30.5,21.0,11.5,0.00
2024-05-03,25.0,17.9,8.8,0.00
2024-05-04,15.5,11.0,9.7,11.40
2024-05-05,14.7,11.4,8.7,6.60
2024-05-06,25.2,17.1,12.2,0.30
2024-05-07,25.4,18.5,13.8,0.00
2024-05-08,27.5,21.8,16.2,2.31
2024-05-09,18.5,13.3,9.5,13.89
2024-05-10,10.7,9.6,8.7,28.19
2024-05-11,16.4,10.4,7.9,8.71
2024-05-12,16.5,11.4,7.6,3.51"""


def load_and_merge_data():
    """Load events and weather data, merge on date."""
    daily = pd.read_csv(StringIO(DAILY_EVENTS), parse_dates=['video_date'])
    weather = pd.read_csv(StringIO(PWS_WEATHER), parse_dates=['date'])
    
    # Create complete date range
    all_dates = pd.date_range(start='2024-04-16', end='2024-05-12', freq='D')
    df = pd.DataFrame({'date': all_dates})
    
    # Merge events (rename column for consistency)
    df = df.merge(daily.rename(columns={'video_date': 'date'}), on='date', how='left')
    df = df.merge(weather, on='date', how='left')
    
    # Data quality flags
    # - 'missing': equipment failure (NaN events) - EXCLUDED from analysis
    # - 'partial': <20 videos - EXCLUDED from analysis  
    # - 'good': ≥20 videos - INCLUDED in analysis
    df['quality'] = 'good'
    df.loc[df['n_videos'].isna(), 'quality'] = 'missing'
    df.loc[(df['n_videos'] > 0) & (df['n_videos'] < 20), 'quality'] = 'partial'
    
    return df


def plot_events_weather(df, output_path='beemonitor_events_weather.png'):
    """Create 4-panel figure showing events and weather."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('BeeMonitor Foraging Activity & Weather\n'
                 'State College, PA — April 16 - May 12, 2024', 
                 fontsize=14, fontweight='bold')
    
    x = df['date']
    
    # =========================================================================
    # Panel 1: Daily Events
    # =========================================================================
    ax1 = axes[0]
    
    for i, row in df.iterrows():
        if row['quality'] == 'missing':
            # Equipment failure - hatched bar
            ax1.bar(row['date'], 200, color='white', edgecolor='gray', 
                    hatch='///', alpha=0.5, linewidth=1.5)
        elif row['quality'] == 'partial':
            ax1.bar(row['date'], row['n_events'], color='orange', 
                    edgecolor='black', alpha=0.8)
        else:
            ax1.bar(row['date'], row['n_events'], color='steelblue', 
                    edgecolor='black', alpha=0.8)
    
    # Label equipment failure days
    for i, row in df.iterrows():
        if row['quality'] == 'missing':
            ax1.annotate('Equipment\nFailure', (row['date'], 100), ha='center', 
                         fontsize=7, color='gray', style='italic')
    
    ax1.set_ylabel('Daily Events', fontsize=11)
    ax1.set_title('Foraging Events', fontsize=11)
    ax1.set_ylim(0, df['n_events'].max() * 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Full data (≥20 videos)'),
        Patch(facecolor='orange', edgecolor='black', label='Partial data (<20 videos)'),
        Patch(facecolor='white', edgecolor='gray', hatch='///', label='Equipment failure (excluded)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # =========================================================================
    # Panel 2: Temperature
    # =========================================================================
    ax2 = axes[1]
    ax2.fill_between(x, df['temp_min'], df['temp_max'], alpha=0.3, color='red', 
                     label='Min-Max range')
    ax2.plot(x, df['temp_max'], 'r.-', linewidth=2, markersize=6, label='Max temp')
    ax2.plot(x, df['temp_avg'], 'k.-', linewidth=1.5, markersize=4, label='Avg temp')
    ax2.axhline(y=15, color='blue', linestyle='--', alpha=0.5, label='15°C threshold')
    
    # Shade equipment failure days
    for i, row in df.iterrows():
        if row['quality'] == 'missing':
            ax2.axvspan(row['date'] - pd.Timedelta(hours=12), 
                        row['date'] + pd.Timedelta(hours=12), 
                        alpha=0.2, color='gray')
    
    ax2.set_ylabel('Temperature (°C)', fontsize=11)
    ax2.set_title('Daily Temperature (gray shading = equipment failure days)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel 3: Precipitation
    # =========================================================================
    ax3 = axes[2]
    precip_colors = ['navy' if p > 0.5 else 'lightblue' for p in df['precip_mm']]
    ax3.bar(x, df['precip_mm'], color=precip_colors, edgecolor='black', alpha=0.8)
    
    # Shade equipment failure days
    for i, row in df.iterrows():
        if row['quality'] == 'missing':
            ax3.axvspan(row['date'] - pd.Timedelta(hours=12), 
                        row['date'] + pd.Timedelta(hours=12), 
                        alpha=0.2, color='gray')
    
    ax3.set_ylabel('Precipitation (mm)', fontsize=11)
    ax3.set_title('Daily Precipitation (dark blue = >0.5mm)', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # Annotate heavy rain days
    for i, row in df.iterrows():
        if row['precip_mm'] > 5:
            ax3.annotate(f"{row['precip_mm']:.1f}", (row['date'], row['precip_mm'] + 1), 
                         ha='center', fontsize=8, color='navy')
    
    # =========================================================================
    # Panel 4: Video Coverage
    # =========================================================================
    ax4 = axes[3]
    
    for i, row in df.iterrows():
        if row['quality'] == 'missing':
            ax4.bar(row['date'], 5, color='white', edgecolor='gray', 
                    hatch='///', alpha=0.5)
        else:
            ax4.bar(row['date'], row['n_videos'], color='forestgreen', 
                    edgecolor='black', alpha=0.8)
    
    ax4.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='Min threshold (20)')
    ax4.axhline(y=60, color='blue', linestyle='--', linewidth=1.5, label='Full coverage (60)')
    ax4.set_ylabel('Videos', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title('Daily Video Coverage', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    
    return fig


def print_analysis_summary(df):
    """Print correlation analysis summary."""
    
    # Filter to good data only
    good_data = df[df['quality'] == 'good'].copy()
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nData quality breakdown:")
    print(f"  Full data (≥20 videos):    {(df['quality'] == 'good').sum()} days - INCLUDED")
    print(f"  Partial data (<20 videos): {(df['quality'] == 'partial').sum()} days - EXCLUDED")
    print(f"  Equipment failure:         {(df['quality'] == 'missing').sum()} days - EXCLUDED")
    
    print(f"\nCorrelation analysis (n={len(good_data)} days):")
    
    r_temp, p_temp = stats.pearsonr(good_data['temp_max'], good_data['n_events'])
    print(f"  Temperature vs Events: r = {r_temp:.3f}, p = {p_temp:.4f}")
    
    if good_data['precip_mm'].std() > 0:
        r_precip, p_precip = stats.pearsonr(good_data['precip_mm'], good_data['n_events'])
        print(f"  Precipitation vs Events: r = {r_precip:.3f}, p = {p_precip:.4f}")
    
    print(f"\nWeather summary (analysis days only):")
    print(f"  Rain days (>0.5mm): {(good_data['precip_mm'] > 0.5).sum()}")
    print(f"  Dry days:           {(good_data['precip_mm'] <= 0.5).sum()}")
    print(f"  Temp range:         {good_data['temp_max'].min():.1f} - {good_data['temp_max'].max():.1f}°C")
    
    print(f"\nEquipment failure days (weather still recorded):")
    missing = df[df['quality'] == 'missing']
    for _, row in missing.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: "
              f"Temp={row['temp_max']:.1f}°C, Precip={row['precip_mm']:.1f}mm")


def main():
    # Load and merge data
    df = load_and_merge_data()
    
    print("Combined data:")
    print(df[['date', 'n_events', 'n_videos', 'temp_max', 'precip_mm', 'quality']].to_string(index=False))
    
    # Generate figure
    plot_events_weather(df, output_path='beemonitor_events_weather.png')
    
    # Print analysis summary
    print_analysis_summary(df)


if __name__ == '__main__':
    main()