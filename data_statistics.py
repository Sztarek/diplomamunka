import numpy as np
import pandas as pd
import os
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

def analyze_dataset(data_folder='dataset/'):
    """
    Analyze the dataset and provide comprehensive statistics.
    """
    print("Loading and analyzing dataset...")
    
    # Load all files
    all_files = os.listdir(data_folder)
    dfs = []
    total_peaks = 0
    total_files = len(all_files)
    
    # Collect basic statistics
    file_stats = {
        'total_files': total_files,
        'total_peaks': 0,
        'files_with_peaks': 0,
        'peaks_per_file': [],
        'arps_columns_per_file': [],
        'max_length': 0,
        'min_length': float('inf'),
        'total_length': 0
    }
    
    for file in all_files:
        with open(os.path.join(data_folder, file), 'rb') as f:
            df = pickle.load(f)
            df['is_peak'] = 0
            idx_list = []
            
            # Count ARPS columns
            arps_cols = [col for col in df.columns if col.startswith('arps')]
            file_stats['arps_columns_per_file'].append(len(arps_cols))
            
            # Find peaks
            for col in arps_cols:
                if (df[col] > 0).any():
                    idx_list.append(df[df[col] > 0].index[0])
            
            # Update peak statistics
            num_peaks = len(idx_list)
            if num_peaks > 0:
                file_stats['files_with_peaks'] += 1
                file_stats['peaks_per_file'].append(num_peaks)
                file_stats['total_peaks'] += num_peaks
            
            # Update length statistics
            file_length = len(df)
            file_stats['max_length'] = max(file_stats['max_length'], file_length)
            file_stats['min_length'] = min(file_stats['min_length'], file_length)
            file_stats['total_length'] += file_length
            
            for idx in idx_list:
                df.loc[idx, 'is_peak'] = 1
            dfs.append(df)
    
    # Calculate segment statistics
    window_size = 20
    stride = 15
    segment_stats = {
        'total_segments': 0,
        'segments_with_peaks': 0,
        'segments_without_peaks': 0,
        'peak_distribution': []
    }
    
    for df in dfs:
        # Create segments
        for i in range(0, len(df) - window_size + 1, stride):
            segment = df['is_peak'].values[i:i + window_size]
            segment_stats['total_segments'] += 1
            if any(segment):
                segment_stats['segments_with_peaks'] += 1
                segment_stats['peak_distribution'].append(sum(segment))
            else:
                segment_stats['segments_without_peaks'] += 1
    
    # Print comprehensive statistics
    print("\n=== Dataset Statistics ===")
    print(f"\nFile Statistics:")
    print(f"Total number of files: {file_stats['total_files']}")
    print(f"Files containing peaks: {file_stats['files_with_peaks']} ({(file_stats['files_with_peaks']/file_stats['total_files'])*100:.1f}%)")
    print(f"Total number of peaks: {file_stats['total_peaks']}")
    print(f"Average peaks per file: {np.mean(file_stats['peaks_per_file']):.2f}")
    print(f"Average ARPS columns per file: {np.mean(file_stats['arps_columns_per_file']):.2f}")
    
    print(f"\nTime Series Length Statistics:")
    print(f"Maximum length: {file_stats['max_length']}")
    print(f"Minimum length: {file_stats['min_length']}")
    print(f"Average length: {file_stats['total_length']/file_stats['total_files']:.2f}")
    
    print(f"\nSegment Statistics:")
    print(f"Total segments: {segment_stats['total_segments']}")
    print(f"Segments with peaks: {segment_stats['segments_with_peaks']} ({(segment_stats['segments_with_peaks']/segment_stats['total_segments'])*100:.1f}%)")
    print(f"Segments without peaks: {segment_stats['segments_without_peaks']} ({(segment_stats['segments_without_peaks']/segment_stats['total_segments'])*100:.1f}%)")
    
    if segment_stats['peak_distribution']:
        print(f"\nPeak Distribution in Segments:")
        print(f"Average peaks per segment: {np.mean(segment_stats['peak_distribution']):.2f}")
        print(f"Maximum peaks in a segment: {max(segment_stats['peak_distribution'])}")
        print(f"Minimum peaks in a segment: {min(segment_stats['peak_distribution'])}")
    
    # Create interactive visualizations using Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Csúcsok eloszlása fájlonként',
            'ARPS oszlopok eloszlása fájlonként',
            'Csúcsok eloszlása szegmensenként',
            'Szegmensek eloszlása'
        ),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "pie"}]]
    )
    
    # Convert numpy arrays to lists and ensure proper binning
    peaks_per_file = [int(x) for x in file_stats['peaks_per_file']]
    arps_columns = [int(x) for x in file_stats['arps_columns_per_file']]
    peak_distribution = [int(x) for x in segment_stats['peak_distribution']]
    
    # Plot 1: Distribution of peaks per file
    fig.add_trace(
        go.Histogram(
            x=peaks_per_file,
            name='Csúcsok fájlonként',
            nbinsx=int(max(peaks_per_file)) + 1 if peaks_per_file else 10
        ),
        row=1, col=1
    )
    
    # Plot 2: Distribution of ARPS columns per file
    fig.add_trace(
        go.Histogram(
            x=arps_columns,
            name='ARPS oszlopok fájlonként',
            nbinsx=int(max(arps_columns)) + 1 if arps_columns else 10
        ),
        row=1, col=2
    )
    
    # Plot 3: Distribution of peaks in segments
    fig.add_trace(
        go.Histogram(
            x=peak_distribution,
            name='Csúcsok szegmensenként',
            nbinsx=int(max(peak_distribution)) + 1 if peak_distribution else 10
        ),
        row=2, col=1
    )
    
    # Plot 4: Pie chart of segments with/without peaks
    fig.add_trace(
        go.Pie(
            labels=['Csúcsokkal', 'Csúcsok nélkül'],
            values=[segment_stats['segments_with_peaks'], segment_stats['segments_without_peaks']],
            name='Szegmensek eloszlása'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Adathalmaz Statisztikák",
        showlegend=False,
        font=dict(
            family="Arial",
            size=12
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Csúcsok száma", row=1, col=1)
    fig.update_yaxes(title_text="Fájlok száma", row=1, col=1)
    fig.update_xaxes(title_text="ARPS oszlopok száma", row=1, col=2)
    fig.update_yaxes(title_text="Fájlok száma", row=1, col=2)
    fig.update_xaxes(title_text="Csúcsok száma", row=2, col=1)
    fig.update_yaxes(title_text="Szegmensek száma", row=2, col=1)
    
    # Save the interactive plot as HTML
    fig.write_html("dataset_statistics.html")
    print("\nInteractive visualizations have been saved to 'dataset_statistics.html'")

if __name__ == "__main__":
    analyze_dataset() 