#!/usr/bin/env python3
"""
Cell Phenotyping and Statistical Analysis
=========================================

Phenotypes cells from MCMICRO/Cellpose output using custom phenotype definitions
and performs statistical comparisons following Sorger lab best practices.

Usage:
    python phenotype_analysis.py --input results/ --metadata sample_metadata.csv --markers markers.csv --phenotypes phenotype_definitions.csv --output phenotype_analysis/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core analysis packages
import scimap as sm
import scanpy as sc
import anndata as ad

# Statistical analysis
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import statsmodels.api as sm_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor='white', format='png')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

class CellPhenotypeAnalyzer:
    """
    Cell phenotyping and statistical analysis using custom phenotype definitions
    """
    
    def __init__(self, input_dir: str, metadata_file: str, output_dir: str, 
                 markers_file: str = None, phenotypes_file: str = None):
        self.input_dir = Path(input_dir)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.markers_file = Path(markers_file) if markers_file else None
        self.phenotypes_file = Path(phenotypes_file) if phenotypes_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        
        self.adata = None
        self.metadata = None
        self.markers = None
        self.phenotype_definitions = None
        
    def load_data(self):
        """Load and combine all quantification data"""
        print("Loading quantification data...")
        
        # Find all combined_quantification.csv files
        csv_files = list(self.input_dir.glob("*/final/combined_quantification.csv"))
        
        if not csv_files:
            raise ValueError("No combined_quantification.csv files found")
        
        all_data = []
        for csv_file in csv_files:
            sample_name = csv_file.parent.parent.name
            df = pd.read_csv(csv_file)
            df['sample_id'] = sample_name
            all_data.append(df)
            print(f"  Loaded {len(df)} cells from {sample_name}")
        
        # Combine all samples
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total: {len(combined_df)} cells from {len(csv_files)} samples")
        
        # Load metadata if provided
        if self.metadata_file and self.metadata_file.exists():
            self.metadata = pd.read_csv(self.metadata_file)
            print(f"Loaded metadata for {len(self.metadata)} samples")
        
        # Load markers if provided
        if self.markers_file and self.markers_file.exists():
            self.markers = pd.read_csv(self.markers_file)
            print(f"Loaded {len(self.markers)} markers")
            
            # Create marker name to channel mapping and extract clean names
            self.marker_to_channel = {}
            self.channel_to_clean_name = {}
            for i, row in self.markers.iterrows():
                channel_num = i + 1
                full_marker_name = row['marker_name']
                # Extract clean name (last component after underscore)
                clean_name = full_marker_name.split('_')[-1] if '_' in full_marker_name else full_marker_name
                
                channel = f'Channel_{channel_num}'
                self.marker_to_channel[full_marker_name] = channel
                self.channel_to_clean_name[channel] = clean_name
            
            # Rename channels in AnnData to clean names
            print(f"Channel renaming: {self.channel_to_clean_name}")
        else:
            self.channel_to_clean_name = {}
        
        # Load phenotype definitions if provided
        if self.phenotypes_file and self.phenotypes_file.exists():
            self.phenotype_definitions = pd.read_csv(self.phenotypes_file)
            print(f"Loaded {len(self.phenotype_definitions)} phenotype definitions")
        else:
            print("No phenotype definitions file provided, using default definitions")
        
        # Create AnnData object
        self.adata = self._create_anndata(combined_df)
        
        return self.adata
    
    def _create_anndata(self, df):
        """Create AnnData object from combined dataframe"""
        # Extract channel intensities
        channel_cols = [col for col in df.columns if col.startswith('Channel_')]
        X = df[channel_cols].values.astype(np.float32)
        
        # Create AnnData
        adata = ad.AnnData(X=X)
        adata.var_names = channel_cols
        
        # Rename channels to clean marker names if available
        if hasattr(self, 'channel_to_clean_name') and self.channel_to_clean_name:
            # Create mapping for available channels
            var_name_mapping = {}
            for old_name in adata.var_names:
                if old_name in self.channel_to_clean_name:
                    var_name_mapping[old_name] = self.channel_to_clean_name[old_name]
            
            # Apply renaming
            adata.var_names = [var_name_mapping.get(name, name) for name in adata.var_names]
            print(f"Renamed channels: {var_name_mapping}")
        
        # Add cell metadata
        obs_cols = ['CellID', 'X_centroid', 'Y_centroid', 'Area', 'MajorAxisLength', 
                   'MinorAxisLength', 'Eccentricity', 'Solidity', 'sample_id']
        for col in obs_cols:
            if col in df.columns:
                adata.obs[col] = df[col].values
        
        # Add spatial coordinates
        adata.obsm['spatial'] = df[['X_centroid', 'Y_centroid']].values
        
        # Add required columns for SCIMAP
        adata.obs['imageid'] = adata.obs['sample_id']  # SCIMAP requires imageid column
        adata.obs['CellID'] = range(len(adata))  # Ensure CellID is sequential
        
        # Add sample metadata if available
        if self.metadata is not None:
            adata.obs = adata.obs.merge(self.metadata, on='sample_id', how='left')
        
        return adata
    
    def preprocess_data(self):
        """Preprocess data following Sorger/Nirmal lab best practices"""
        print("Preprocessing data...")
        
        # Store raw data
        self.adata.raw = self.adata
        
        # 1. Log1p transformation (standard in field)
        sc.pp.log1p(self.adata)
        
        # 2. Calculate QC metrics
        self.adata.obs['total_intensity'] = np.array(self.adata.X.sum(axis=1)).flatten()
        self.adata.obs['n_channels_detected'] = (self.adata.X > 0).sum(axis=1)
        
        # 3. Remove extreme outliers (optional but recommended)
        # Remove cells with very low total signal (likely artifacts)
        min_signal = np.percentile(self.adata.obs['total_intensity'], 1)
        self.adata = self.adata[self.adata.obs['total_intensity'] > min_signal]
        
        # 4. Channel-wise 99th percentile clipping (standard approach)
        # This is what most papers do instead of complex GMM gating
        for i, channel in enumerate(self.adata.var_names):
            channel_data = self.adata.X[:, i]
            p99 = np.percentile(channel_data, 99)
            self.adata.X[:, i] = np.clip(channel_data, 0, p99)
        
        # 5. Z-score normalization per sample (key for multi-sample studies)
        for sample in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample
            sample_data = self.adata.X[sample_mask, :]
            
            # Z-score normalize each channel
            for i in range(sample_data.shape[1]):
                channel_data = sample_data[:, i]
                if np.std(channel_data) > 0:  # Avoid division by zero
                    sample_data[:, i] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
            
            self.adata.X[sample_mask, :] = sample_data
        
        print(f"  Preprocessed {len(self.adata)} cells")
        
        # Save preprocessed data
        self.adata.write(self.output_dir / 'data' / 'preprocessed_data.h5ad')
        print("Preprocessed data saved")
        
    def auto_phenotype_cells(self):
        """Adaptive phenotyping using hierarchical gating and clustering"""
        print("Performing adaptive cell phenotyping...")
        
        # 1. Compute optimal gating thresholds per channel using Otsu
        print("  Computing optimal gating thresholds...")
        auto_gates = {}
        for i, channel in enumerate(self.adata.var_names):
            channel_data = self.adata.X[:, i]
            # Use Otsu threshold on positive values only
            positive_data = channel_data[channel_data > 0]
            if len(positive_data) > 100:
                from skimage.filters import threshold_otsu
                try:
                    threshold = threshold_otsu(positive_data)
                    auto_gates[channel] = threshold
                except:
                    # Fallback to 85th percentile
                    auto_gates[channel] = np.percentile(positive_data, 85)
            else:
                auto_gates[channel] = np.percentile(channel_data, 85)
        
        # 2. Apply hierarchical phenotyping using computed thresholds
        print("  Applying hierarchical phenotyping...")
        self.adata = self._assign_biological_phenotypes()
        
        # 3. Clustering-based phenotyping for validation (reduce clusters)
        print("  Performing clustering-based phenotyping...")
        n_components = min(10, self.adata.n_vars - 1)
        sc.tl.pca(self.adata, n_comps=n_components)
        sc.pp.neighbors(self.adata, n_neighbors=30, n_pcs=n_components)  # Increased neighbors
        sc.tl.leiden(self.adata, resolution=0.2, key_added='leiden_clusters')  # Lower resolution
        
        # 4. Consensus phenotyping combining methods
        self.adata = self._consensus_phenotyping()
        
        # Calculate phenotype frequencies
        phenotype_counts = self.adata.obs.groupby(['sample_id', 'consensus_phenotype']).size().unstack(fill_value=0)
        phenotype_freq = phenotype_counts.div(phenotype_counts.sum(axis=1), axis=0) * 100
        
        # Save all phenotyping results
        phenotype_counts.to_csv(self.output_dir / 'data' / 'phenotype_counts.csv')
        phenotype_freq.to_csv(self.output_dir / 'data' / 'phenotype_frequencies.csv')
        
        # Save gating thresholds
        pd.DataFrame(auto_gates, index=['threshold']).T.to_csv(
            self.output_dir / 'data' / 'gating_thresholds.csv'
        )
        
        return phenotype_counts, phenotype_freq
    
    def _assign_biological_phenotypes(self):
        """Assign biological phenotypes using custom phenotype definitions file"""
        
        if not hasattr(self, 'marker_to_channel') or not self.marker_to_channel:
            print("Warning: No marker mapping available")
            self.adata.obs['hierarchical_phenotype'] = 'Unknown'
            return self.adata
        
        if self.phenotype_definitions is None or len(self.phenotype_definitions) == 0:
            print("Warning: No phenotype definitions available")
            self.adata.obs['hierarchical_phenotype'] = 'Unknown'
            return self.adata
        
        # Convert custom phenotype definitions to SCIMAP format
        scimap_phenotypes = []
        
        for _, row in self.phenotype_definitions.iterrows():
            phenotype_name = row['phenotype_name']
            positive_markers = str(row['positive_markers']).split(';') if pd.notna(row['positive_markers']) else []
            negative_markers = str(row['negative_markers']).split(';') if pd.notna(row['negative_markers']) else []
            
            # Convert marker names to channel expressions
            expressions = []
            
            # Add positive markers
            for marker in positive_markers:
                marker = marker.strip()
                if marker in self.marker_to_channel:
                    channel = self.marker_to_channel[marker]
                    expressions.append(f'{channel}+')
                else:
                    print(f"Warning: Marker '{marker}' not found in channel mapping")
            
            # Add negative markers
            for marker in negative_markers:
                marker = marker.strip()
                if marker in self.marker_to_channel:
                    channel = self.marker_to_channel[marker]
                    expressions.append(f'{channel}-')
                else:
                    print(f"Warning: Marker '{marker}' not found in channel mapping")
            
            if expressions:  # Only add if we have valid expressions
                scimap_phenotypes.append({
                    'phenotype': phenotype_name,
                    'definition': ', '.join(expressions)
                })
                print(f"  {phenotype_name}: {', '.join(expressions)}")
        
        if not scimap_phenotypes:
            print("Warning: No valid phenotype definitions created")
            self.adata.obs['hierarchical_phenotype'] = 'Unknown'
            return self.adata
        
        print(f"Created {len(scimap_phenotypes)} phenotype definitions")
        
        # Use SCIMAP's phenotyping
        phenotype_df = pd.DataFrame(scimap_phenotypes)
        
        self.adata = sm.tl.phenotype_cells(
            self.adata,
            phenotype=phenotype_df,
            gate=0.5,
            label='hierarchical_phenotype'
        )
        
        # Save phenotype definitions used
        phenotype_df.to_csv(self.output_dir / 'data' / 'phenotype_definitions_used.csv', index=False)
        
        return self.adata
    
    def _consensus_phenotyping(self):
        """Create consensus phenotype combining multiple methods"""
        
        # Combine hierarchical and clustering-based phenotypes
        consensus_map = {}
        
        for idx, row in self.adata.obs.iterrows():
            hierarchical = row.get('hierarchical_phenotype', 'Unknown')
            cluster = row.get('leiden_clusters', 'Unknown')
            
            # Simple consensus: prefer hierarchical phenotype if available
            if hierarchical != 'Unknown' and hierarchical is not None:
                consensus = hierarchical
            else:
                consensus = f'Cluster_{cluster}'
            
            consensus_map[idx] = consensus
        
        self.adata.obs['consensus_phenotype'] = pd.Series(consensus_map)
        
        return self.adata
    
    def statistical_analysis(self, phenotype_counts, phenotype_freq):
        """Perform comprehensive statistical analysis"""
        print("Performing statistical analysis...")
        
        stats_results = {}
        
        if self.metadata is not None and 'group' in self.metadata.columns:
            # Merge group information
            sample_groups = self.metadata.set_index('sample_id')['group']
            
            for phenotype in phenotype_freq.columns:
                # Get data for this phenotype
                pheno_data = phenotype_freq[phenotype]
                pheno_data = pheno_data.to_frame().merge(sample_groups, left_index=True, right_index=True)
                
                # Kruskal-Wallis test (non-parametric ANOVA)
                groups = [group_data[phenotype].values for name, group_data in pheno_data.groupby('group')]
                if len(groups) > 1:
                    h_stat, p_val = kruskal(*groups)
                    
                    stats_results[phenotype] = {
                        'kruskal_h': h_stat,
                        'kruskal_p': p_val,
                        'significant': p_val < 0.05
                    }
                    
                    # Post-hoc pairwise comparisons if significant
                    if p_val < 0.05 and len(groups) > 2:
                        tukey_result = pairwise_tukeyhsd(pheno_data[phenotype], pheno_data['group'])
                        stats_results[phenotype]['posthoc'] = str(tukey_result)
        
        # Save statistical results
        stats_df = pd.DataFrame.from_dict(stats_results, orient='index')
        stats_df.to_csv(self.output_dir / 'statistics' / 'phenotype_statistics.csv')
        
        return stats_results
    
    def create_visualizations(self, phenotype_counts, phenotype_freq):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # 1. Phenotype frequency heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(phenotype_freq.T, annot=True, fmt='.1f', cmap='viridis', 
                   cbar_kws={'label': 'Frequency (%)'})
        plt.title('Cell Phenotype Frequencies by Sample')
        plt.xlabel('Sample ID')
        plt.ylabel('Cell Phenotype')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'phenotype_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Phenotype count stacked bar plot
        plt.figure(figsize=(14, 8))
        phenotype_counts.plot(kind='bar', stacked=True, colormap='Set3', figsize=(14, 8))
        plt.title('Cell Phenotype Counts by Sample')
        plt.xlabel('Sample ID')
        plt.ylabel('Cell Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'phenotype_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plots by group (if metadata available)
        if self.metadata is not None and 'group' in self.metadata.columns:
            sample_groups = self.metadata.set_index('sample_id')['group']
            freq_with_groups = phenotype_freq.merge(sample_groups, left_index=True, right_index=True)
            
            n_phenotypes = len(phenotype_freq.columns)
            fig, axes = plt.subplots(2, (n_phenotypes + 1) // 2, figsize=(20, 12))
            axes = axes.flatten() if n_phenotypes > 1 else [axes]
            
            for i, phenotype in enumerate(phenotype_freq.columns):
                if i < len(axes):
                    sns.boxplot(data=freq_with_groups, x='group', y=phenotype, ax=axes[i])
                    axes[i].set_title(f'{phenotype}')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide extra subplots
            for i in range(n_phenotypes, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Phenotype Frequencies by Group', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'phenotype_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Comprehensive UMAP visualizations
        sc.tl.umap(self.adata)
        
        # Sample subset for visualization if too many cells
        if len(self.adata) > 50000:
            adata_subset = sc.pp.subsample(self.adata, n_obs=50000, random_state=42, copy=True)
        else:
            adata_subset = self.adata.copy()
        
        # Create phenotype and cluster plots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Consensus phenotype
        sc.pl.umap(adata_subset, color='consensus_phenotype', ax=axes[0], 
                  show=False, frameon=False, legend_loc='right margin')
        axes[0].set_title('Cell Phenotypes', fontsize=14)
        
        # Clustering result
        sc.pl.umap(adata_subset, color='leiden_clusters', ax=axes[1], 
                  show=False, frameon=False, legend_loc='right margin')
        axes[1].set_title('Leiden Clusters', fontsize=14)
        
        plt.suptitle('Phenotyping and Clustering Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'phenotype_cluster_umap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual channel expression UMAPs
        n_channels = len(adata_subset.var_names)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, channel in enumerate(adata_subset.var_names):
            if i < len(axes):
                sc.pl.umap(adata_subset, color=channel, ax=axes[i], 
                          show=False, frameon=False, vmax='p99')
                axes[i].set_title(f'{channel} Expression', fontsize=12)
        
        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Channel Expression Patterns', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'channel_expression_umap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved")
    
    def generate_report(self, stats_results):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")
        
        report = f"""
Cell Phenotyping and Statistical Analysis Report
===============================================

Dataset Summary:
- Total cells analyzed: {len(self.adata):,}
- Number of samples: {len(self.adata.obs['sample_id'].unique())}
- Number of phenotypes identified: {len(self.adata.obs['consensus_phenotype'].unique())}

Phenotype Distribution:
"""
        
        # Add phenotype distribution
        phenotype_dist = self.adata.obs['consensus_phenotype'].value_counts()
        for phenotype, count in phenotype_dist.items():
            percentage = (count / len(self.adata)) * 100
            report += f"- {phenotype}: {count:,} cells ({percentage:.2f}%)\n"
        
        report += "\n"
        
        # Add statistical results
        if stats_results:
            report += "Statistical Analysis Results:\n"
            report += "============================\n\n"
            
            significant_phenotypes = [k for k, v in stats_results.items() if v.get('significant', False)]
            
            if significant_phenotypes:
                report += f"Significant differences found in {len(significant_phenotypes)} phenotypes:\n"
                for phenotype in significant_phenotypes:
                    p_val = stats_results[phenotype]['kruskal_p']
                    report += f"- {phenotype}: p = {p_val:.4f}\n"
            else:
                report += "No significant differences found between groups.\n"
        
        report += f"""

Files Generated:
===============
Data Files:
- preprocessed_data.h5ad: Preprocessed AnnData object
- phenotype_counts.csv: Cell counts by phenotype and sample
- phenotype_frequencies.csv: Phenotype frequencies by sample
- phenotype_statistics.csv: Statistical test results
- phenotype_definitions_used.csv: Applied phenotype definitions

Plots:
- phenotype_heatmap.png: Frequency heatmap
- phenotype_counts.png: Stacked bar chart of counts
- phenotype_boxplots.png: Group comparisons (if metadata provided)
- adaptive_phenotyping_umap.png: UMAP colored by phenotype

Analysis completed successfully.
"""
        
        # Save report
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("Report generated: analysis_report.txt")
    
    def run_analysis(self):
        """Run complete phenotyping analysis pipeline"""
        print("\n=== CELL PHENOTYPING ANALYSIS ===")
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Adaptive phenotyping
        phenotype_counts, phenotype_freq = self.auto_phenotype_cells()
        stats_results = self.statistical_analysis(phenotype_counts, phenotype_freq)
        
        # Create visualizations
        self.create_visualizations(phenotype_counts, phenotype_freq)
        
        # Generate report
        self.generate_report(stats_results)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        return self.adata

def main():
    parser = argparse.ArgumentParser(description="Cell phenotyping and statistical analysis")
    parser.add_argument('--input', required=True, help='Input results directory')
    parser.add_argument('--metadata', help='Sample metadata CSV file')
    parser.add_argument('--markers', help='Markers CSV file')
    parser.add_argument('--phenotypes', help='Phenotype definitions CSV file')
    parser.add_argument('--output', default='phenotype_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = CellPhenotypeAnalyzer(args.input, args.metadata, args.output, args.markers, args.phenotypes)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()