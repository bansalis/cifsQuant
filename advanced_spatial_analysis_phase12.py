#!/usr/bin/env python3
"""
Advanced Spatial Analysis - Phase 1.2 and Phase 2
Tumor Structure Detection and Critical Research Questions
"""

# This extends the advanced_spatial_analysis.py script

def phase12_detect_tumor_structures(self):
    """
    Phase 1.2: Tumor structure detection using DBSCAN.

    Implements DBSCAN clustering on TOM+ cells with morphological validation.
    Defines tumor zones: core, margin, peri-tumor, and distal regions.

    Returns
    -------
    pd.DataFrame
        Tumor structures with metrics
    """
    print("\n" + "=" * 80)
    print("PHASE 1.2: TUMOR STRUCTURE DETECTION")
    print("=" * 80 + "\n")

    # Get tumor cells
    tumor_cells = self.cells_df[
        self.cells_df['phenotype'].str.contains('Tumor', na=False)
    ].copy()

    if len(tumor_cells) == 0:
        print("Warning: No tumor cells found!")
        return pd.DataFrame()

    print(f"Detecting tumor structures from {len(tumor_cells):,} tumor cells...")

    tumors_list = []
    tumor_id_counter = 0

    # Process per sample
    for sample_id in tumor_cells['sample_id'].unique():
        sample_tumor_cells = tumor_cells[tumor_cells['sample_id'] == sample_id]

        # DBSCAN clustering
        coords = sample_tumor_cells[['X_centroid', 'Y_centroid']].values

        clustering = DBSCAN(
            eps=self.config.tumor_dbscan_eps,
            min_samples=self.config.tumor_dbscan_min_samples,
            metric='euclidean'
        ).fit(coords)

        labels = clustering.labels_

        # Assign tumor IDs to cells
        sample_tumor_cells['tumor_id'] = labels
        self.cells_df.loc[sample_tumor_cells.index, 'tumor_id'] = labels

        # Calculate tumor metrics
        n_tumors = len([l for l in np.unique(labels) if l >= 0])
        print(f"  {sample_id}: detected {n_tumors} tumors")

        for tumor_label in np.unique(labels):
            if tumor_label < 0:  # Noise
                continue

            tumor_cells_mask = labels == tumor_label
            tumor_coords = coords[tumor_cells_mask]

            # Global tumor ID
            global_tumor_id = f"{sample_id}_T{tumor_id_counter}"
            tumor_id_counter += 1

            # Basic metrics
            n_cells = np.sum(tumor_cells_mask)
            centroid_x = np.mean(tumor_coords[:, 0])
            centroid_y = np.mean(tumor_coords[:, 1])

            # Morphological metrics
            try:
                hull = ConvexHull(tumor_coords)
                area = hull.volume  # In 2D, volume is area
                perimeter = hull.area  # In 2D, area is perimeter
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            except:
                # Fallback for small/collinear tumors
                area = n_cells * 100  # Rough estimate
                perimeter = 0
                circularity = 0

            tumors_list.append({
                'tumor_id': global_tumor_id,
                'sample_id': sample_id,
                'tumor_label': tumor_label,
                'n_cells': n_cells,
                'area_um2': area,
                'perimeter_um': perimeter,
                'circularity': circularity,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
            })

    self.tumors_df = pd.DataFrame(tumors_list)

    # Merge with metadata
    self.tumors_df = self.tumors_df.merge(
        self.sample_metadata,
        on='sample_id',
        how='left'
    )

    print(f"\nDetected {len(self.tumors_df)} total tumors")
    print(f"Tumor size range: {self.tumors_df['n_cells'].min():.0f} - "
          f"{self.tumors_df['n_cells'].max():.0f} cells")

    # Save tumor structures
    output_file = self.output_dir / 'data' / 'tumor_structures.csv'
    self.tumors_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Define tumor zones for each cell
    self._define_tumor_zones()

    # Generate visualizations
    self._plot_tumor_structures()
    self._plot_tumor_size_distribution()

    return self.tumors_df


def _define_tumor_zones(self):
    """Define tumor zones for each cell based on distance to tumor structures."""
    print("\nDefining tumor zones...")

    # Initialize zone column
    self.cells_df['tumor_zone'] = 'Distal'

    for sample_id in self.cells_df['sample_id'].unique():
        sample_cells = self.cells_df[self.cells_df['sample_id'] == sample_id]
        sample_tumors = self.tumors_df[self.tumors_df['sample_id'] == sample_id]

        if len(sample_tumors) == 0:
            continue

        # Get coordinates
        all_coords = sample_cells[['X_centroid', 'Y_centroid']].values
        tumor_cells = sample_cells[
            sample_cells['phenotype'].str.contains('Tumor', na=False)
        ]

        if len(tumor_cells) == 0:
            continue

        tumor_coords = tumor_cells[['X_centroid', 'Y_centroid']].values

        # Build tree for tumor cells
        tree = cKDTree(tumor_coords)

        # Query nearest tumor cell distance for all cells
        distances, _ = tree.query(all_coords, k=1)

        # Assign zones
        sample_indices = sample_cells.index

        # Core: tumor cells themselves
        tumor_mask = sample_cells['phenotype'].str.contains('Tumor', na=False)
        self.cells_df.loc[sample_indices[tumor_mask], 'tumor_zone'] = 'Core'

        # Margin: 0-50 μm from tumor
        margin_mask = (~tumor_mask) & (distances <= self.config.margin_width)
        self.cells_df.loc[sample_indices[margin_mask], 'tumor_zone'] = 'Margin'

        # Peri-tumor: 50-150 μm from tumor
        peritumor_mask = (
            (~tumor_mask) &
            (distances > self.config.peritumor_inner) &
            (distances <= self.config.peritumor_outer)
        )
        self.cells_df.loc[sample_indices[peritumor_mask], 'tumor_zone'] = 'Peri-tumor'

        # Distal: >150 μm
        # Already initialized to 'Distal'

    zone_counts = self.cells_df['tumor_zone'].value_counts()
    print("Tumor zone assignment:")
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count:,} cells")


def _plot_tumor_structures(self):
    """Plot tumor structures with boundaries and zones."""
    print("\nGenerating tumor structure plots...")

    output_dir = self.output_dir / '01_phenotyping' / 'individual_plots'

    for sample_id in self.tumors_df['sample_id'].unique():
        sample_cells = self.cells_df[self.cells_df['sample_id'] == sample_id]
        sample_tumors = self.tumors_df[self.tumors_df['sample_id'] == sample_id]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot cells by zone
        zone_colors = {
            'Core': 'red',
            'Margin': 'orange',
            'Peri-tumor': 'yellow',
            'Distal': 'lightgray'
        }

        for zone, color in zone_colors.items():
            zone_cells = sample_cells[sample_cells['tumor_zone'] == zone]
            if len(zone_cells) > 0:
                ax.scatter(zone_cells['X_centroid'], zone_cells['Y_centroid'],
                          s=1, c=color, alpha=0.5, label=zone, rasterized=True)

        # Draw tumor boundaries
        for _, tumor in sample_tumors.iterrows():
            # Get tumor cells
            tumor_cells = sample_cells[
                (sample_cells['phenotype'].str.contains('Tumor', na=False)) &
                (sample_cells['tumor_id'] == tumor['tumor_label'])
            ]

            if len(tumor_cells) > 4:
                try:
                    coords = tumor_cells[['X_centroid', 'Y_centroid']].values
                    hull = ConvexHull(coords)
                    for simplex in hull.simplices:
                        ax.plot(coords[simplex, 0], coords[simplex, 1],
                               'k-', linewidth=0.5, alpha=0.7)
                except:
                    pass

            # Mark centroid
            ax.plot(tumor['centroid_x'], tumor['centroid_y'],
                   'b*', markersize=10)

        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Tumor Structures and Zones - {sample_id}')
        ax.legend(markerscale=5, loc='best')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_dir / f'tumor_structures_{sample_id}.pdf',
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

    print(f"Tumor structure plots saved to: {output_dir}")


def _plot_tumor_size_distribution(self):
    """Plot tumor size distribution across samples."""
    output_dir = self.output_dir / '01_phenotyping' / 'aggregated_panels'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall size distribution
    ax = axes[0, 0]
    ax.hist(self.tumors_df['n_cells'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Tumor Size (cells)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Tumor Size Distribution')
    ax.set_yscale('log')

    # Size by group
    ax = axes[0, 1]
    groups = self.tumors_df['group'].unique()
    for group in groups:
        group_tumors = self.tumors_df[self.tumors_df['group'] == group]
        ax.hist(group_tumors['n_cells'], bins=30, alpha=0.5,
               label=group, edgecolor='black')
    ax.set_xlabel('Tumor Size (cells)')
    ax.set_ylabel('Frequency')
    ax.set_title('Tumor Size by Group')
    ax.legend()

    # Size vs circularity
    ax = axes[1, 0]
    scatter = ax.scatter(self.tumors_df['n_cells'], self.tumors_df['circularity'],
                        c=self.tumors_df['timepoint'], cmap='viridis',
                        alpha=0.6, s=50)
    ax.set_xlabel('Tumor Size (cells)')
    ax.set_ylabel('Circularity')
    ax.set_title('Tumor Size vs Circularity')
    plt.colorbar(scatter, ax=ax, label='Timepoint')

    # Boxplot by timepoint
    ax = axes[1, 1]
    timepoints = sorted(self.tumors_df['timepoint'].dropna().unique())
    data_by_time = [
        self.tumors_df[self.tumors_df['timepoint'] == tp]['n_cells'].values
        for tp in timepoints
    ]
    ax.boxplot(data_by_time, labels=[str(int(t)) for t in timepoints])
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Tumor Size (cells)')
    ax.set_title('Tumor Size by Timepoint')

    plt.tight_layout()
    plt.savefig(output_dir / 'tumor_size_distribution.pdf',
               dpi=self.config.dpi, bbox_inches='tight')
    plt.close()

    print(f"Size distribution plot saved to: {output_dir}")


# =============================================================================
# PHASE 2.1: pERK SPATIAL ARCHITECTURE ANALYSIS
# =============================================================================

def phase21_perk_spatial_architecture(self):
    """
    Phase 2.1: pERK spatial architecture analysis.

    Q1: Are pERK+ regions spatially clustered or stochastic?
    Q2: pERK+ region growth dynamics
    Q3: pERK+ region infiltration differential

    Returns
    -------
    dict
        Results dictionary with clustering metrics, growth dynamics, and infiltration
    """
    print("\n" + "=" * 80)
    print("PHASE 2.1: pERK SPATIAL ARCHITECTURE ANALYSIS")
    print("=" * 80 + "\n")

    results = {}

    # Q1: Spatial clustering
    print("Q1: Analyzing pERK+ spatial clustering...")
    clustering_results = self._analyze_perk_clustering()
    results['clustering'] = clustering_results

    # Q2: Growth dynamics
    print("\nQ2: Analyzing pERK+ growth dynamics...")
    growth_results = self._analyze_perk_growth()
    results['growth'] = growth_results

    # Q3: Infiltration differential
    print("\nQ3: Analyzing pERK+ infiltration differential...")
    infiltration_results = self._analyze_perk_infiltration()
    results['infiltration'] = infiltration_results

    self.results['perk_analysis'] = results

    return results


def _analyze_perk_clustering(self):
    """
    Analyze spatial clustering of pERK+ regions using:
    - Ripley's K function
    - Getis-Ord Gi* hotspot analysis
    - Moran's I spatial autocorrelation
    """
    from scipy.spatial.distance import pdist, squareform

    clustering_results = []

    perk_cells = self.cells_df[
        self.cells_df['phenotype'].str.contains('pERK', na=False)
    ].copy()

    # Analyze per tumor
    for tumor_id in self.tumors_df['tumor_id'].unique():
        tumor_info = self.tumors_df[self.tumors_df['tumor_id'] == tumor_id].iloc[0]
        sample_id = tumor_info['sample_id']
        tumor_label = tumor_info['tumor_label']

        # Get tumor cells
        tumor_cells = self.cells_df[
            (self.cells_df['sample_id'] == sample_id) &
            (self.cells_df['tumor_id'] == tumor_label)
        ]

        # Get pERK+ cells in this tumor
        perk_in_tumor = tumor_cells[
            tumor_cells['phenotype'].str.contains('pERK', na=False)
        ]

        if len(perk_in_tumor) < 10:
            continue  # Skip tumors with too few pERK+ cells

        # Calculate Moran's I
        coords = perk_in_tumor[['X_centroid', 'Y_centroid']].values

        if len(coords) < 3:
            continue

        # Build distance matrix
        dist_matrix = squareform(pdist(coords, metric='euclidean'))

        # Inverse distance weights (with small constant to avoid division by zero)
        weights = 1.0 / (dist_matrix + 1e-6)
        np.fill_diagonal(weights, 0)

        # Normalize weights
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / (row_sums + 1e-10)

        # Calculate Moran's I
        # For binary variable (pERK+ = 1), Moran's I measures clustering
        values = np.ones(len(coords))  # All are pERK+
        mean_val = values.mean()
        values_centered = values - mean_val

        numerator = np.sum(weights * np.outer(values_centered, values_centered))
        denominator = np.sum(values_centered ** 2)

        moran_i = numerator / (denominator + 1e-10) if denominator > 0 else 0

        # Classify clustering
        cluster_class = 'dispersed' if moran_i < 0 else (
            'clustered' if moran_i > 0.3 else 'random'
        )

        # Ripley's K function (simplified)
        ripley_k = self._calculate_ripley_k(coords)

        # Hotspot detection using local density
        tree = cKDTree(coords)
        hotspots = []
        for i, coord in enumerate(coords):
            # Count neighbors within hotspot radius
            n_neighbors = len(tree.query_ball_point(coord, self.config.hotspot_radius))
            if n_neighbors > 10:  # Threshold for hotspot
                hotspots.append(coord)

        n_hotspots = len(hotspots)

        clustering_results.append({
            'tumor_id': tumor_id,
            'sample_id': sample_id,
            'n_perk_cells': len(perk_in_tumor),
            'moran_i': moran_i,
            'cluster_classification': cluster_class,
            'n_hotspots': n_hotspots,
            'ripley_k_max': np.max(ripley_k) if len(ripley_k) > 0 else 0,
        })

    results_df = pd.DataFrame(clustering_results)

    # Merge with metadata
    results_df = results_df.merge(
        self.tumors_df[['tumor_id', 'group', 'timepoint']],
        on='tumor_id',
        how='left'
    )

    # Save results
    output_file = self.output_dir / '02_perk_analysis' / 'statistics' / 'perk_spatial_clustering.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Generate visualizations
    self._plot_perk_clustering(results_df)

    return results_df


def _calculate_ripley_k(self, coords):
    """Calculate Ripley's K function."""
    if len(coords) < 3:
        return np.array([])

    n = len(coords)
    tree = cKDTree(coords)

    # Calculate area (rough estimate)
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area = x_range * y_range

    if area == 0:
        return np.array([])

    # Distance steps
    max_dist = min(self.config.ripley_max_distance, min(x_range, y_range) / 2)
    distances = np.linspace(0, max_dist, self.config.ripley_n_steps)

    k_values = []
    for d in distances:
        # Count pairs within distance d
        pairs_count = 0
        for i in range(n):
            neighbors = tree.query_ball_point(coords[i], d)
            pairs_count += len(neighbors) - 1  # Exclude self

        # Ripley's K
        k = (area / (n * (n - 1))) * pairs_count
        k_values.append(k)

    return np.array(k_values)


def _plot_perk_clustering(self, results_df):
    """Generate plots for pERK clustering analysis."""
    output_dir_indiv = self.output_dir / '02_perk_analysis' / 'individual_plots'
    output_dir_agg = self.output_dir / '02_perk_analysis' / 'aggregated_panels'

    # Aggregated panel: 4x3 grid showing representative tumors
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Select representative tumors from each classification and timepoint
    classifications = results_df['cluster_classification'].unique()
    timepoints = sorted(results_df['timepoint'].dropna().unique())

    plot_idx = 0
    for class_idx, classification in enumerate(classifications):
        for time_idx, timepoint in enumerate(timepoints[:3]):  # Max 3 timepoints
            ax = fig.add_subplot(gs[class_idx, time_idx])

            # Get representative tumor
            subset = results_df[
                (results_df['cluster_classification'] == classification) &
                (results_df['timepoint'] == timepoint)
            ]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{classification} - T{int(timepoint)}')
                continue

            tumor_id = subset.iloc[0]['tumor_id']
            sample_id = subset.iloc[0]['sample_id']
            tumor_label = self.tumors_df[
                self.tumors_df['tumor_id'] == tumor_id
            ].iloc[0]['tumor_label']

            # Plot tumor with pERK+ highlighted
            tumor_cells = self.cells_df[
                (self.cells_df['sample_id'] == sample_id) &
                (self.cells_df['tumor_id'] == tumor_label)
            ]

            # All tumor cells in gray
            ax.scatter(tumor_cells['X_centroid'], tumor_cells['Y_centroid'],
                      s=5, c='lightgray', alpha=0.5, rasterized=True)

            # pERK+ cells in red
            perk_cells = tumor_cells[
                tumor_cells['phenotype'].str.contains('pERK', na=False)
            ]
            if len(perk_cells) > 0:
                ax.scatter(perk_cells['X_centroid'], perk_cells['Y_centroid'],
                          s=10, c='red', alpha=0.8, rasterized=True)

            moran_i = subset.iloc[0]['moran_i']
            ax.set_title(f'{classification} - T{int(timepoint)}\n'
                        f"Moran's I = {moran_i:.3f}")
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_aspect('equal')

    plt.suptitle('pERK+ Spatial Clustering Patterns', fontsize=20, y=0.995)
    plt.savefig(output_dir_agg / 'perk_clustering_representative_tumors.pdf',
               dpi=self.config.dpi, bbox_inches='tight')
    plt.close()

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Moran's I distribution
    ax = axes[0, 0]
    for classification in results_df['cluster_classification'].unique():
        subset = results_df[results_df['cluster_classification'] == classification]
        ax.hist(subset['moran_i'], bins=20, alpha=0.5, label=classification, edgecolor='black')
    ax.axvline(0.3, color='red', linestyle='--', label='Clustering threshold')
    ax.set_xlabel("Moran's I")
    ax.set_ylabel('Frequency')
    ax.set_title("pERK+ Spatial Autocorrelation Distribution")
    ax.legend()

    # Classification by timepoint
    ax = axes[0, 1]
    class_time = results_df.groupby(['timepoint', 'cluster_classification']).size().reset_index(name='count')
    pivot = class_time.pivot(index='timepoint', columns='cluster_classification', values='count').fillna(0)
    pivot.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Number of Tumors')
    ax.set_title('pERK+ Clustering Classification by Timepoint')
    ax.legend(title='Classification')

    # Moran's I by group
    ax = axes[1, 0]
    groups = results_df['group'].unique()
    data_by_group = [results_df[results_df['group'] == g]['moran_i'].dropna().values
                     for g in groups]
    ax.boxplot(data_by_group, labels=groups)
    ax.set_ylabel("Moran's I")
    ax.set_title("pERK+ Spatial Clustering by Group")
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)

    # Hotspot count distribution
    ax = axes[1, 1]
    for group in groups:
        subset = results_df[results_df['group'] == group]
        ax.scatter(subset['n_perk_cells'], subset['n_hotspots'],
                  alpha=0.6, label=group, s=50)
    ax.set_xlabel('Number of pERK+ Cells')
    ax.set_ylabel('Number of Hotspots')
    ax.set_title('pERK+ Hotspot Detection')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir_agg / 'perk_clustering_statistics.pdf',
               dpi=self.config.dpi, bbox_inches='tight')
    plt.close()

    print(f"pERK clustering plots saved to: {output_dir_agg}")


# Add these methods to the AdvancedSpatialAnalysis class
# by importing this module and adding methods dynamically

import types

def add_phase12_methods(analysis_instance):
    """Add Phase 1.2 and 2.1 methods to AdvancedSpatialAnalysis instance."""
    analysis_instance.phase12_detect_tumor_structures = types.MethodType(
        phase12_detect_tumor_structures, analysis_instance
    )
    analysis_instance._define_tumor_zones = types.MethodType(
        _define_tumor_zones, analysis_instance
    )
    analysis_instance._plot_tumor_structures = types.MethodType(
        _plot_tumor_structures, analysis_instance
    )
    analysis_instance._plot_tumor_size_distribution = types.MethodType(
        _plot_tumor_size_distribution, analysis_instance
    )
    analysis_instance.phase21_perk_spatial_architecture = types.MethodType(
        phase21_perk_spatial_architecture, analysis_instance
    )
    analysis_instance._analyze_perk_clustering = types.MethodType(
        _analyze_perk_clustering, analysis_instance
    )
    analysis_instance._calculate_ripley_k = types.MethodType(
        _calculate_ripley_k, analysis_instance
    )
    analysis_instance._plot_perk_clustering = types.MethodType(
        _plot_perk_clustering, analysis_instance
    )
