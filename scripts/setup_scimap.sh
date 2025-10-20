#!/bin/bash

# Create configuration files for SCIMAP pipeline
echo "Creating SCIMAP configuration files..."

# 1. metadata.csv - Sample metadata
cat > metadata.csv << 'EOF'
sample_id,age_weeks,genotype,treatment,sex,batch
GUEST29,8,cis,Control,M,1
GUEST30,8,trans,Control,M,1
EOF

# 2. phenotypes.csv - Phenotype workflow (SCIMAP format)
#CD4_T_cells,CD3,CD4,CD8,,, pos,pos,neg
cat > phenotypes.csv << 'EOF'
phenotype,marker1,marker2,marker3,marker4,marker5,gate
Immune,CD45,,,,, pos
T_cells,CD3,,,,, pos
CD8_T_cells,CD3,CD8,,,, allpos
Tumor_cells,Tom,,,,, pos
Proliferating_T_cells,CD3,Ki67,,,, allpos
Proliferating_Tumor,Tom,Ki67,,,, allpos
NINJA_Tumor,Tom,aGFP,,,, allpos
EOF

# 3. manual_gates.csv - Optional manual gates (SCIMAP format)
cat > manual_gates.csv << 'EOF'
markers,gates
DAPI,
CD3,
CD8,
PanCK,
CD68,
Ki67,
PDL1,
CD45,
FOXP3,
CD20
EOF

echo "Configuration files created:"
echo "- metadata.csv (sample metadata)"
echo "- phenotypes.csv (phenotype workflow)" 
echo "- manual_gates.csv (manual gating - optional)"
echo ""
echo "Edit these files to match your specific experiment before running the analysis."