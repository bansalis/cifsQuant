# MCMICRO Tiled Processor

**An integrated workflow for processing large immunofluorescence images through tiling, segmentation, and spatial analysis**

## Running Phenotype analysis
docker run --rm --gpus all -v $(pwd):/app phenotype-analysis \
  --input results/ \
  --metadata sample_metadata.csv \
  --markers markers.csv \
  --output phenotype_analysis/

## Citation

If you use this workflow, please cite:

- **MCMICRO**: Schapiro et al. (2022) Nature Methods
- **SCIMAP**: Nirmal et al. (2021) Nature Methods  
- **Your analysis**: [Add your publication]

## License

This project is licensed under the MIT License - see LICENSE file for details.