# ai-model-for-connectomics
AI Model for Connectomics

## Set up virtual environment
This is designed to run as a conda environment. So use the standard commands to set up and enter the environment
```bash
conda env create -f environment.yml
conda activate fau_connectomics
```

## Data
The [data folder](data) contains datasets and utilities for working with FAFB and hemibrain connectome data:

### Supplemental Data from Paper
Located in `supplemental_data_from_paper/`:
* **DataS1_CellTypesUsedForGroundTruth.csv** - Cell type classifications used for validation
* **DataS2_NeuronalReconstructionsUsedForGroundTruth.csv** - Reference neuronal reconstructions
* **DataS3_HemiBrainReconstructionData.csv** - Hemibrain connectome reconstruction data
* **DataS4_FAFBreconstruction.csv** - FAFB reconstruction data (also available as `simplified_DataS4.csv`)
* **DataS6_SummaryResults.csv** - Summary statistics and results
* **DataS7.csv** - Additional supplementary data

### Test Data from Collaborators
Located in `test_data_from_alex/`:
* **hemi_lineages.json** - Hemibrain lineage information
* **skeletons.json** / **skeletons_fixed.json** - Neuron skeleton data in JSON format
* **synapses_xyz_pos.json** - Synapse spatial coordinates
* **single_neuron/** - Individual neuron data for testing

### Utility Scripts
* **fix_json.py** - Converts JSONL (JSON Lines) format to standard JSON array format
* **read_jsonl_example.py** - Example script for reading JSONL files
* **single_neuron_presynapses.parquet** - Parquet format data for single neuron presynaptic terminals

## load_data
The [load_data folder](load_data) contains Python modules for data loading and analysis:

### Core Modules
* **connect_clients.py** - Functions to connect to CAVE and FlyWire clients
  - Handles authentication token management
  - Provides secure connection setup with environment variable support
  - Prompts for tokens if not found in `.env` file

* **get_synapse_locations.py** - Functions to extract synapse location data from FAFB
  - Initializes FAFB CAVE client
  - Retrieves synapse coordinates and metadata
  - Handles voxel resolution conversions

* **fafb_cotransmission_investigation.py** - Core analysis functions for cotransmission detection
  - `get_codex_synapse_predictions()` - Retrieves NT predictions for a given root ID
  - `get_neuron_skeleton()` - Fetches neuron skeleton geometry
  - `identify_nt_contributions()` - Analyzes NT composition per neuron
  - Implements mean-shift clustering for spatial analysis
  - Generates visualization functions for NT distributions

### Usage
These modules are imported by the analysis notebooks to provide consistent data access and processing functions. They require authentication tokens stored in a `.env` file (see setup instructions above).


## notebooks
Jupyter notebooks that explore the data. Note to gain access to the cave client and flywire client you will need to get/use your own secret keys.

NB Corresponding output data is stored in subfolders. For all the possible root-ids investigated I suggest using the [fauai-13_data](notebooks/fauai-13_data) folder.

### fauai-9_GatherSynapsePredictions.ipynb
**Purpose:** This uses the CAVEclient and Flywire in order to extract the Eckstein et al. synaptic predictions and look at their distributions and generally get a feel for the data.

**Major findings:**
1. Created a simple plot of 8 panels to visualise the neurotransmitter of all presynaptic terminals belonging to that neuron. 
* Top row: Left to Right
    * Heatmap of NT predictions: Row = each sympse, col=NT identity, colour indicates the softmax scores from the CNN.
    * Vectors of heatmap predictions: Vectors, each line represents the softmax scores for a particular presynapse.
    * Stripplot showing the determined identity of each presynapse (based off the largest softmax score of the presynapse).
    * Barplot showing the raw count plots for the number of presynapses with each NT.
* Bottom row: Left to right
    * Coronal (xy) plot showing the structure of the neuron and the colour-coded positions of each presynaptic terminal.
    * Sagittal (yz) plot showing the structure of the neuron and the colour-coded positions of each presynaptic terminal.
    * Horizontal (xz) plot showing the structure of the neuron and the colour-coded positions of each presynaptic terminal.
    * Barplot showing the ratio plots for the number of presynapses with each NT.
2. We found that a threshold of 0.25 for the lowest maximum softmax score for a synapse removed many of the worst estimates
3. We also saw that we'd have to consider spatial positioning too as there are hotspots of neurotransmitter specific hotspots. Even if the number of these synapses are low this seems to be vital.

### fauai-10_IdentifyCotransmission.ipynb
**Purpose:** Building on the initial exploration, this notebook develops methods to identify co-transmitting neurons using both ratio-based thresholds and spatial clustering approaches.

**Major findings:**
1. Implemented mean-shift clustering to identify spatial clusters of synapses with consistent neurotransmitter predictions
2. Established dual criteria for cotransmission detection:
    * Ratio threshold method: NT must comprise a minimum percentage of total synapses
    * Clustering method: Spatial clusters of synapses with consistent NT predictions
3. Combined both methods to create robust cotransmission predictions

### fauai-11_EvaluateAllVncCellTypes.ipynb
**Purpose:** Systematic evaluation of all VNC (Ventral Nerve Cord) cell types to identify neurotransmitter expression patterns and cotransmission across different neuron classifications.

**Major findings:**
1. Evaluated neurons across three classification systems: cell_type, cell_class, and cell_flow
2. Processed neurons in batches to handle large-scale analysis efficiently
3. Applied cotransmission detection methods across different cell type categories (afferent, efferent, ascending neurons, etc.)
4. Generated comprehensive results files organized by classification system

### fauai-12_ViewCoTransmissionPatterns.ipynb
**Purpose:** Visualization and statistical analysis of cotransmission patterns identified in the previous analyses.

**Major findings:**
1. Created comprehensive visualizations showing:
    * Distribution of number of neurotransmitters per neuron
    * Frequency of specific neurotransmitter combinations
    * Cotransmission rates across different cell types
2. Identified potential false positives requiring further investigation
3. Established baseline statistics for cotransmission prevalence

### fauai-13_DetailedCotransmissionPatterns.ipynb
**Purpose:** In-depth investigation of cotransmission patterns to identify and address false positives, refining the detection methodology.

**Major findings:**
1. Investigated sources of false positives in cotransmission detection
2. Refined threshold parameters:
    * Softmax threshold: 0.25
    * Bandwidth quantile for clustering: 0.01
    * Minimum synapses ratio: 0.05 (cluster must be ≥5% of neuron's synapses)
3. Tested various combinations of ratio and clustering thresholds
4. Generated detailed results organized by classification system (cell_type, cell_class, cell_flow)

### fauai-14_Investigate_CellTypeClassification.ipynb
**Purpose:** Comprehensive analysis of neurotransmitter expression patterns across different cell type classifications.

**Major findings:**
1. Created clustered heatmaps showing NT expression patterns across cell types
2. Identified optimal threshold parameters:
    * Ratio threshold: 0.2 (20%)
    * Bandwidth: 0.01
    * Min synapses ratio: 0.05
3. Analyzed cotransmission patterns for specific cell type groupings
4. Generated interactive visualizations for exploring cell type-specific NT patterns

### fauai-15_EvaluateSpecificNeurons.ipynb  
**Purpose:** Detailed evaluation of specific neurons requested by collaborators (Pena lab), including comprehensive visualizations of their neurotransmitter profiles.

**Major findings:**
1. Analyzed 10 specific neurons of interest provided by collaborators:
    * All neurons were from the 'efferent' category in the 'flow' classification
2. Generated detailed 8-panel visualizations for each neuron showing:
    * Synapse prediction heatmaps and vectors
    * Spatial distribution of synapses on neuron skeleton (3 views)
    * NT counts and ratios
    * Softmax confidence distributions
3. Applied optimized threshold parameters (ratio: 0.2, bandwidth: 0.01, min_synapses: 0.05)
4. Created summary outputs (CSV and figures) for collaborator communication

### fauai-16_AnalyseAllCODEXdata.ipynb
**Purpose:** Comprehensive analysis of all VNC CODEX data, synthesizing results from all previous analyses to identify overall cotransmission patterns and neurotransmitter combinations.

**Major findings:**
1. Consolidated data from all three classification systems (cell_type, cell_class, cell_flow)
2. Analyzed all possible NT combinations to identify common and rare patterns
3. Created hierarchical clustering visualizations showing:
    * Cell types clustered by NT expression similarity
    * Interactive plotly dendrograms with heatmaps
4. Identified final optimal threshold settings for the entire dataset
5. Generated comprehensive statistics on:
    * NT expression frequencies across cell types
    * Most common cotransmission combinations
    * Cell type-specific NT profiles


