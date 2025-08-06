# Libraries
import numpy as np
import pandas as pd
from fafbseg import flywire
from sklearn.cluster import estimate_bandwidth, mean_shift
import seaborn as sns
import matplotlib.pyplot as plt
import navis


### Base Functions
# Function 1: Get the dataframes
def get_codex_synapse_predictions(root_id, client):
    """
    Function to gather synapse predictions for a given root ID.
    """

    # Pull initial data from the CAVE client
    pre_df = client.materialize.query_table(
        table="synapses_nt_v1",
        filter_in_dict={"pre_pt_root_id": [root_id]},
    )

    return pre_df


# Function 2: Get the neuron skeleton
def get_neuron_skeleton(root_id):
    """
    Function to gather the neuron skeleton for a given root ID.
    """
    neuron_skeleton = flywire.get_skeletons(root_id)

    return neuron_skeleton


# Function 3: Identify NT contributions per neuron
def identify_nt_contributions(pre_df, softmax_threshold=0.25):
    """
    Function to identify neurotransmitter contributions per neuron.
    Inputs:
    pre_df: DataFrame containing synapse predictions with columns for each neurotransmitter type.
    softmax_threshold: Float, the threshold below which a neurotransmitter is considered 'Unknown'.

    Returns:
    nt_counts: a dictionary with counts of each neurotransmitter type.
    synapse_predictions: a numpy array with synapse predictions for each neurotransmitter type. (NB excludes 'Unknown' type)
    """

    # Synapse predictions
    synapse_predictions = np.zeros((len(pre_df), 6), dtype=np.float32)
    for i in range(pre_df.shape[0]):
        row = [
            pre_df.iloc[i]["gaba"],
            pre_df.iloc[i]["ach"],
            pre_df.iloc[i]["glut"],
            pre_df.iloc[i]["oct"],
            pre_df.iloc[i]["ser"],
            pre_df.iloc[i]["da"],
        ]
        synapse_predictions[i] = row

    # Define neurotransmitter types
    nt_types = ["GABA", "ACh", "Glut", "Oct", "Ser", "DA", "Unknown"]

    # Initialize counts and filter along softmax values
    nt_counts = {nt: 0 for nt in nt_types}
    softmax_prediction_vals = {nt: [] for nt in nt_types}

    for i in range(pre_df.shape[0]):
        # Determine the synapse type based on the maximum prediction value
        synapse_type = np.argmax(
            pre_df.iloc[i][["gaba", "ach", "glut", "oct", "ser", "da"]].values
        )
        softmax_val = np.max(
            pre_df.iloc[i][["gaba", "ach", "glut", "oct", "ser", "da"]].values
        )
        if softmax_val < softmax_threshold:
            synapse_type = 6  # Assign 'Unknown' if below threshold

        # Add the softmax value to the corresponding neurotransmitter type
        softmax_prediction_vals[nt_types[synapse_type]].append(softmax_val)

        # Increment the count for the neurotransmitter type
        nt_counts[nt_types[synapse_type]] += 1

    # Convert to ratios
    total_synapses = pre_df.shape[0]
    for nt in nt_counts.keys():
        if total_synapses > 0:
            nt_counts[nt] = nt_counts[nt] / total_synapses
        else:
            nt_counts[nt] = 0

    return nt_counts, synapse_predictions


### Co-transmission Investigation Functions
# Function 4: Predict which NTs the neuron uses via ratios
def predict_neurotransmitter_usage_ratios(nt_used, nt_counts, ratio_threshold) -> list:
    """
    Function to predict which neurotransmitters a neuron uses based on ratios.
    Returns a list of neurotransmitter types that exceed the ratio threshold.

    Inputs:
    nt_used: List of boolean values indicating whether each neurotransmitter is used.
    nt_counts: Dictionary with ratio of each neurotransmitter type.
    ratio_threshold: Float, the threshold above which a neurotransmitter is considered used.
    """
    c = 0
    for nt, ratio in nt_counts.items():
        if ratio >= ratio_threshold:
            nt_used[c] = True
        c += 1

    return nt_used


# Function 5: Mean Shift Clustering
def predict_neurotransmitter_usage_clustering(
    pre_df,
    nt_used,
    synapse_predictions,
    bandwidth_quantile=1 / 7,
    min_synapses_ratio=0.01,
) -> list:
    """
    Function to predict neurotransmitter usage based on clustering.
    Returns a list of neurotransmitter types that are used based on clustering.

    Inputs:
    pre_df: DataFrame containing synapse predictions with columns for each neurotransmitter type.
    nt_used: List of boolean values indicating whether each neurotransmitter is used.
    synapse_predictions: Numpy array with synapse predictions for each neurotransmitter type.
    bandwidth_quantile: Float, the quantile for estimating bandwidth in mean shift clustering.
    min_synapses_ratio: Float, the minimum ratio of synapses per cluster to consider it valid.

    Returns:
    nt_used: List of boolean values indicating whether each neurotransmitter is used.
    """
    temp_df = pre_df.copy()
    all_coords = np.array([pos for pos in pre_df["pre_pt_position"].values])
    total_synapses = all_coords.shape[0]

    # Calculate Parameters for mean shift clustering
    bandwidth = estimate_bandwidth(
        all_coords, quantile=bandwidth_quantile
    )  # Set to 1/the total number of NT types (1/7)
    min_synapses = int(
        total_synapses * min_synapses_ratio
    )  # Set to 1% of the total number of synapses
    print(f"Estimated bandwidth: {bandwidth}")
    print(f"Minimum synapses per cluster: {min_synapses}")

    # Add the synapse type to the DataFrame
    synapse_ids = []
    for i in range(pre_df.shape[0]):
        # Create a code for the synapse type
        synapse_type = np.argmax(synapse_predictions[i])
        softmax_val = np.max(synapse_predictions[i])
        if softmax_val < 0.25:
            synapse_type = 6
        synapse_ids.append(synapse_type)
    temp_df["synapse_type"] = synapse_ids

    # Loop through each neurotransmitter type and apply mean shift clustering
    for nt_type in range(7):
        nt_df = temp_df[temp_df["synapse_type"] == nt_type]
        coords = np.array([pos for pos in nt_df["pre_pt_position"].values])
        if coords.shape[0] < 2:
            print(f"Skipping NT type {nt_type} due to insufficient synapses.")
            continue
        cluster_centers, labels = mean_shift(coords, bandwidth=bandwidth)

        # Filter clusters with fewer than min_synapses
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= min_synapses]
        n_clusters = len(valid_labels)

        if n_clusters > 0:
            nt_used[nt_type] = True
        else:
            nt_used[nt_type] = False

    return nt_used


### Final Function
# Function 6: Identify NT contributions for a neuron
def identify_nt_contributions_for_neuron(
    root_id,
    client,
    softmax_threshold=0.25,
    ratio_threshold=0.16,
    bandwidth_quantile=1 / 7,
    min_synapses_ratio=0.01,
    display_fig=False,
) -> dict:
    """
    Function to identify neurotransmitter contributions per neuron.
    Inputs:
    root_id: The root ID of the neuron to analyze.
    client: The CAVE client to use for querying data.
    softmax_threshold: Float, the threshold below which a neurotransmitter is considered 'Unknown'.
    ratio_threshold: Float, the threshold above which a neurotransmitter is considered used.
    bandwidth_quantile: Float, the quantile to use for bandwidth estimation in clustering.
    min_synapses_ratio: Float, the ratio of minimum synapses required for a cluster to be considered valid.
    display_fig: Boolean, whether to display the figures or not.

    Returns a list of neurotransmitter types that exceed the ratio threshold.
    """
    # Get the synapse predictions
    pre_df = get_codex_synapse_predictions(root_id, client)

    # Identify neurotransmitter contributions
    nt_counts, synapse_predictions = identify_nt_contributions(
        pre_df, softmax_threshold
    )

    # Predict neurotransmitter usage based on ratios
    nt_used = np.zeros(7, dtype=bool)
    nt_used = predict_neurotransmitter_usage_ratios(nt_used, nt_counts, ratio_threshold)

    # Predict the neurotransmitter usage via clustering
    nt_used = predict_neurotransmitter_usage_clustering(
        pre_df,
        nt_used,
        synapse_predictions,
        bandwidth_quantile=bandwidth_quantile,
        min_synapses_ratio=min_synapses_ratio,
    )

    # Create a output dictionary to store the results
    nt_results = {
        "root_id": root_id,
    }
    # Add the nt_used boolean array to the results with neurotransmitter names as key.
    nt_names = nt_counts.keys()
    for i, nt in enumerate(nt_names):
        nt_results[nt] = nt_used[i]

    if display_fig:
        neuron_skeleton = get_neuron_skeleton(root_id)
        # Plot the neuron skeleton with neurotransmitter contributions
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))

        # Plot the synapse predictions as a heatmap
        sns.heatmap(synapse_predictions, ax=ax[0, 0])
        ax[0, 0].set_title("Synapse Predictions Heatmap")
        ax[0, 0].set_xlabel("Neurotransmitter Type")
        ax[0, 0].set_xticks(np.arange(6) + 0.5)
        ax[0, 0].set_xticklabels(
            ["GABA", "ACh", "Glut", "Oct", "Ser", "DA"], rotation=45
        )

        # Plot each row of the synapse predictions as a line plot
        for i in range(synapse_predictions.shape[0]):
            ax[0, 1].plot(
                synapse_predictions[i], color="k", label=f"Synapse {i+1}", alpha=0.1
            )
        mean_predictions = np.mean(synapse_predictions, axis=0)
        median_predictions = np.median(synapse_predictions, axis=0)
        ax[0, 1].plot(mean_predictions, label="Mean", color="r")
        ax[0, 1].plot(median_predictions, label="Median", color="blue")
        ax[0, 1].set_xlabel("Neurotransmitter Type")
        ax[0, 1].set_xticks([0, 1, 2, 3, 4, 5])
        ax[0, 1].set_xticklabels(
            ["GABA", "ACh", "Glut", "Oct", "Ser", "DA"], rotation=45
        )

        # Plot the neuron skeleton with the synapse predictions - all 3 combinations
        navis.plot2d(neuron_skeleton, ax=ax[1, 0], color="k", view="xy")
        navis.plot2d(neuron_skeleton, ax=ax[1, 1], color="k", view="yz")
        navis.plot2d(neuron_skeleton, ax=ax[1, 2], color="k", view="xz")

        # Define neurotransmitter types and colors
        present_types = set()
        softmax_prediction_vals = {
            "GABA": [],
            "ACh": [],
            "Glut": [],
            "Oct": [],
            "Ser": [],
            "DA": [],
            "Unknown": [],
        }
        nt_types = ["GABA", "ACh", "Glut", "Oct", "Ser", "DA", "Unknown"]
        colours = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray"]

        for i in range(synapse_predictions.shape[0]):
            # Determine the synapse type based on the maximum prediction value
            synapse_type = np.argmax(synapse_predictions[i])
            softmax_val = np.max(synapse_predictions[i])
            if softmax_val < softmax_threshold:
                synapse_type = 6  # Assign 'Unknown' if below threshold
            colour = colours[synapse_type]
            present_types.add(synapse_type)

            # Add the softmax value to the corresponding neurotransmitter type
            softmax_prediction_vals[nt_types[synapse_type]].append(softmax_val)

            # Find the position of the pre-synapse
            pos = pre_df.iloc[i]["pre_pt_position"]
            pos_xy = (pos[0], pos[1])
            pos_yz = (pos[1], pos[2])
            pos_xz = (pos[0], pos[2])

            # Plot the synapse prediction on the neuron skeleton
            ax[1, 0].plot(
                pos_xy[0], pos_xy[1], "o", color=colour, markersize=2, alpha=0.75
            )
            ax[1, 1].plot(
                pos_yz[0], pos_yz[1], "o", color=colour, markersize=2, alpha=0.75
            )
            ax[1, 2].plot(
                pos_xz[0], pos_xz[1], "o", color=colour, markersize=2, alpha=0.75
            )

        # Create legend entries only for neurotransmitter types that are present
        legend_elements = []
        for nt_idx in sorted(present_types):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colours[nt_idx],
                    markersize=8,
                    label=nt_types[nt_idx],
                )
            )

        # Place the legend horizontally below the x-axis
        ax[1, 0].legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            borderaxespad=0.0,
            ncol=round(len(legend_elements) / 2),
            frameon=False,
        )
        ax[1, 0].set_title("Neuron Skeleton with Synapse Predictions - Coronal")
        ax[1, 1].set_title("Neuron Skeleton with Synapse Predictions - Sagittal")
        ax[1, 2].set_title("Neuron Skeleton with Synapse Predictions - Axial")

        # Plot the counts of each neurotransmitter type
        ax[0, 3].bar(nt_counts.keys(), nt_counts.values(), color=colours)
        ax[0, 3].set_xlabel("Neurotransmitter Type")
        ax[0, 3].set_ylabel("Count")
        ax[0, 3].set_title("Counts of Neurotransmitter Types")

        # Plot the counts of each neurotransmitter type
        pcts = [nt_counts[nt] / synapse_predictions.shape[0] for nt in nt_counts.keys()]
        ax[1, 3].bar(nt_counts.keys(), pcts, color=colours)
        ax[1, 3].set_xlabel("Neurotransmitter Type")
        ax[1, 3].set_ylabel("Ratios")
        ax[1, 3].set_title("Ratios of Neurotransmitter Types")

        # Softmax confidence of the predictions
        sns.stripplot(data=softmax_prediction_vals, ax=ax[0, 2])
        ax[0, 2].set_title("Softmax Confidence of Predictions")
        ax[0, 2].set_xlabel("Neurotransmitter Type")
        ax[0, 2].set_ylabel("Softmax Confidence of chosen neurotransmitter")

        plt.suptitle(
            f"Synapse Predictions for Neuron {root_id} with Softmax Threshold {softmax_threshold}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()

    return nt_results
