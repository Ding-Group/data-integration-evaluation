/*
 * This file contains the pipelines to run the clustering experiments
 * with Leiden/Louvain/K-means/GMM under different parameter settings.
 */

import java.security.MessageDigest
import java.math.BigInteger

include { INTEGRATION } from './integration'


process RUN_DEFAULT_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_default_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            -s ${neighbors_params.seed} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}


process RUN_EXACT_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_exact_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}

process RUN_ANNOY_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_annoy_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            -s ${neighbors_params.seed} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}


process RUN_FAISS_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_faiss_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            -s ${neighbors_params.seed} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}


process RUN_FLANN_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_flann_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            -s ${neighbors_params.seed} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}


process RUN_HNSWLIB_NEIGHBORS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}-${integration}_${neighbors_name}" 

    input:
    tuple val(dataset),
        val(integration),
        val(batch_col),
        val(label_col),
        path(integrated_data),
        path(raw_file),  
        val(neighbors_name),
        val(neighbors_params)

    output:
    tuple val(dataset), // Dataset name
        val(integration), // Integration name
        val(batch_col),  // Batch column, needed to transfer to metrics
        val(label_col),  // Label column, needed to transfer to metrics
        path("neighbors_adata.h5ad"), // Path to the data with neighbors computed
        path(raw_file)  // Path to the raw file

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_hnswlib_neighbors.py \\
            -o ./ \\
            -k ${neighbors_params.k} \\
            -s ${neighbors_params.seed} \\
            ${integrated_data}
        """

    stub:
        """
        touch neighbors_adata.h5ad
        """
}


process RUN_LEIDEN {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    publishDir "$params.outdir/adata/${dataset}/${integration}",
        pattern: "*.h5ad"

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}_${integration}_leiden"

    input:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path(neighbors_adata), path(raw_file), val(leiden_params), val(hex)
    output:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path("clustered-${hex}.h5ad"), path(raw_file), val(hex)

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_leiden.py \\
            -o "clustered-${hex}.h5ad" \\
            -s ${leiden_params.seed} \\
            -f ${leiden_params.flavor} \\
            ${neighbors_adata}
        """
    
    stub:
        """
        touch clustered-${hex}.h5ad
        """
}


process RUN_LOUVAIN {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    publishDir "$params.outdir/adata/${dataset}/${integration}",
        pattern: "*.h5ad"

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}_${integration}_louvain"

    input:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path(neighbors_adata), path(raw_file), val(louvain_params), val(hex)
    output:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path("clustered-${hex}.h5ad"), path(raw_file), val(hex)

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_louvain.py \\
            -o "clustered-${hex}.h5ad" \\
            -s ${louvain_params.seed} \\
            -f ${louvain_params.flavor} \\
            ${neighbors_adata}
        """
    
    stub:
        """
        touch clustered-${hex}.h5ad
        """
}


process RUN_KMEANS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'terminate'

    publishDir "$params.outdir/adata/${dataset}/${integration}",
        pattern: "*.h5ad"

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}_${integration}_kmeans"

    input:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_adata), path(raw_file), val(kmeans_params), val(hex)
    output:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path("clustered-${hex}.h5ad"), path(raw_file), val(hex)

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_kmeans.py \\
            -o "clustered-${hex}.h5ad" \\
            -t ${label_col} \\
            -s ${kmeans_params.seed} \\
            -i ${kmeans_params.init} \\
            ${integrated_adata}
        """
    
    stub:
        """
        touch clustered-${hex}.h5ad
        """
}


process RUN_GMM {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy 'ignore'

    publishDir "$params.outdir/adata/${dataset}/${integration}",
        pattern: "*.h5ad"

    memory { get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB") }

    tag "${dataset}_${integration}_gmm"

    input:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_adata), path(raw_file), val(gmm_params), val(hex)
    output:
    tuple val(dataset), val(integration), val(batch_col), val(label_col), path("clustered-${hex}.h5ad"), path(raw_file), val(hex)

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/compute_gmm.py \\
            -o "clustered-${hex}.h5ad" \\
            -t ${label_col} \\
            -s ${gmm_params.seed} \\
            -i ${gmm_params.init_params} \\
            -n ${gmm_params.n_init} \\
            ${integrated_adata}
        """
    
    stub:
        """
        touch clustered-${hex}.h5ad
        """
}


process RUN_METRIC {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    tag "RUN_METRIC_${integration}_${dataset}_${metric}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}",
        pattern: "*.csv",
        saveAs: { filename -> "${metric}.csv" }

    memory {
        metric == "ALL_CLUSTER" ? 
            get_memory(raw_file.size(), "4.GB", task.attempt, "4.GB") :
        metric == "KBET" ?
            get_memory(raw_file.size(), "4.GB", task.attempt, "8.GB") :
            get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB")
    }

    label "error_retry"

    input:
        tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_file),
            path(raw_file), val(hex), val(metric), val(metric_params)

    output:
        tuple val(dataset), val(integration), path(integrated_file), path("${dataset}-${integration}-${metric}-${hex}.csv"), val(hex)

    script:
        """
        set +e

        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/run_metrics.py \\
            -b "${batch_col[0]}" \\
            -c "${label_col}" \\
            -d "${dataset}" \\
            -m "${integration}" \\
            ${metric} \\
            ${raw_file} \\
            ${integrated_file} \\
            "${dataset}-${integration}-${metric}-${hex}.csv" \\
            ${metric_params.collect { k, v -> "--${k} ${v}" }.join(' ')}
        
        exit_code=\$?

        if [ -e "${dataset}-${integration}-${metric}.csv" ]; then
            exit 0
        fi

        if [ \$exit_code -eq 1 ]; then
            python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/handle_metric_failure.py \\
                -d "${dataset}" \\
                -m "${integration}" \\
                ${metric} \\
                "${dataset}-${integration}-${metric}-${hex}.csv"
            exit 0
        fi

        echo "Publishing to: $params.outdir/metrics/${dataset}/${integration}" >&2
        exit \$exit_code
        """

    stub:
        """
        touch "${dataset}-${integration}-${metric}-${hex}.csv"
        """
}

process ORGANIZE_METRICS {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"
    publishDir "$params.outdir/metrics/"

    memory { get_memory("1.GB", "2.GB", task.attempt, "4.GB") }

    input:
        path files

    output:
        path("all-metrics.csv")

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/metric_ablation/organize_clustering_metrics.py \\
            --out-file "all-metrics.csv" \\
            ${files.join(' ')}
        """
    
    stub:
        """
        touch all-metrics.csv
        """
}


workflow LEIDEN_ABLATION {
    metrics_ch = Channel
        .fromList(params.metrics)
        .map { metric ->
            tuple(
                metric.name,
                metric.params
            )
        }

    // Create a Channel of all parameter combinations
    neighbor_channels = Channel.fromList(params.neighbor_params)
        // Emit each method definition as (name, list_of_param_maps)
        .map { method -> 
            [ method.name, method.params ] 
        }
        // Emit each specific param map within the method definition and calculate combinations
        .map { name, param_maps -> 
            param_maps.collect { param_map -> 
                
                // --- THIS IS THE CORE LOGIC THAT HANDLES ANY PARAMETER NAME ---
                def combinations = param_map.values()
                    .stream()
                    .collect { it } // Get all value lists (e.g., [[5, 10], [1, 2]])
                    .combinations() // Get the Cartesian product
                    .collect { combination ->
                        // Recreate a single map for this specific run
                        def run_params = [:]
                        def keys = param_map.keySet().asList() // Gets keys: [k, seed] or [radius, metric]
                        keys.eachWithIndex { key, i ->
                            run_params[key] = combination[i]
                        }
                        return run_params // Returns [k:5, seed:1] OR [radius:0.1, metric:'euclidean']
                    }
                    .toList()
                    
                return [ name, combinations ]
            }
        }
        .map { it -> it[0] }

    // Flatten the results to get one channel for each run (name, params)
    neighbor_channels
        .flatMap { name, runs -> 
            runs.collect { run_params -> 
                [ name, run_params]
            }
        }
        .set { neighbor_runs_ch }

    neighbor_runs_ch
        .branch { method_name, params ->
            default_neighbors: method_name == "default_neighbors"
                return [method_name, params]
            annoy: method_name == "annoy"
                return [method_name, params]
            exact: method_name == "exact"
                return [method_name, params]
            faiss: method_name == "faiss"
                return [method_name, params]
            flann: method_name == "flann"
                return [method_name, params]
            hnswlib: method_name == "hnswlib"
                return [method_name, params]
        }
        .set { branched_neighbor_runs }

    leiden_ch = Channel.fromList(params.leiden_params)
        .flatMap { param_map -> 
            def combinations = param_map.values()
                .stream()
                .collect { it }
                .combinations()
                .collect { combination ->
                    // Recreate a single map for this specific run
                    def run_params = [:]
                    def keys = param_map.keySet().asList()
                    keys.eachWithIndex { key, i ->
                        run_params[key] = combination[i]
                    }
                    return run_params
                }
                .toList()
            
            return combinations
        }

    integrated_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("work/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad")
            )
        }

    default_neighbors_runs_ch = integrated_ch.combine(branched_neighbor_runs.default_neighbors)
    annoy_runs_ch = integrated_ch.combine(branched_neighbor_runs.annoy)
    exact_runs_ch = integrated_ch.combine(branched_neighbor_runs.exact)
    faiss_runs_ch = integrated_ch.combine(branched_neighbor_runs.faiss)
    flann_runs_ch = integrated_ch.combine(branched_neighbor_runs.flann)
    hnswlib_runs_ch = integrated_ch.combine(branched_neighbor_runs.hnswlib)

    RUN_DEFAULT_NEIGHBORS(default_neighbors_runs_ch)
    RUN_ANNOY_NEIGHBORS(annoy_runs_ch)
    RUN_EXACT_NEIGHBORS(exact_runs_ch)
    RUN_FAISS_NEIGHBORS(faiss_runs_ch)
    RUN_FLANN_NEIGHBORS(flann_runs_ch)
    RUN_HNSWLIB_NEIGHBORS(hnswlib_runs_ch)

    leiden_input_ch = Channel.empty()
        .mix(
            RUN_DEFAULT_NEIGHBORS.out,
            RUN_ANNOY_NEIGHBORS.out,
            RUN_EXACT_NEIGHBORS.out,
            RUN_FAISS_NEIGHBORS.out,
            RUN_FLANN_NEIGHBORS.out,
            RUN_HNSWLIB_NEIGHBORS.out
        )
        .combine(leiden_ch)
        .map { dataset, integration, batch_col, label_col, neighbors_adata, raw_file, leiden_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${neighbors_adata}_${raw_file}_${leiden_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, neighbors_adata, raw_file, leiden_params , hex ]
        }

    RUN_LEIDEN(leiden_input_ch)

    all_ch = RUN_LEIDEN.out
        .combine(metrics_ch)

    RUN_METRIC(all_ch)

    collected_files_ch = RUN_METRIC.out
        .collect { it -> tuple(file(it[2]), file(it[3]))}

    ORGANIZE_METRICS(collected_files_ch)
}


workflow LOUVAIN_ABLATION {
    metrics_ch = Channel
        .fromList(params.metrics)
        .map { metric ->
            tuple(
                metric.name,
                metric.params
            )
        }

    // Create a Channel of all parameter combinations
    neighbor_channels = Channel.fromList(params.neighbor_params)
        // Emit each method definition as (name, list_of_param_maps)
        .map { method -> 
            [ method.name, method.params ] 
        }
        // Emit each specific param map within the method definition and calculate combinations
        .map { name, param_maps -> 
            param_maps.collect { param_map -> 
                
                // --- THIS IS THE CORE LOGIC THAT HANDLES ANY PARAMETER NAME ---
                def combinations = param_map.values()
                    .stream()
                    .collect { it } // Get all value lists (e.g., [[5, 10], [1, 2]])
                    .combinations() // Get the Cartesian product
                    .collect { combination ->
                        // Recreate a single map for this specific run
                        def run_params = [:]
                        def keys = param_map.keySet().asList() // Gets keys: [k, seed] or [radius, metric]
                        keys.eachWithIndex { key, i ->
                            run_params[key] = combination[i]
                        }
                        return run_params // Returns [k:5, seed:1] OR [radius:0.1, metric:'euclidean']
                    }
                    .toList()
                    
                return [ name, combinations ]
            }
        }
        .map { it -> it[0] }

    // Flatten the results to get one channel for each run (name, params)
    neighbor_channels
        .flatMap { name, runs -> 
            runs.collect { run_params -> 
                [ name, run_params]
            }
        }
        .set { neighbor_runs_ch }

    neighbor_runs_ch
        .branch { method_name, params ->
            default_neighbors: method_name == "default_neighbors"
                return [method_name, params]
            annoy: method_name == "annoy"
                return [method_name, params]
            exact: method_name == "exact"
                return [method_name, params]
            faiss: method_name == "faiss"
                return [method_name, params]
            flann: method_name == "flann"
                return [method_name, params]
            hnswlib: method_name == "hnswlib"
                return [method_name, params]
        }
        .set { branched_neighbor_runs }

    louvain_ch = Channel.fromList(params.louvain_params)
        .flatMap { param_map -> 
            def combinations = param_map.values()
                .stream()
                .collect { it }
                .combinations()
                .collect { combination ->
                    // Recreate a single map for this specific run
                    def run_params = [:]
                    def keys = param_map.keySet().asList()
                    keys.eachWithIndex { key, i ->
                        run_params[key] = combination[i]
                    }
                    return run_params
                }
                .toList()
            
            return combinations
        }

    integrated_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("work/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad")
            )
        }

    default_neighbors_runs_ch = integrated_ch.combine(branched_neighbor_runs.default_neighbors)
    annoy_runs_ch = integrated_ch.combine(branched_neighbor_runs.annoy)
    exact_runs_ch = integrated_ch.combine(branched_neighbor_runs.exact)
    faiss_runs_ch = integrated_ch.combine(branched_neighbor_runs.faiss)
    flann_runs_ch = integrated_ch.combine(branched_neighbor_runs.flann)
    hnswlib_runs_ch = integrated_ch.combine(branched_neighbor_runs.hnswlib)

    RUN_DEFAULT_NEIGHBORS(default_neighbors_runs_ch)
    RUN_ANNOY_NEIGHBORS(annoy_runs_ch)
    RUN_EXACT_NEIGHBORS(exact_runs_ch)
    RUN_FAISS_NEIGHBORS(faiss_runs_ch)
    RUN_FLANN_NEIGHBORS(flann_runs_ch)
    RUN_HNSWLIB_NEIGHBORS(hnswlib_runs_ch)

    louvain_input_ch = Channel.empty()
        .mix(
            RUN_DEFAULT_NEIGHBORS.out,
            RUN_ANNOY_NEIGHBORS.out,
            RUN_EXACT_NEIGHBORS.out,
            RUN_FAISS_NEIGHBORS.out,
            RUN_FLANN_NEIGHBORS.out,
            RUN_HNSWLIB_NEIGHBORS.out
        )
        .combine(louvain_ch)
        .map { dataset, integration, batch_col, label_col, neighbors_adata, raw_file, louvain_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${neighbors_adata}_${raw_file}_${louvain_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, neighbors_adata, raw_file, louvain_params , hex ]
        }
    // louvain_input_ch.view()
    RUN_LOUVAIN(louvain_input_ch)

    all_ch = RUN_LOUVAIN.out
        .combine(metrics_ch)

    RUN_METRIC(all_ch)

    collected_files_ch = RUN_METRIC.out
        .collect { it -> tuple(file(it[2]), file(it[3]))}

    ORGANIZE_METRICS(collected_files_ch)
}


workflow KMEANS_ABLATION {
    metrics_ch = Channel
        .fromList(params.metrics)
        .map { metric ->
            tuple(
                metric.name,
                metric.params
            )
        }

    kmeans_ch = Channel.fromList(params.kmeans_params)
        .flatMap { param_map -> 
            def combinations = param_map.values()
                .stream()
                .collect { it }
                .combinations()
                .collect { combination ->
                    // Recreate a single map for this specific run
                    def run_params = [:]
                    def keys = param_map.keySet().asList()
                    keys.eachWithIndex { key, i ->
                        run_params[key] = combination[i]
                    }
                    return run_params
                }
                .toList()
            
            return combinations
        }

    integrated_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("work/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad")
            )
        }

    kmeans_input_ch = integrated_ch
        .combine(kmeans_ch)
        .map { dataset, integration, batch_col, label_col, integrated_adata, raw_file, kmeans_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${integrated_adata}_${raw_file}_${kmeans_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, integrated_adata, raw_file, kmeans_params , hex ]
        }    
    RUN_KMEANS(kmeans_input_ch)

    all_ch = RUN_KMEANS.out
        .combine(metrics_ch)

    RUN_METRIC(all_ch)

    collected_files_ch = RUN_METRIC.out
        .collect { it -> tuple(file(it[2]), file(it[3]))}

    ORGANIZE_METRICS(collected_files_ch)
}


workflow GMM_ABLATION {
    metrics_ch = Channel
        .fromList(params.metrics)
        .map { metric ->
            tuple(
                metric.name,
                metric.params
            )
        }

    gmm_ch = Channel.fromList(params.gmm_params)
        .flatMap { param_map -> 
            def combinations = param_map.values()
                .stream()
                .collect { it }
                .combinations()
                .collect { combination ->
                    // Recreate a single map for this specific run
                    def run_params = [:]
                    def keys = param_map.keySet().asList()
                    keys.eachWithIndex { key, i ->
                        run_params[key] = combination[i]
                    }
                    return run_params
                }
                .toList()
            return combinations
        }

    integrated_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("work/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad")
            )
        }

    gmm_input_ch = integrated_ch
        .combine(gmm_ch)
        .map { dataset, integration, batch_col, label_col, integrated_adata, raw_file, gmm_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${integrated_adata}_${raw_file}_${gmm_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, integrated_adata, raw_file, gmm_params , hex ]
        }    
    RUN_GMM(gmm_input_ch)

    all_ch = RUN_GMM.out
        .combine(metrics_ch)

    RUN_METRIC(all_ch)

    collected_files_ch = RUN_METRIC.out
        .collect { it -> tuple(file(it[2]), file(it[3]))}

    ORGANIZE_METRICS(collected_files_ch)
}



def get_memory(file_size, mem_per_gb = "1.GB", attempt = 1, overhead = "8.GB") {
    file_mem = new nextflow.util.MemoryUnit(file_size)
    mem_per_gb = new nextflow.util.MemoryUnit(mem_per_gb)
    overhead = new nextflow.util.MemoryUnit(overhead)
    max_mem = new nextflow.util.MemoryUnit(params.max_memory)

    file_gb = file_mem.toGiga()
    file_gb = file_gb > 0 ? file_gb : 1
    mem_use = (mem_per_gb * file_gb * attempt) + overhead

    if (mem_use > max_mem) {
        mem_use = max_mem
    }

    return mem_use
}