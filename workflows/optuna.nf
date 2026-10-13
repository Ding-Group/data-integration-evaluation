import groovy.json.JsonOutput

process RUN_INT_F_OPT {
    // errorStrategy {
    //     (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    // }
    errorStrategy 'ignore'

    tag "TRAIN_${integration}_${dataset}_${seed}_t${trial_idx}"

    publishDir "$params.outdir/integration-models/${dataset}/${integration}",
        pattern: "save_dir",
        saveAs: { pathname -> pathname + "-${seed}" }

    label params.use_gpu ? 'gpu_process' : null

    memory {
        def mem = memory_usage == 'low' ?
            get_memory(data.size(), "2.GB", task.attempt, "8.GB") :
        memory_usage == 'medium' ?
            get_memory(data.size(), "4.GB", task.attempt, "12.GB") :
        memory_usage == 'high' ?
            get_memory(data.size(), "10.GB", task.attempt, "24.GB") :
            get_memory(data.size(), "12.GB", task.attempt, "72.GB")
        return mem
    }

    resourceLimits memory: "256.GB"

    input:
        tuple val(dataset), val(integration), val(seed), val(trial_idx),
            val(batch_col), val(label_col), path(data), val(conda_env),
            val(metric_params), val(memory_usage), path(run_script), path(ask_script)

    output:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col),
            path(data), path("save_dir"), val(metric_params), path("params_${trial_idx}.json")

    script:
        def study_dir = file("${params.outdir}/integration-models/${dataset}/${integration}-${seed}").toAbsolutePath()
        """
        source ${params.conda_location}
        conda activate scmetrics
        python ${ask_script} \\
            --study-dir "${study_dir}" \\
            --out params_${trial_idx}.json

        mkdir save_dir

        set +u
        conda activate ${conda_env}
        set -u
        export PATH="\${CONDA_PREFIX}/bin:$PATH"

        python ${run_script} \\
            -o save_dir \\
            -b ${batch_col.join(' -b ')} \\
            -c ${label_col} \\
            --params-json params_${trial_idx}.json \\
            --seed ${seed} \\
            ${data}
        """

    stub:
        """
        echo '{"trial_number": ${trial_idx}}' > params_${trial_idx}.json
        mkdir save_dir
        touch save_dir/adata.h5ad
        """
}

process RUN_METRIC_FINAL {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }

    tag "SCORE_${integration}_${dataset}_${seed}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}-${seed}",
        pattern: "*.csv",
        saveAs: { filename -> "trial_metrics.csv" }

    memory {
        get_memory(raw_file.size(), "4.GB", task.attempt, "16.GB")
    }

    input:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col),
            path(raw_file), path(integrated_dir), val(metric_params), path(params_json)

    output:
        tuple val(dataset), val(integration), val(seed), path("${dataset}-${integration}-${seed}.csv")

    script:
        def json_str = JsonOutput.toJson(metric_params)
        def study_dir = file("${params.outdir}/integration-models/${dataset}/${integration}-${seed}").toAbsolutePath()
        """
        set +e
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/run_metrics_final_optuna.py \\
            -b "${batch_col[0]}" \\
            -c "${label_col}" \\
            -d "${dataset}" \\
            -m "${integration}" \\
            -p '${json_str}' \\
            --study-dir "${study_dir}" \\
            --params-json ${params_json} \\
            ${raw_file} \\
            ${integrated_dir}/adata.h5ad \\
            "${dataset}-${integration}-${seed}.csv"
        """

    stub:
        """
        touch "${dataset}-${integration}-${seed}.csv"
        """
}


process FINALIZE_BEST {
    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }

    tag "FINALIZE_${integration}_${dataset}_${seed}"

    publishDir "$params.outdir/final/${dataset}/${integration}",
        pattern: "adata.h5ad",
        saveAs: { filename -> "adata-${seed}.h5ad" }

    publishDir "$params.outdir/final/${dataset}/${integration}",
        pattern: "best_params.json",
        saveAs: { filename -> "best_params-${seed}.json" }

    publishDir "$params.outdir/final/${dataset}/${integration}",
        pattern: "best_metrics.json",
        saveAs: { filename -> "best_metrics-${seed}.json" }

    label params.use_gpu ? 'gpu_process' : null

    memory {
        def mem = memory_usage == 'low' ?
            get_memory(data.size(), "2.GB", task.attempt, "8.GB") :
        memory_usage == 'medium' ?
            get_memory(data.size(), "4.GB", task.attempt, "12.GB") :
        memory_usage == 'high' ?
            get_memory(data.size(), "10.GB", task.attempt, "24.GB") :
            get_memory(data.size(), "16.GB", task.attempt, "48.GB")
        return mem
    }

    resourceLimits memory: "256.GB"

    input:
        tuple val(dataset), val(integration), val(seed),
            val(batch_col), val(label_col), path(data),
            val(conda_env), val(metric_params), val(memory_usage),
            path(run_script), val(all_trial_csvs)   // all_trial_csvs forces this
                                                      // process to wait until every
                                                      // trial's SCORE step has run

    output:
        tuple val(dataset), val(integration), val(seed),
            path("adata.h5ad"), path("best_params.json"), path("best_metrics.json")

    script:
        def study_dir = file("${params.outdir}/integration-models/${dataset}/${integration}-${seed}").toAbsolutePath()
        """
        source ${params.conda_location}
        set +u
        conda activate scmetrics
        set -u
        export PATH="\${CONDA_PREFIX}/bin:$PATH"

        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/get_best_${integration}.py \\
            --study-dir "${study_dir}" \\
            --out best_params.json \\
            --best-metrics-out best_metrics.json

        conda deactivate
        set +u
        conda activate ${conda_env}
        set -u
        export PATH="\${CONDA_PREFIX}/bin:$PATH"

        python ${run_script} \\
            -o . \\
            -b ${batch_col.join(' -b ')} \\
            -c ${label_col} \\
            --params-json best_params.json \\
            --seed ${seed} \\
            ${data}
        """

    stub:
        """
        touch adata.h5ad best_params.json best_metrics.json
        """
}


process FINAL_METRIC {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }

    tag "FINAL_METRIC_${integration}_${dataset}_${seed}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}-${seed}",
        pattern: "*.csv",
        saveAs: { filename -> "final-metrics.csv" }

    memory {
        get_memory(raw_file.size(), "4.GB", task.attempt, "16.GB")
    }

    input:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col),
            path(raw_file), path(final_adata), val(metric_params)

    output:
        path("${dataset}-${integration}-${seed}.csv")

    script:
        def json_str = JsonOutput.toJson(metric_params)
        """
        set +e
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/run_metrics_final.py \\
            -b "${batch_col[0]}" \\
            -c "${label_col}" \\
            -d "${dataset}" \\
            -m "${integration}-${seed}" \\
            -p '${json_str}' \\
            ${raw_file} \\
            ${final_adata} \\
            "${dataset}-${integration}-${seed}.csv"
        """

    stub:
        """
        touch "${dataset}-${integration}-${seed}.csv"
        """
}


process COMBINE_METRICS {
    cache false

    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    publishDir "$params.outdir/metrics/"

    memory '4 GB'

    input:
        path metrics
    
    output:
        tuple path("all-metrics.csv"), path("missing-summary.csv"), path("missing-metrics.txt")

    script:
        """
        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/combine_metrics.py \\
            --out-file "all-metrics.csv" \\
            --missing-values "missing-summary.csv" \\
            --missing-files "missing-metrics.txt" \\
            ${metrics.join(' ')}
        """

    stub:
        """
        touch "all-metrics.tsv"
        touch "missing-summary.tsv"
        touch "missing-metrics.txt"
        """
}


workflow WF_OPTUNA {

    // --- datasets channel: one row per dataset entry in the YAML ---
    dataset_ch = Channel.fromList(params.datasets).map { d ->
        tuple(d.name, d.batch_col, d.label_col, file(params.datadir + d.name + ".h5ad"))
    }

    // --- integrations channel: one row per integration entry ---
    integration_ch = Channel.fromList(params.integration).map { i ->
        tuple(i.name, i.conda, i.params, i.metric_params, i.memory)
    }

    // --- seeds ---
    seed_ch = Channel.fromList(params.seeds)

    // cross product: dataset x integration x seed
    runs_ch = dataset_ch
        .combine(integration_ch)
        .combine(seed_ch)
        .map { dataset, batch_col, label_col, data,
               integration, conda_env, integration_params, metric_params, memory_usage,
               seed ->
            tuple(dataset, batch_col, label_col, data, seed, integration,
                  conda_env, metric_params, memory_usage,
                  file("${params.bindir}/run_${integration}.py"),
                  file("${params.bindir}/ask_trial_${integration}.py")
                )
        }

    // fan out over n_trials
    trial_ch = runs_ch
        .combine(Channel.of(0..<params.n_trials))
        .map { dataset, batch_col, label_col, data, seed, integration,
               conda_env, metric_params, memory_usage, run_script, ask_script, trial_idx ->
            tuple(dataset, integration, seed, trial_idx,
                  batch_col, label_col, data, conda_env,
                  metric_params, memory_usage, run_script, ask_script)
        }

    // asked_ch   = ASK_TRIAL(trial_ch)
    trained_ch = RUN_INT_F_OPT(trial_ch)
    scored_ch  = RUN_METRIC_FINAL(trained_ch)

    grouped_trials_ch = scored_ch
        .map { dataset, integration, seed, csv -> tuple(dataset, integration, seed, csv) }
        .groupTuple(by: [0, 1, 2])
        // -> tuple(dataset, integration, seed, [csv, csv, ...])

    // --- build FINALIZE_BEST's input exactly, in its declared order ---
    finalize_input_ch = grouped_trials_ch
        .join(
            runs_ch.map { dataset, batch_col, label_col, data, seed, integration,
                          conda_env, metric_params, memory_usage, run_script, ask_script ->
                tuple(dataset, integration, seed,
                      batch_col, label_col, data,
                      conda_env, metric_params, memory_usage, run_script)
            },
            by: [0, 1, 2]
        )
        .map { dataset, integration, seed, all_trial_csvs,
               batch_col, label_col, data,
               conda_env, metric_params, memory_usage, run_script ->
            tuple(dataset, integration, seed,
                  batch_col, label_col, data,
                  conda_env, metric_params, memory_usage,
                  run_script, all_trial_csvs)
        }

    finalized_ch = FINALIZE_BEST(finalize_input_ch)
    // -> tuple(dataset, integration, seed, adata, best_params_json, best_metrics_json)

    // --- rejoin with batch_col/label_col/raw_file/metric_params for FINAL_METRIC ---
    final_metric_input_ch = finalized_ch
        .map { dataset, integration, seed, adata, best_params_json, best_metrics_json ->
            tuple(dataset, integration, seed, adata)
        }
        .join(
            runs_ch.map { dataset, batch_col, label_col, data, seed, integration,
                          conda_env, metric_params, memory_usage, run_script, ask_script ->
                tuple(dataset, integration, seed, batch_col, label_col, data, metric_params)
            },
            by: [0, 1, 2]
        )
        .map { dataset, integration, seed, adata,
               batch_col, label_col, raw_file, metric_params ->
            tuple(dataset, integration, seed,
                  batch_col, label_col, raw_file, adata, metric_params)
        }

    final_metrics_ch = FINAL_METRIC(final_metric_input_ch)

    // --- combine every final metrics CSV across all combos ---
    COMBINE_METRICS(final_metrics_ch.collect())
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