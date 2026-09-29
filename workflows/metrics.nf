import groovy.json.JsonOutput

process RUN_METRIC {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }

    tag "RUN_METRIC_${integration}_${dataset}_${metric}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}",
        pattern: "*.csv",
        saveAs: { filename -> "${metric}.csv" }

    memory {
        metric == "ALL_CLUSTER" ? 
            get_memory(raw_file.size(), "4.GB", task.attempt, "16.GB") :
        (metric == "CSMC" || metric == "KBET" || metric == "BRAS") ?
            get_memory(raw_file.size(), "8.GB", task.attempt, "24.GB") :
            get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB")
    }

    input:
        tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_file),
            path(raw_file), val(organism), val(metric), val(metric_params)

    output:
        tuple val(dataset), val(integration), path("${dataset}-${integration}-${metric}.csv")

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
            "${dataset}-${integration}-${metric}.csv" \\
            ${metric_params.collect { k, v -> "--${k}=${v}" }.join(' ')}
        
        exit_code=\$?

        if [ -e "${dataset}-${integration}-${metric}.csv" ]; then
            exit 0
        fi

        if [ \$exit_code -eq 1 ]; then
            python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/handle_metric_failure.py \\
                -d "${dataset}" \\
                -m "${integration}" \\
                ${metric} \\
                "${dataset}-${integration}-${metric}.csv"
            exit 0
        fi

        echo "Publishing to: $params.outdir/metrics/${dataset}/${integration}" >&2
        exit \$exit_code
        """

    stub:
        """
        touch "${dataset}-${integration}-${metric}.csv"
        """
}


process RUN_METRIC_FINAL {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }

    tag "RUN_METRIC_${integration}_${dataset}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}",
        pattern: "*.csv",
        saveAs: { filename -> "metrics.csv" }

    memory {
        get_memory(raw_file.size(), "4.GB", task.attempt, "16.GB")
    }

    input:
        tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_file),
            path(raw_file), val(metric_params)

    output:
        tuple val(dataset), val(integration), path("${dataset}-${integration}.csv")

    script:
        def json_str = JsonOutput.toJson(metric_params)
        """
        set +e
        echo ${json_str}

        python /ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/run_metrics_final.py \\
            -b "${batch_col[0]}" \\
            -c "${label_col}" \\
            -d "${dataset}" \\
            -m "${integration}" \\
            -p '${json_str}' \\
            ${raw_file} \\
            ${integrated_file} \\
            "${dataset}-${integration}.csv" \\

        """

    stub:
        """
        touch "${dataset}-${integration}.csv"
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

workflow METRICS {
    take:
        integrated_ch
        metrics_ch
    
    main:
        all_ch = integrated_ch
            .combine(metrics_ch)

        RUN_METRIC(all_ch)

        collected_metrics_ch = RUN_METRIC.out
            .collect { it -> file(it[2])}
            .view()

        COMBINE_METRICS(collected_metrics_ch)

    emit:
        calculated_metric_sh = RUN_METRIC.out
            .map( it -> file(it[2]) )

}

workflow METRICS_FINAL {
    take:
        integrated_ch    
    main:
        RUN_METRIC_FINAL(integrated_ch)

        collected_metrics_ch = RUN_METRIC_FINAL.out
            .collect { it -> file(it[2])}
            .view()

        COMBINE_METRICS(collected_metrics_ch)
    emit:
        calculated_metric_sh = RUN_METRIC_FINAL.out
            .map( it -> file(it[2]) )

}



/*
========================================================================================
    FUNCTIONS
========================================================================================
*/

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

def check_max_time(obj) {
    try {
        if (obj.compareTo(params.max_time as nextflow.util.Duration) == 1)
            return params.max_time as nextflow.util.Duration
        else
            return obj
    } catch (all) {
        println "   ### ERROR ###   Max time '${params.max_time}' is not valid! Using default value: $obj"
        return obj
    }
}
