import java.security.MessageDigest
import java.math.BigInteger

include { COMBINE_METRICS } from "./metrics"

process RUN_METRIC {
    conda "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/"

    tag "RUN_METRIC_${integration}_${dataset}_${metric}"

    publishDir "$params.outdir/metrics/${dataset}/${integration}",
        pattern: "*.csv",
        saveAs: { filename -> "${metric}.csv" }

    memory {
        (metric == "ALL_CLUSTER" || metric == 'CSMC') ? 
            get_memory(raw_file.size(), "4.GB", task.attempt, "8.GB") :
        metric == "KBET" ?
            get_memory(raw_file.size(), "4.GB", task.attempt, "48.GB") :
            get_memory(raw_file.size(), "2.GB", task.attempt, "4.GB")
    }

    label "error_retry"

    input:
        tuple val(dataset), val(integration), val(batch_col), val(label_col), file(integrated_file),
            file(raw_file), val(hex), val(metric), val(metric_params)

    output:
        tuple val(dataset), val(integration), path("${dataset}-${integration}-${metric}-${hex}.csv")

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
            ${metric_params.collect { k, v -> "--${k}=${v}" }.join(' ')}
        
        exit_code=\$?

        if [ -e "${dataset}-${integration}-${metric}-${hex}.csv" ]; then
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

workflow GRANULARITY {
    integrations_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad")
            )
        }
    metrics_ch = Channel
        .fromList(params.metrics)
        .map { metric ->
            tuple(
                metric.name,
                metric.params
            )
        }

    runs_ch = integrations_ch.combine(metrics_ch)
    runs_ch = runs_ch
        .map { dataset, integration, batch_col, label_col, neighbors_adata, raw_file, metric_name, metric_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${neighbors_adata}_${raw_file}_${metric_name}_${metric_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, neighbors_adata, raw_file, hex, metric_name, metric_params ]
        }
    RUN_METRIC(runs_ch)

    collected_metrics_ch = RUN_METRIC.out
        .collect { it -> file(it[2])}

    COMBINE_METRICS(collected_metrics_ch)
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