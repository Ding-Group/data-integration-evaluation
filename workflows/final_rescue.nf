// Using pre-computed integrations, perform the final benchmarking using the eight core metrics
import java.security.MessageDigest
import java.math.BigInteger
import groovy.json.JsonOutput

include { COMBINE_METRICS } from "./metrics"

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
        get_memory(raw_file.size(), "4.GB", task.attempt, "12.GB")
    }

    input:
        tuple val(dataset), val(integration), val(batch_col), val(label_col), path(integrated_file),
            path(raw_file), val(hex), val(metric_params)

    output:
        tuple val(dataset), val(integration), path("${dataset}-${integration}-${hex}.csv")

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
            "${dataset}-${integration}-${hex}.csv"
        """

    stub:
        """
        touch "${dataset}-${integration}-${hex}.csv"
        """
}



workflow FINAL_RESCUE {
    integrations_ch = Channel
        .fromList(params.integrations)
        .map { integration -> 
            tuple(
                integration.dataset,
                integration.integration,
                integration.batch_col,
                integration.label_col,
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/" + integration.dir + "/save_dir/adata.h5ad"),
                file("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/" + integration.dataset + ".h5ad"),
                integration.metric_params
            )
        }


    runs_ch = integrations_ch
        .map { dataset, integration, batch_col, label_col, integrated_adata, raw_file, metric_params ->
            def deterministic_key = "${dataset}_${integration}_${batch_col}_${label_col}_${integrated_adata}_${raw_file}_${metric_params}"
            MessageDigest md = MessageDigest.getInstance("SHA-1")
            md.update(deterministic_key.getBytes('UTF-8'))
            def hex = new BigInteger(1, md.digest()).toString(16).padLeft(40, '0')
            [ dataset, integration, batch_col, label_col, integrated_adata, raw_file, hex, metric_params ]
        }
    RUN_METRIC_FINAL(runs_ch)

    collected_metrics_ch = RUN_METRIC_FINAL.out
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