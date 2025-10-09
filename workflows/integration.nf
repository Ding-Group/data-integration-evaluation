/*
========================================================================================
    PROCESSES
========================================================================================
*/

process RUN_INTEGRATE {

    publishDir "$params.outdir/integration-models/${dataset}/${integration}",
        pattern: "save_dir",
        saveAs: { pathname -> pathname + "-${seed}" }

    // label "process_medium"

    memory { get_memory(data.size(), "6.GB", task.attempt) }

    input:
        tuple val(dataset), val(batch_col), val(label_col), val(organism), path(data), val(seed),
            val(integration), val(conda_env), val(integration_params), path(run_script)

    output:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col), path(data), path("save_dir"), val(organism)

    script:
        """
        source /ubc/cs/research/beaver/projects/yinian/miniconda3/etc/profile.d/conda.sh
        conda activate ${conda_env}
        mkdir save_dir
        python ${run_script} \\
            -o save_dir \\
            -b ${batch_col.join(' -b')} \\
            -c ${label_col} \\
            --seed ${seed} \\
            ${data}
        """

    stub:
        """
        mkdir save_dir
        touch save_dir/adata.h5ad
        """
}

/*
========================================================================================
    WORKFLOW
========================================================================================
*/

workflow INTEGRATION {

    take:
        datasets_ch
        integrations_ch

    main:

        all_integrations_ch = datasets_ch
            .combine(Channel.fromList(params.seeds))
            .combine(integrations_ch)

        RUN_INTEGRATE(all_integrations_ch)

    emit:
        integrated_ch = RUN_INTEGRATE.out
            .map { it ->
                tuple(
                    it[0],                       // Dataset name
                    it[1] + "-" + it[2],         // Integration name
                    it[3],                       // Batch columns
                    it[4],                       // Cell type label column
                    file(it[6] + "/adata.h5ad"), // Path to reference H5AD
                    it[5],                       // Path to reference expression H5AD
                    it[7]                        // Organism
                )
            }
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