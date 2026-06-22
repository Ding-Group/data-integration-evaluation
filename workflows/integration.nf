/*
========================================================================================
    PROCESSES
========================================================================================
*/

process RUN_INTEGRATE {
    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= 2) ? 'retry' : 'ignore'
    }
    // errorStrategy 'retry'

    tag "RUN_INTEGRATE_${integration}_${dataset}_${seed}"

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
            get_memory(data.size(), "16.GB", task.attempt, "48.GB")
        println "[${task.process}] Memory requested: ${mem}"
        return mem
    }

    resourceLimits memory: "256.GB"

    input:
        tuple val(dataset), val(batch_col), val(label_col), val(organism), path(data), val(seed),
            val(integration), val(conda_env), val(integration_params), val(memory_usage), path(run_script)

    output:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col), path(data), path("save_dir"), val(organism)

    script:
        def params_map = integration_params as Map
        def param_str = params_map.collect { k, v -> "--${k}=${v}" }.join(' ')
        """
        source ${params.conda_location}
        conda activate ${conda_env}
        export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib/R/lib:\${CONDA_PREFIX}/lib:\$LD_LIBRARY_PATH
        export PATH="\${CONDA_PREFIX}/bin:$PATH"
        mkdir save_dir
        echo ${param_str}
        python ${run_script} \\
            -o save_dir \\
            -b ${batch_col.join(' -b ')} \\
            -c ${label_col} \\
            ${param_str} \\
            --seed ${seed} \\
            ${data}
        """

    stub:
        """
        mkdir save_dir
        touch save_dir/adata.h5ad
        """
}


process RUN_INTEGRATE_MULTIOME {
    errorStrategy {
        (task.exitStatus in [135,143,137,104,134,139,140] && task.attempt <= params.maxRetries) ? 'retry' : 'ignore'
    }

    tag "RUN_INTEGRATE_${integration}_${dataset}_${seed}"

    publishDir "$params.outdir/integration-models/${dataset}/${integration}",
        pattern: "save_dir",
        saveAs: { pathname -> pathname + "-${seed}" }

    label params.use_gpu ? 'gpu_process' : null

    memory {
        def mem = memory_usage == 'low' ?
            get_memory(atac_data.size(), "2.GB", task.attempt, "4.GB") :
        memory_usage == 'medium' ?
            get_memory(atac_data.size(), "4.GB", task.attempt, "8.GB") :
        memory_usage == 'high' ?
            get_memory(atac_data.size(), "10.GB", task.attempt, "16.GB") :
            get_memory(atac_data.size(), "24.GB", task.attempt, "48.GB")

        println "[${task.process}] Memory requested: ${mem}"
        return mem
    }

    resourceLimits memory: "256.GB"

    input:
        tuple val(dataset), val(batch_col), val(label_col), val(organism), path(rna_data), path(atac_data), val(seed),
            val(integration), val(conda_env), val(integration_params), val(memory_usage), path(run_script)

    output:
        tuple val(dataset), val(integration), val(seed), val(batch_col), val(label_col), path(rna_data), path("save_dir"), val(organism)

    script:
        def params_map = integration_params as Map
        def param_str = params_map.collect { k, v -> "--${k}=${v}" }.join(' ')
        """
        source ${params.conda_location}
        conda activate ${conda_env}
        export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib/R/lib:\${CONDA_PREFIX}/lib:\$LD_LIBRARY_PATH
        export PATH="\${CONDA_PREFIX}/bin:$PATH"
        mkdir save_dir
        echo ${param_str}
        python ${run_script} \\
            -o save_dir \\
            -b ${batch_col.join(' -b ')} \\
            -c ${label_col} \\
            ${param_str} \\
            --seed ${seed} \\
            ${rna_data} \\
            ${atac_data}
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
                    file(it[6] + "/adata.h5ad"), // Path to integrated H5AD
                    it[5],                       // Path to reference expression H5AD
                    it[7]                        // Organism
                )
            }
}


workflow INTEGRATION_MULTIOME {

    take:
        multiome_datasets_ch
        integrations_ch

    main:

        all_integrations_ch = multiome_datasets_ch
            .combine(Channel.fromList(params.seeds))
            .combine(integrations_ch)

        RUN_INTEGRATE_MULTIOME(all_integrations_ch)

    emit:
        integrated_ch = RUN_INTEGRATE_MULTIOME.out
            .map { it ->
                tuple(
                    it[0],                       // Dataset name
                    it[1] + "-" + it[2],         // Integration name
                    it[3],                       // Batch columns
                    it[4],                       // Cell type label column
                    file(it[6] + "/adata.h5ad"), // Path to integrated H5AD
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
    mem_use = (mem_per_gb * file_gb * attempt) + overhead * attempt

    if (mem_use > max_mem) {
        mem_use = max_mem
    }

    return mem_use
}