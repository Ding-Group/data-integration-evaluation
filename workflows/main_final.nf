/*
 * Alternative main NF file for the final benchmarking
 */ 

nextflow.enable.dsl = 2

/*
========================================================================================
    WORKFLOWS
========================================================================================
*/

include { INTEGRATION_FINAL } from './integration'
include { INT_FINAL_OPT } from './integration'

include { METRICS_FINAL } from './metrics'

/*
 * Channel that streams in all the datasets from the 
 * YAML configuration file.
 */
datasets_ch = Channel
    .fromList(params.datasets)
    .map { dataset ->
        tuple(
            dataset.name,
            dataset.batch_col,
            dataset.label_col,
            file(params.datadir + dataset.name + ".h5ad")
        )
    }

integrations_ch = Channel
    .fromList(params.integration)
    .map { integration -> 
        tuple(
            integration.name,    // Method name
            integration.conda,   // Location of Conda environment
            integration.params,  // Method hyperparameters
            integration.metric_params, // Metric hyperparameters for this specific integration method
            integration.memory,  // Whether high or low memory usage
            file(params.bindir + '/run_' + integration.name + '.py')  // Script to run method
        )
    }

workflow WF_ALL {
    INTEGRATION_FINAL(datasets_ch, integrations_ch)
    METRICS_FINAL(INTEGRATION_FINAL.out.integrated_ch)
}

workflow WF_OPTUNA {
    INT_FINAL_OPT(datasets_ch, integrations_ch)
    METRICS_FINAL(INT_FINAL_OPT.out.integrated_ch)
}