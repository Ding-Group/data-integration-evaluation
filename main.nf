#!/usr/bin/env nextflow
/*
========================================================================================
    Single-cell integration metrics
========================================================================================
    GitHub : 
    Website:
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    WORKFLOWS
========================================================================================
*/

include { INTEGRATION } from './workflows/integration'
include { METRICS } from './workflows/metrics'
// include { REPORTS } from './workflows/reports'

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
            dataset.organism,
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
            integration.memory,  // Whether high or low memory usage
            file(params.bindir + '/run_' + integration.name + '.py')  // Script to run method
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

workflow WF_ALL {
    INTEGRATION(datasets_ch, integrations_ch)
    METRICS(INTEGRATION.out.integrated_ch, metrics_ch)
}

/*
========================================================================================
    DEFAULT WORKFLOW
========================================================================================
*/

//
// Implicit workflow executed when no entry is specified
//
workflow {
    WF_MAIN ()
}