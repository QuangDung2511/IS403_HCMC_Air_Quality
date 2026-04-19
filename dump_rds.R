library(jsonlite); a <- readRDS('outputs_R/predictions/tuned_tree_preds.rds'); write_json(a, 'outputs/predictions/tuned_tree_preds.json', auto_unbox=TRUE)
