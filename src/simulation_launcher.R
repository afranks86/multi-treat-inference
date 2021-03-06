library(lubridate)
library(parallel)
## RSCRIPT_ALIAS <- "/opt/R/3.5.3/bin/Rscript"
RSCRIPT_ALIAS <- "/opt/conda/bin/Rscript"

model <- 1:5
n <- c(100, 1000)

k <- c(5, 10, 20)
m <- c(2)

frac_non_null <- c(0.1, 0.2, 0.5, 0.8)

all_settings <- expand.grid(model, n, k, m)


all_settings$frac_non_null <- -1
sparse_model_indices <- which(all_settings[, 1]==4)

tmp <- all_settings[rep(sparse_model_indices, each=length(frac_non_null)), ]

option_names <- c('model', 'n', 'k', 'm')
option_types <- c('%i', '%i', '%i', '%i')
option_fstring <- paste('--', option_names, '=', option_types, collapse=' ', sep='')

script_fstring <- paste(RSCRIPT_ALIAS, "run_simulation.R", option_fstring)

logfile_options <- paste(option_names, option_types, collapse='_', sep='')
logfile_fstring <- paste("logs/log_", logfile_options, "_%s.log", sep='')

run_setting <- function(row){
    row <- unlist(row)
    model <- row[1]
    n <- row[2]
    k <- row[3]
    m <- row[4]

    call <- sprintf(script_fstring, model, n, k, m)
    print(call)
    logfile <- sprintf(logfile_fstring, model, n, k, m,
                       gsub(" ", "", today(), fixed=TRUE))
    system(paste(call, ">", logfile, "2>&1"))
}


retcodes <- mclapply(1:nrow(all_settings),
         function(i){
             run_setting(all_settings[i,])
         }, mc.cores=detectCores())

print(retcodes)
save(retcodes, all_settings,
     file=paste("logs/experiment_exit_status_",
                gsub(" ", "", now(), fixed=TRUE), ".log", sep=""))
