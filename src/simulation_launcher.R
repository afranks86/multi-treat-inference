library(lubridate)
library(parallel)
RSCRIPT_ALIAS <- "/opt/R/3.5.3/bin/Rscript"

iters <- 100
sigma_y_prior  <- 1:4 ## prior on the outcome variance
beta_prior  <- 1:5 ## prior on beta

all_settings <- expand.grid(sigma_y_prior, beta_prior)

option_names <- c('sigma_y_prior', 'beta_prior')
option_types <- c('%i', '%i')
option_fstring <- paste('--', option_names, '=', option_types, collapse=' ', sep='')

script_fstring <- paste(RSCRIPT_ALIAS, "run_simulation.R", option_fstring, sprintf("--iters=%i", iters))

logfile_options <- paste(option_names, option_types, collapse='_', sep='')
logfile_fstring <- paste("logs/log_", logfile_options, "_%s.log", sep='')

run_setting <- function(row){
    row <- unlist(row)
    n <- row[1]
    p <- row[2]
    cc <- row[3]
    ee <- row[4]
    mm <- row[5]
    aa <- row[6]
    est <- as.logical(row[7])

    call <- sprintf(script_fstring, n, p, cc, ab_dp, ee, mm, aa, est)
    print(call)
    logfile <- sprintf(logfile_fstring, n, p, cc, ab_dp, ee, mm, aa, est,
                       gsub(" ", "", now(), fixed=TRUE))
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
