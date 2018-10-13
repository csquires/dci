library('argparse')
library('yaml')
library('foreach')
library('doParallel')

cores <- detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

source('config.R')

parser <- ArgumentParser(description='Run the GES algorithm on a dataset')
parser$add_argument('--folder', type='character', help='Folder name of data set')
parser$add_argument('--nsamples', type='integer', help='Number of samples to use in analyzing results')

args <- parser$parse_args()
dataset_folder = file.path(DATA_FOLDER, args$folder)
dataset_config = read_yaml(file.path(dataset_folder, 'config.yaml'))


nothing<-foreach(i=0:(dataset_config$npairs-1)) %dopar% {
    library('pcalg')
    library('GetoptLong')

    nsamples <- args$nsamples
    samples_folder = file.path(dataset_folder, qq('pair@{i}'),qq('samples_n=@{args$nsamples}') )
    X1 = read.table(file.path(samples_folder, 'X1.txt'))
    X2 = read.table(file.path(samples_folder, 'X2.txt'))
    l0score1 = new("GaussL0penObsScore", data = X1, intercept = FALSE)
    l0score2 = new("GaussL0penObsScore", data = X2, intercept = FALSE)

    RES_FOLDER = file.path(samples_folder, 'results', 'ges')
    dir.create(RES_FOLDER, recursive=TRUE)
    cpdag1 = ges(score=l0score1, labels=colnames(X1))
    cpdag2 = ges(score=l0score2, labels=colnames(X2))
    write(as(as(as(cpdag1$essgraph, "graphNEL"), "graphAM"), "matrix"), file=file.path(RES_FOLDER, 'A1.txt'), ncolumns=ncol(X1))
    write(as(as(as(cpdag2$essgraph, "graphNEL"), "graphAM"), "matrix"), file=file.path(RES_FOLDER, 'A2.txt'), ncolumns=ncol(X2))
}