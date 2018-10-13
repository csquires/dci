library('argparse')
library('yaml')
library('foreach')
library('doParallel')

cores <- detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

source('config.R')

parser <- ArgumentParser(description='Run the PC algorithm on a dataset')
parser$add_argument('--folder', type='character', help='Folder name of data set')
parser$add_argument('--nsamples', type='integer', help='Number of samples to use in analyzing results')
parser$add_argument('--alphas', nargs='+', type='double', help='Alpha value to run PCALG with')

args <- parser$parse_args()
dataset_folder = file.path(DATA_FOLDER, args$folder)
dataset_config = read_yaml(file.path(dataset_folder, 'config.yaml'))

nothing<-foreach(i=0:(dataset_config$npairs-1)) %dopar% {
    library('pcalg')
    library('GetoptLong')

    samples_folder = file.path(dataset_folder, qq('pair@{i}'),qq('samples_n=@{args$nsamples}') )
    X1 = read.table(file.path(samples_folder, 'X1.txt'))
    X2 = read.table(file.path(samples_folder, 'X2.txt'))
    S1 = list(C = cor(X1), n = nrow(X1))
    S2 = list(C = cor(X2), n = nrow(X2))

    for (alpha in args$alphas) {
        alpha_str = sprintf('%4.2e', alpha)
        RES_FOLDER = file.path(samples_folder, 'results', 'pcalg', qq('alpha=@{alpha_str}'))
        dir.create(RES_FOLDER, recursive=TRUE)
        cpdag1 = pc(suffStat = S1, indepTest=gaussCItest, alpha=alpha, labels=colnames(X1))
        cpdag2 = pc(suffStat = S2, indepTest=gaussCItest, alpha=alpha, labels=colnames(X2))
        write(as(cpdag1, 'amat'), file=file.path(RES_FOLDER, 'A1.txt'), ncolumns=ncol(X1))
        write(as(cpdag2, 'amat'), file=file.path(RES_FOLDER, 'A2.txt'), ncolumns=ncol(X2))
    }
}