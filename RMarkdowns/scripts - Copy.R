
library(phyloseq)
library(stringr)
library(dplyr)
library(ggplot2); library(magrittr); library(ggfortify) ;require(reshape2)
suppressWarnings({suppressMessages({library(PD16Sdata)
  library(tidyverse)
  library(vegan)
  library(kableExtra)
  library(knitr)
  library(RColorBrewer)
  library(glmnet)
  library(caTools)
  library(readxl)
  library(plotly)
  library(MASS) ; library(plsdof) ; library(caret) ; library(boot); library(robustbase)  
  library(compositions)
}
)
}
)
add_species <- function(seq_species,physeq ){
  
  df2 <- as.data.frame(physeq@tax_table) ; df2$Sequence <- rownames(df2)
  merged_df <- merge(df2,seq_species, by="Sequence", all.x=T) 
  rownames(merged_df) <- merged_df$Sequence  
  merged_df <- merged_df%>% dplyr::select(-Sequence)
  
  physeq@tax_table <- tax_table(as.matrix(merged_df))
  return(physeq)
}

prev.fam2 <- function (data_list, p_thresh, taxa_rank ){
  
  ps <- data_list
  
  ps <- subset_taxa(ps, !is.na(Phylum) & !Phylum %in% c("", "uncharacterized"))
  
  prevdf = apply(X = otu_table(ps),
                 MARGIN = ifelse(taxa_are_rows(ps), yes = 1, no = 2),
                 FUN = function(x){sum(x > 0)})
  
  prevdf = data.frame(Prevalence = prevdf,
                      TotalAbundance = taxa_sums(ps),
                      tax_table(ps))
  
  prevdf1 = subset(prevdf, Phylum %in% get_taxa_unique(ps, "Phylum"))
  
  prevalenceThreshold = floor(p_thresh * nsamples(ps))
  
  keepTaxa = rownames(prevdf1)[(prevdf1$Prevalence >= prevalenceThreshold)]
  
  
  ps = prune_taxa(keepTaxa, ps)
  
  ps_Fam <- tax_glom(ps, taxrank = taxa_rank )
  
  if (taxa_rank == 'Genus'){
    t_rank = 6
  }
  else if (taxa_rank == 'Family'){
    t_rank  = 5
  }
  else if (taxa_rank == 'Order'){
    t_rank = 4
    
  }
  else if (taxa_rank == 'Phylum'){
    t_rank = 2
    
  }
  else if (taxa_rank == 'Species'){
    t_rank = 7
  } 
  
  
  psra_Fam <- transform_sample_counts(ps_Fam, function(x){x / sum(x)})
  colnames(psra_Fam@otu_table) <- paste('P', psra_Fam@tax_table[,2],psra_Fam@tax_table[,t_rank], sep='_')
  
  rownames(psra_Fam@tax_table) <- paste('P', psra_Fam@tax_table[,2],psra_Fam@tax_table[,t_rank], sep='_')
  
  return(psra_Fam)
}
prev.fam <- function (data_list, p_thresh, taxa_rank ){
  
  ps <- data_list
  
  #ps <- subset_taxa(ps, !is.na(Genus) & !Genus %in% c("", "uncharacterized"))
  # 
  # prevdf = apply(X = otu_table(ps),
  #              MARGIN = ifelse(taxa_are_rows(ps), yes = 1, no = 2),
  #              FUN = function(x){sum(x > 0)})
  # 
  # prevdf = data.frame(Prevalence = prevdf,
  #                   TotalAbundance = taxa_sums(ps),
  #                   tax_table(ps))
  # 
  # prevdf1 = subset(prevdf, Phylum %in% get_taxa_unique(ps, "Phylum"))
  # 
  # prevalenceThreshold = floor(p_thresh * nsamples(ps))
  # 
  # keepTaxa = rownames(prevdf1)[(prevdf1$Prevalence >= prevalenceThreshold)]
  # 
  # ps = prune_taxa(keepTaxa, psra)
  
  ps_Fam <- tax_glom(ps, taxrank = taxa_rank )
  
  if (taxa_rank == 'Genus'){
    t_rank = 6
  }
  else if (taxa_rank == 'Family'){
    t_rank  = 5
  }
  else if (taxa_rank == 'Order'){
    t_rank = 4
    
  }
  else if (taxa_rank == 'Phylum'){
    t_rank = 2
    
  }
  psra_Fam <- transform_sample_counts(ps_Fam, function(x){x / sum(x)})
  
  colnames(psra_Fam@otu_table) <- paste('P', psra_Fam@tax_table[,2], substr(taxa_rank,0,1),psra_Fam@tax_table[,t_rank], sep='_')
  
  rownames(psra_Fam@tax_table) <- paste('P', psra_Fam@tax_table[,2], substr(taxa_rank,0,1),psra_Fam@tax_table[,t_rank], sep='_')
  
  
  return(psra_Fam)
}
dl_csv <- function(data_l, name_l, path , train_name){
  
  dir.create(file.path(path, train_name))
  
  path2 <- paste(path,train_name,'', sep = '\\')
  
  
  for(i in 1:length(name_l)){
    
    dat <- data_l[[i]]
    
    datname <- paste(name_l[[i]], '.csv', sep = '')
    
    
    write.csv(dat, file.path(path2, datname),row.names=T)
  }
}
dl_csv2 <- function(data_l, name_l, path){
  
  dir.create(file.path(path))
  
  path2 <- paste(path,'', sep = '\\')
  
  
  for(i in 1:length(name_l)){
    
    dat <- data_l[[i]]
    
    datname <- paste(name_l[[i]], '.csv', sep = '')
    
    
    write.csv(dat, file.path(path2, datname),row.names=T)
  }
}
corr_order <- function(in_df){
  
  
  target_holder <- in_df$y
  
  in_df <- in_df %>% dplyr::select(-y)
  
  trans_df <- data.frame(t(in_df))
  
  phylum_groups <- c("Proteobacteria", "Actinobacteriota", "Firmicutes", "other")
  
  holder <- list()
  
  for(i in phylum_groups){
    
    
    df_lub <- trans_df %>% filter(Phylum==i) %>% dplyr::select(-Phylum) 
    
    cor_trans_df <- data.frame(t(df_lub))
    
    cor_trans_df <- data.frame(lapply(cor_trans_df,as.numeric))
    
    cor_trans_df[cor_trans_df == 0] <- NA
    
    
    corr_mat <- compositions::cor(cor_trans_df,method="spearman",use = "pairwise.complete.obs")
    
    corr_sum<-rowSums(abs(corr_mat),  na.rm = T)
    corr_sum_df<-as.data.frame(as.numeric(corr_sum))
    corr_sum_df$ID<-1:nrow(corr_sum_df) 
    corr_sum_df$Taxa <- rownames(corr_mat)
    
    
    
    corr_sum_sorted <- corr_sum_df[order(corr_sum_df$`as.numeric(corr_sum)`, decreasing = T),]
    #order(corr_sum_df$`as.numeric(corr_sum)`),
    holder[[length(holder)+1]] <- corr_sum_sorted$Taxa 
    
    
  }
  holder <- unlist(holder)
  holder[[length(holder)+1]] <- 'y'
  return(holder)
  
  
}
corr_order2 <- function(in_df){
  
  
  target_holder <- in_df$y
  
  in_df <- in_df %>% dplyr::select(-y)
  
  trans_df <- data.frame(t(in_df))
  
  phylum_groups <- c("Proteobacteria", "Actinobacteriota", "Firmicutes", "other")
  
  holder <- list()
  print(trans_df)
  
  for(i in phylum_groups){

    df_lub <- trans_df %>% filter(Phylum==i) %>% dplyr::select(-Phylum) 
    
    cor_trans_df <- data.frame(t(df_lub))
    
    cor_trans_df <- data.frame(lapply(cor_trans_df,as.numeric))

    corr_mat <- cor(cor_trans_df,method="spearman",use = "pairwise.complete.obs")
    
    corr_sum<-rowSums(abs(corr_mat),  na.rm = T)
    if (length(corr_sum)==0){
    }
    else {
      corr_sum_df<-as.data.frame(as.numeric(corr_sum))
      corr_sum_df$ID<-1:nrow(corr_sum_df) 
      corr_sum_df$Taxa <- rownames(corr_mat)
      
      
      
      corr_sum_sorted <- corr_sum_df[order(corr_sum_df$`as.numeric(corr_sum)`,decreasing = T),]
      #order(corr_sum_df$`as.numeric(corr_sum)`),
      holder[[length(holder)+1]] <- corr_sum_sorted$Taxa 
      
    }
  }
  holder <- unlist(holder)
  holder[[length(holder)+1]] <- 'y'
  return(holder)
  
  
}
outer_approach <- function(ra_list, name_l, t_name){
  feat_lengths <- list()
  out_list <- list()
  for (df in ra_list) {
    
    feat_lengths[[length(feat_lengths)+1]] <- colnames(df)
  }
  unique_feats <- unlist(feat_lengths) %>% unique()
  
  for (ind in seq_len(length(ra_list))){
    
    working_df <- ra_list[[ind]]
    
    missing_taxa <- setdiff(unique_feats,colnames(working_df))
    
    working_df[missing_taxa] <- 0
    
    trans_df <- data.frame(t(working_df))
    
    
    Phy <- vapply(strsplit(rownames(trans_df),"_"), `[`, 2, FUN.VALUE=character(1))
    #print(Phy)
    trans_df$Phylum <- Phy
    
    shouldBecomeOther<-!(trans_df$Phylum %in% c("Proteobacteria", "Actinobacteriota", "Firmicutes",NA))
    
    trans_df$Phylum[shouldBecomeOther]<- "other"
    
    out_list[[length(out_list)+1]] <- data.frame(t(trans_df))
    
  }
  train_ind <- which(name_l == t_name)
  
  train_ord <- corr_order(out_list[[train_ind]])
  
  for (j in 1:length(out_list)){
    out_list[[j]] <- out_list[[j]][train_ord]
    print(any(colnames(out_list[[j]])==train_ord)==F)
  }
  
  
  return(out_list)
}
outer_approach2 <- function(ra_list, name_l, t_name){
  feat_lengths <- list()
  out_list <- list()
  for (df in ra_list) {
    
    feat_lengths[[length(feat_lengths)+1]] <- colnames(df)
  }
  unique_feats <- unlist(feat_lengths) %>% unique()
  
  for (ind in seq_len(length(ra_list))){
    
    working_df <- ra_list[[ind]]
    
    missing_taxa <- setdiff(unique_feats,colnames(working_df))
    
    working_df[missing_taxa] <- 0
    
    trans_df <- data.frame(t(working_df))
    
    
    Phy <- vapply(strsplit(rownames(trans_df),"_"), `[`, 2, FUN.VALUE=character(1))
    #print(Phy)
    trans_df$Phylum <- Phy
    
    shouldBecomeOther<-!(trans_df$Phylum %in% c("Proteobacteria", "Actinobacteriota", "Firmicutes",NA))
    
    trans_df$Phylum[shouldBecomeOther]<- "other"
    
    out_list[[length(out_list)+1]] <- data.frame(t(trans_df))
    
  }
  
  
  for (j in 1:length(out_list)){
    train_ord <- corr_order(out_list[[j]])
    out_list[[j]] <- out_list[[j]][train_ord]
    
    
    print(any(colnames(out_list[[j]])==train_ord)==F)
  }
  
  
  return(out_list)
}
remove_duplicates <- function(in_mat){
  df <- data.frame(in_mat)
  rnames <- row.names(df)
  names_no_dot <- sub("\\..*", "", names(df))
  
  # For each unique name, get the rowSums of all columns with that name
  df2 <- sapply(unique(names_no_dot), function(x) {
    subset_df <- df[, names_no_dot == x, drop = FALSE]
    if (ncol(subset_df) == 1) {
      return(subset_df[,1])
    } else {
      return(rowSums(subset_df))
    }
  })
  
  # Assign the correct column names to the new data frame
  colnames(df2) <- unique(names_no_dot)
  
  # If you need a data.frame instead of a matrix:
  df2 <- data.frame(df2)
  row.names(df2) <- rnames
  return(df2)
}
ilr.dat <- function(otu_data){
  
  X <- otu_data@otu_table@.Data
  
  
  X.data <- compositions::ilr(X)
  X.data <- remove_duplicates(X.data)
  
  y <- otu_data@sam_data$PD
  
  y<-factor(y , levels = c('PD','HC'), ordered = T)
  y<- factor(ifelse(y=='PD',1,0), levels = c(0,1), ordered=T)
  
  df.1<-na.omit(cbind( X.data,y = y)); df.check <- df.1 %>% dplyr::select(where(is.numeric))
  df.check$tot <- rowSums(df.check)
  df.check <- df.check[order(df.check$tot, decreasing = T),]
  df.check <- df.check %>% dplyr::filter(df.check$tot>0)
  
  df.1 <- df.1[(rownames(df.1) %in% rownames(df.check)),]
  
  if (any(grep("t-0", rownames(df.1))) == TRUE){
    df.1 = df.1[grep("t-0", rownames(df.1)),]
  }
  
  
  return(df.1)
}
clr.dat <- function(otu_data){
  
  X <- otu_data@otu_table@.Data
  # print(data.frame(X))
  #X_psu_log <- log(X + 0.5,exp(1))
  
  X.data <- compositions::clr(X)
  X.data <- remove_duplicates(X.data)
  # print(data.frame(X.data))
  
  y <- otu_data@sam_data$PD
  
  y<-factor(y , levels = c('PD','HC'), ordered = T)
  y<- factor(ifelse(y=='PD',1,0), levels = c(0,1), ordered=T)
  
  df.1<-na.omit(cbind( X.data,y = y)); df.check <- data.frame(X) %>% dplyr::select(where(is.numeric))
  df.check$tot <- rowSums(df.check)
  df.check <- df.check[order(df.check$tot, decreasing = T),]
  df.check <- df.check %>% dplyr::filter(df.check$tot>0)
  print(df.check)
  
  df.1 <- df.1[(rownames(df.1) %in% rownames(df.check)),]
  
  # if (any(grep("t-0", rownames(df.1))) == TRUE){
  #   #print('Lubomski')
  #   df.1 = df.1[grep("t-0", rownames(df.1)),]
  # }
  # else {
  #   #print('NotLub')
  # }
  
  
  return(df.1)
  
  
}
asin.dat <- function(otu_data){
  
  X <- otu_data@otu_table@.Data
  
  #X_psu_log <- log(X + 0.5,exp(1))
  X[X < 0] <- 0
  X[X > 1] <- 1
  
  X.data <- asin(sqrt(X))
  X.data <- remove_duplicates(X.data)

  y <- otu_data@sam_data$PD
  
  y<-factor(y , levels = c('PD','HC'), ordered = T)
  y<- factor(ifelse(y=='PD',1,0), levels = c(0,1), ordered=T)
  
  df.1<-na.omit(cbind( X.data,y = y)); df.check <- df.1 %>% dplyr::select(where(is.numeric))
  df.check$tot <- rowSums(df.check)
  df.check <- df.check[order(df.check$tot, decreasing = T),]
  df.check <- df.check %>% dplyr::filter(df.check$tot>0)
  
  df.1 <- df.1[(rownames(df.1) %in% rownames(df.check)),]
  
  # if (any(grep("t-0", rownames(df.1))) == TRUE){
  #   #print('Lubomski')
  #   df.1 = df.1[grep("t-0", rownames(df.1)),]
  # }
  # else {
  #   #print('NotLub')
  # }
  
  
  return(df.1)
}
none.dat <- function(otu_data){
  
  
  X <- otu_data@otu_table@.Data
  
  #X_psu_log <- log(X + 0.5,exp(1))
  
  X.data <- X
  X.data <- remove_duplicates(X.data)
  
  y <- otu_data@sam_data$PD
  
  y<-factor(y , levels = c('PD','HC'), ordered = T)
  y<- factor(ifelse(y=='PD',1,0), levels = c(0,1), ordered=T)
  
  df.1<-na.omit(cbind( X.data,y = y)); df.check <- df.1 %>% dplyr::select(where(is.numeric))
  df.check$tot <- rowSums(df.check)
  df.check <- df.check[order(df.check$tot, decreasing = T),]
  df.check <- df.check %>% dplyr::filter(df.check$tot>0)
  
  df.1 <- df.1[(rownames(df.1) %in% rownames(df.check)),]
  
  # if (any(grep("t-0", rownames(df.1))) == TRUE){
  #   #print('Lubomski')
  #   df.1 = df.1[grep("t-0", rownames(df.1)),]
  # }
  # else {
  #   #print('NotLub')
  # }
  
  
  return(df.1)
}
clean_taxa_names <- function(taxa_vec){
  holder <- c()
  #first 
  for (i in 1:length(taxa_vec)){
    holder[i] <- substr(taxa_vec[i], 1, ((nchar(taxa_vec[i]))/2))
  }
  return(holder)
}
abs_diff_taxa <- function(coef_df){
  
  #first taxa in pair
  
  abs_taxa_1 <- vapply(strsplit(coef_df$Row.names,"--"), `[`, 1, FUN.VALUE=character(1))
  
  #second taxa in pair
  
  abs_taxa_2 <- vapply(strsplit(coef_df$Row.names,"--"), `[`, 2, FUN.VALUE=character(1))
  
  #remove repeated val in first taxa from pairwise diff function
  
  clean_1 <- clean_taxa_names(abs_taxa_1)
  #clean_2 <- clean_taxa_names(abs_taxa_2)
  taxa_n <- table(c(clean_1, abs_taxa_2)) ; return_obj <- as.data.frame(taxa_n)
  return(return_obj)
}
cleaner <- function(unclean_l){
  t1 <- vapply(strsplit(unclean_l,"--"), `[`, 1, FUN.VALUE=character(1))
  t2 <- vapply(strsplit(unclean_l,"--"), `[`, 2, FUN.VALUE=character(1))
  clean_1 <- clean_taxa_names(t1)
  t_n <- paste(clean_1, t2, sep = '--')
  return(t_n)
}
order_outer_list <- function(input_list, u_feats){
  outer_list <- list()
  rank_list <- list()
  
  for (j in 1:length(input_list)){
    w_df <-  input_list[[j]]
    outer_df3 <- outer_approach3(w_df,u_feats)
    outer_df3 <- outer_df3[sample(nrow(outer_df3), replace = F),]
    outer_df3 <- outer_df3[,sort(colnames(outer_df3))]
    trans_df <- data.frame(t(outer_df3))
    Phy <- vapply(strsplit(rownames(trans_df),"_"), `[`, 2, FUN.VALUE=character(1))
    trans_df$Phylum <- Phy
    shouldBecomeOther<-!(trans_df$Phylum %in% c("Proteobacteria", "Actinobacteriota", "Firmicutes",NA))
    trans_df$Phylum[shouldBecomeOther]<- "other"
    trans_df <- data.frame(t(trans_df))
    outer_list[[length(outer_list)+1]] <- trans_df
    
  }
  new_order <-corr_order2(outer_list[[1]])
  for (k in 1:length(outer_list)){
    
  rank_list[[k]] <-  subset(outer_list[[k]] , select = new_order)
  }
  return(rank_list)
}
corr_order_rank2 <- function(put_in_df){
  
  df_copy <- put_in_df
  
  #print(dim(df_copy))
  target_holder <- df_copy$y
  
  dat_df <- df_copy %>% dplyr::select(-y)
  
  tr_df <- data.frame(t(dat_df))
  
  phylum_groups <- c("Proteobacteria", "Actinobacteriota", "Firmicutes", "other")
  
  holder <- list()
  rank_holder <- list()
  
  
  
  for(i in phylum_groups){
    
    
    df_lub <- tr_df %>% filter(Phylum==i) %>% dplyr::select(-Phylum) 
    
    
    
    cor_trans_df <- data.frame(t(df_lub))
    
    cor_trans_df <- data.frame(lapply(cor_trans_df,as.numeric))
    
    #cor_trans_df[cor_trans_df == 0] <- NA
    
    corr_mat <- compositions::cor(cor_trans_df,method="spearman",use = "pairwise.complete.obs")
    #print(dim(corr_mat))
    
    corr_sum<-rowSums(abs(corr_mat),  na.rm = T)
    corr_sum_df<-as.data.frame(as.numeric(corr_sum))
    corr_sum_df$ID<-1:nrow(corr_sum_df) 
    corr_sum_df$Taxa <- rownames(corr_mat)
    
    
    corr_sum_sorted <- corr_sum_df[order(corr_sum_df$`as.numeric(corr_sum)`, decreasing = T),]
    corr_sum_sorted$rank <- 1:nrow(corr_sum_sorted)
    
    split_taxa <- vapply(strsplit(corr_sum_sorted$Taxa,"_"), `[`, 1, FUN.VALUE=character(1))
    split_taxa2 <- vapply(strsplit(corr_sum_sorted$Taxa,"_"), `[`, 2, FUN.VALUE=character(1))
    
    split_taxa2 <- lapply(split_taxa2, function(x) replace(x, !(x %in% c("Proteobacteria", "Actinobacteriota", "Firmicutes")), 'Other'))
    
    
    
    test_names <- paste(split_taxa,split_taxa2, corr_sum_sorted$rank, sep='_')
    
    #print(test_names)
    
    corr_sum_sorted$Taxa_rank <- test_names
    
    holder[[length(holder)+1]] <- corr_sum_sorted$Taxa 
    rank_holder[[length(rank_holder)+1]] <- corr_sum_sorted$Taxa_rank 
    
    
  }
  holder <- unlist(holder)
  holder[[length(holder)+1]] <- 'y'
  
  rank_holder <- unlist(rank_holder)
  rank_holder[[length(rank_holder)+1]] <- 'y'
  #print(df_copy[,order(holder)])
  
  df_ord <- df_copy[holder]
  #print(df_ord)
  colnames(df_ord) <- rank_holder
  return(df_ord)
}
outer_approach3 <- function(in_df2, outer_feats){
  
  
  working_df <- in_df2
  
  missing_taxa <- setdiff(outer_feats,colnames(working_df))
  # print(missing_taxa)
  
  working_df[missing_taxa] <- 0
  
  return(working_df)
}