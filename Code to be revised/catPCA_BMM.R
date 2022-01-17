library(Gifi)
library(flexmix)
library(data.table)
library(pracma)
library(forecastHybrid)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cluster)
library(Hmisc)
library(Rmisc)
library(cultevo)
#binominale kofidenzintervalle , binominalverteilung
setwd("~/Dropbox/_ISYE6740-Project_Team-36/_Data/_Analysis/R")

############ Load Data ############
# Import the data and look at the first six rows
df.namibia_bi = read.csv(file = 'X_namibia_train.csv')

vars = c('Fruits.and.vegs', 'Soft.drinks', 'Fast.food', 'Brushing.teeth', 
         'Wash.hands.eating', 'Wash.hands.toilet', 'Sitting', 'Amphetamines', 'Marijuana', 
         'Drunk', 'Smokeless.tobacco', 'Smoking', 'Fighting', 'Attacking')
df.namibia_bi_red = df.namibia_bi[vars]

scaled.df.namibia_bi_red = scale(df.namibia_bi_red)


#### catPCA ####
cat_namibia = princals(scaled.df.namibia_bi_red, ordinal=FALSE)
summary(cat_namibia)
catPCA_loadings = cat_namibia$loadings
write.csv(catPCA_loadings,"catPCA_loadings.csv", row.names = FALSE)

plot(cat_namibia, "biplot", col.loadings = "black", col.scores = "orange", main = "Biplot Plot GSHS Namibia Data") 
plot(cat_namibia, "loadplot", main = "Loadings Plot GSHS Namibia Data")
plot(cat_namibia, "screeplot", main="")

#after screeplot, catPCA with 5 dimensions
cat_namibia_5comp = princals(scaled.df.namibia_bi_red, ordinal=FALSE, ndim=5)
summary(cat_namibia_5comp)

catPCA_loadings5 = cat_namibia_5comp$loadings
catPCA_loadings5
#write.csv(summary(cat_namibia), "components.csv", row.names = FALSE)
write.csv(catPCA_loadings5,"catPCA_loadings_5dims.csv")
df_csv = read.csv(file = "catPCA_loadings_5dims.csv", header=FALSE)
t_df_csv = transpose(df_csv)
t_df_csv = subset(t_df_csv, select = -c(1))
write.table(t_df_csv, sep=",", file ="t_catPCA_loadings_5dims.csv", col.names = FALSE, row.names = FALSE)

#### BMM ####
namibia_matrix = as.matrix(df.namibia_bi_red)

cont = list(tolerance = 1e-15, iter.max = 1000)

bmm = stepFlexmix(namibia_matrix ~ 1, data = df.namibia_bi_red, 
                  k=1:10, model=FLXMCmvbinary(), control=cont, nrep=3)

bmm_best_fit = getModel(bmm, "BIC")
prior(bmm_best_fit)

param_bmm = data.frame(parameters(bmm_best_fit))
param_bmm = param_bmm %>% mutate(Type=colnames(namibia_matrix))
head(param_bmm)

#cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

param_bmm %>%
  gather(Components, Lambda, -Type) %>%
  ggplot(aes(x=Type, y=Lambda, fill=Type))+
#  scale_fill_manual(values=cbbPalette)+
  geom_bar(stat="identity")+
  facet_wrap(~Components)+
  theme(axis.text.x=element_text(angle=90,hjust=1),legend.position="none")

hammingdists(namibia_c)
namibia_c = df.namibia_bi_red %>% mutate(CLUSTER=factor(clusters(bmm_best_fit)))
write.table(namibia_c, sep=",", file ="bmm_cluster_assignment.csv", row.names = FALSE)

library(fpc)
pamk.best = pamk(df.namibia_bi_red)
cat("number of clusters estimated by optimum average silhouette width:", pamk.best$nc, "\n")
plot(pam(df.namibia_bi_red, pamk.best$nc))

summary(bmm)
prior(bmm)
param_comp1 = parameters(bmm, component=1)
param_comp2 = parameters(bmm, component=2)
dim(param_comp1)
head(param_comp1)
plot(bmm)
plot(gmm)

#### GMM ####
df.namibia_pc = read.csv(file = 'namibia_train_pc.csv')
df.namibia_pc$X = NULL
namibia_matrix_pc = as.matrix(df.namibia_pc)
gmm = stepFlexmix(namibia_matrix_pc ~ 1, data = df.namibia_bi_red, 
                  k=1:15, model=FLXMCmvnorm(), control=cont, nrep=3)

gmm_best_fit = getModel(gmm, "BIC")
prior(gmm_best_fit)

param_gmm = data.frame(parameters(gmm_best_fit))
param_gmm = param_gmm %>% mutate(Type=colnames(df.namibia_pc))

param_gmm %>%
  gather(Components, Lambda, -Type) %>%
  ggplot(aes(x=Type, y=Lambda, fill=Type))+
  #  scale_fill_manual(values=cbbPalette)+
  geom_bar(stat="identity")+
  facet_wrap(~Components)+
  theme(axis.text.x=element_text(angle=90,hjust=1),legend.position="none")

namibia_gmm = df.namibia_pc %>% mutate(CLUSTER=factor(clusters(gmm_best_fit)))
write.table(namibia_gmm, sep=",", file ="gmm_cluster_assignment.csv", row.names = FALSE)

