library(lme4)
library(lmerTest)
library(effects)
library(dplyr)
library(tidyr)
library(readr)

# ==============================================================================
# Read the data
# ==============================================================================
test_perf <- read.csv("D:/PycharmProjects/Cog_NN/Results/Cognitive_Model_Evaluation/cct_dd_model_by_fold.csv")
test_perf$model_family <- factor(test_perf$model_family, levels = c('HyperNN', 'BasicNN', 'Cognitive'))
                                        
dd <- test_perf %>%
  filter(task == 'dd')

dd$model <- factor(dd$model, levels = c('layers_1_dims_64_rank_full_emb_4_nonlinear_nodes_8_reg_0.0', 
                                        'dd_exponential', 
                                        'dd_hyperbolic', 
                                        'dd_hyperboloid', 
                                        'layers_1_dims_64'))
dd$fold <- factor(dd$fold, levels = c('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
                                        

cct <- test_perf %>%
  filter(task == 'cct')
cct$model <- factor(cct$model, levels = c('layers_1_dims_64_rank_full_emb_4_nonlinear_nodes_8_reg_0.0', 
                                        'cct_pt', 
                                        'cct_pt_prob', 
                                        'cct_pt_loss_shape', 
                                        'layers_1_dims_64'))

testnll <- lmer(test_nll ~ model_family + task + (1 | fold), data=test_perf)
summary(testnll)
plot(allEffects(testnll))

testnll <- lmer(test_nll ~ model + (1 | fold), data=dd)
summary(testnll)
plot(allEffects(testnll))

testacc <- lm(test_accuracy ~ model + (1 | fold), data=dd)
summary(testacc)
plot(allEffects(testacc))
