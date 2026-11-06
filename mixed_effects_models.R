library(lme4)
library(lmerTest)
library(effects)
library(dplyr)
library(tidyr)
library(readr)

# ==============================================================================
# Read the data
# ==============================================================================
test_perf <- read.csv("D:/PycharmProjects/Hypernetwork/Results/Cognitive_Model_Evaluation/cct_dd_model_by_fold.csv")
test_perf$model_family <- factor(test_perf$model_family, levels = c('HyperNN', 'BasicNN', 'Cognitive'))
test_perf$fold <- factor(test_perf$fold, levels = c('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

dd <- test_perf %>%
  filter(task == 'dd')

dd$model <- factor(dd$model, levels = c('layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0', 
                                        'dd_exponential', 
                                        'dd_hyperbolic', 
                                        'dd_hyperboloid', 
                                        'layers_1_dims_64'))

cct <- test_perf %>%
  filter(task == 'cct')
cct$model <- factor(cct$model, levels = c('layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0', 
                                        'cct_pt', 
                                        'cct_pt_prob', 
                                        'cct_pt_loss_shape', 
                                        'layers_1_dims_64'))

stopsignal <- test_perf %>%
  filter(task == 'stopsignal')
stopsignal$model <- factor(stopsignal$model, levels = c('layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0', 
                                          'ss_logistic', 
                                          'ss_hr_exgau', 
                                          'ss_rdex', 
                                          'layers_1_dims_64'))

motor <- test_perf %>%
  filter(task == 'motor')
motor$model <- factor(motor$model, levels = c('layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0', 
                                                        'motor_logistic', 
                                                        'motor_hr_exgau', 
                                                        'motor_rdex', 
                                                        'layers_1_dims_64'))

best_cog <- test_perf %>%
  filter(model%in%(c('cct_pt_loss_shape', 'dd_hyperboloid', 'motor_rdex', 'ss_rdex', 
                     'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0',
                     'layers_1_dims_64')))

testnll <- lmer(test_nll ~ model_family + task + (1 | fold), data=test_perf)
summary(testnll)
plot(allEffects(testnll))

testnll_best <- lmer(test_nll ~ model_family + task + (1 | fold), data=best_cog)
summary(testnll_best)
plot(allEffects(testnll_best))

testnll <- lmer(test_nll ~ model + (1 | fold), data=motor)
summary(testnll)
plot(allEffects(testnll))


