library(lme4)
library(lmerTest)
library(effects)
library(dplyr)
library(tidyr)
library(readr)

# ==============================================================================
# Read the data
# ==============================================================================
test_perf <- read.csv("D:/PycharmProjects/Hypernetwork/Results/Cognitive_Model_Evaluation/test_performance_by_fold.csv")
test_perf$model_family <- factor(test_perf$model_family, levels = c('HyperNN', 'BasicNN', 'cognitive_individual', 'cognitive_group'))
test_perf$fit_level <- factor(test_perf$fit_level, levels = c('individual_fit', 'group_fit'))
test_perf$random_split <- factor(test_perf$random_split, levels = c('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
test_perf$structure <- ifelse(
  test_perf$model_family %in% c("HyperNN", "BasicNN"),
  "NN",
  "Cognitive"
)


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
                     'layers_1_dims_8')))

testnll <- lmer(test_nll ~ model_family + task + (1 | random_split), data=test_perf)
summary(testnll)
plot(allEffects(testnll))

testnll_best <- lmer(test_nll ~ model_family + task + (1 | random_split), data=best_cog)
summary(testnll_best)
plot(allEffects(testnll_best))

fit_level_effect <- lmer(test_nll ~ fit_level * structure + task + (1 | random_split), data=test_perf)
summary(fit_level_effect)
plot(allEffects(fit_level_effect))


