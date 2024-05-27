
rm(list = ls())
library(ggplot2)

proj_path <- getwd()

data <- read.csv('C:/Users/isaac/Projects/bayesian-rl-chapter/github/data/devaluation-choices.csv')
data$agent_type <- factor(data$agent_type, levels = c('MB', 'MF'), labels = c('Model-based', 'Model-free'))
data$phase <- factor(data$phase, levels = c('pre', 'post'), labels = c('Non-devalued', 'Devalued'))

img <- ggplot(data = data, aes(x = phase, y = lever_rate, fill = phase)) +
  geom_violin(show.legend = F, alpha = 0.8, colour = NA) +
  geom_boxplot(fill = 'white', colour = 'black', width = 0.2, outlier.shape = NA) +
  geom_line(aes(group = rep_n), alpha = 0.05) +
  geom_boxplot(fill = 'white', colour = 'black', width = 0.2, outlier.shape = NA, alpha = 0.6) +
  facet_wrap(~ agent_type) +
  # coord_cartesian(ylim = c(0.35, 1)) +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  labs(x = 'Reward type', y = 'Response rate') +
  theme_classic() +
  theme(
    text = element_text(size = 20),
    panel.spacing = unit(1, "cm"),
    panel.border = element_rect(fill = NA)
  )
plot(img)

ggsave(plot = img,
       filename = 'C:/Users/isaac/Projects/bayesian-rl-chapter/github/figs/devaluation-choices.png',
       height = 15,
       width = 20,
       units = 'cm')
