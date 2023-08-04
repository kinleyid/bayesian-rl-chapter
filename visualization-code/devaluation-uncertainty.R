
rm(list = ls())
library(ggplot2)

proj_path <- getwd()

data <- read.csv(file.path(proj_path, 'data', 'devaluation-uncertainty.csv'))
data$agent_type <- factor(data$agent_type, levels = c('MB', 'MF'), labels = c('Model-based', 'Model-free'))

data$train_n <- data$train_n + 1
img <- ggplot(data = data,
              mapping = aes(x = train_n, y = uncertainty, linetype = agent_type, fill = agent_type)) +
  stat_summary(geom = 'ribbon', fun.data = mean_se, alpha = 0.8, colour = NA) +
  stat_summary(geom = 'line', fun = mean) +
  scale_x_log10() +
  # coord_trans(y = 'log10') +
  labs(x = 'Training iteration', y = 'Uncertainty', linetype = 'Agent type', fill = 'Agent type') +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  theme_classic() +
  theme(text = element_text(size = 20))
plot(img)

ggsave(plot = img,
       filename = file.path(proj_path, 'figs', 'devaluation-uncertainty.png'),
       height = 15,
       width = 20,
       units = 'cm')
