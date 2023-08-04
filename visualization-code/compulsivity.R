
rm(list = ls())
library(ggplot2)

proj_path <- getwd()

data <- read.csv(file.path(proj_path, 'data', 'compulsivity.csv'))
data$addictive <- factor(data$addictive, levels = c('False', 'True'), labels = c('Control', 'Addictive'))
data$agent_type <- factor(data$agent_type, levels = c('MF', 'MB'), labels = c('Model-free', 'Model-based'))
data$train_n <- data$train_n + 1

img <- ggplot(data = data,
              mapping = aes(x = train_n, y = Q_lever, linetype = addictive, fill = addictive)) +
  stat_summary(geom = 'ribbon', fun.data = mean_se, alpha = 0.8, colour = NA) +
  stat_summary(geom = 'line', fun = mean) +
  geom_vline(xintercept = 20.5) +
  annotate('text', x = 26, y = 0.3, label = sprintf('Punishment\nintroduced'), hjust = 0, size = 5) +
  annotate("segment", x = 25, xend = 20.5, y = 0.3, yend = 0.3) +
  facet_wrap(~ agent_type) +
  labs(x = 'Training iteration', y = 'Value of level pull', fill = 'Reward type', linetype = 'Reward type') +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  theme_classic() +
  theme(text = element_text(size = 20))
plot(img)

ggsave(plot = img,
       filename = file.path(proj_path, 'figs', 'compulsivity.png'),
       height = 15,
       width = 20,
       units = 'cm')
