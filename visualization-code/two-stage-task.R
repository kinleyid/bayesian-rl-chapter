
rm(list = ls())
library(ggplot2)

proj_path <- getwd()

data <- read.csv(file.path(proj_path, 'data', 'two-stage-task.csv'))

data <- data[complete.cases(data), ]
data$last_r <- factor(data$last_r, levels = c(0, 1), labels = c('Unrewarded', 'Rewarded'))
data$last_trans <- factor(data$last_trans, levels = c('common', 'rare'), labels = c('Common', 'Rare'))
data$agent_type <- factor(data$agent_type, levels = c('MB', 'MF'), labels = c('Model-based', 'Model-free'))
data$cond <- interaction(data$last_trans, data$last_r)

x <- aggregate(stay ~ agent_type, data = data, FUN = mean)
x$rel_stay <- x$stay
x$stay <- NULL
data <- merge(data, x)
data$stay <- data$stay - data$rel_stay + 0.5

img <- ggplot(
  data = aggregate(stay ~ last_trans + last_r + rep_n + agent_type, data = data, FUN = mean),
  mapping = aes(x = last_trans, fill = last_r, y = stay)) +
  stat_summary(geom = 'bar', fun = mean, size = 2, position = 'dodge') +
  stat_summary(geom = 'errorbar', fun.data = mean_se, width = 0.2, position = position_dodge(width = 0.9)) +
  facet_wrap(~ agent_type) +
  labs(x = 'Last transition', fill = 'Last outcome', y = 'Relative repetition probability') +
  coord_cartesian(ylim = c(0.425, 0.575)) +
  scale_y_continuous(labels = function(x) {x - 0.5}) +
  scale_fill_grey() +
  theme_classic() +
  theme(text = element_text(size = 20))
plot(img)

ggsave(plot = img,
       filename = file.path(proj_path, 'figs', 'two-stage-task.png'),
       height = 15,
       width = 20,
       units = 'cm')
