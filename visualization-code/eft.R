
rm(list = ls())
library(ggplot2)
library(cowplot)

proj_path <- getwd()

data <- read.csv('C:/Users/isaac/Projects/bayesian-rl-chapter/dev/eft.csv')
data$variant <- factor(data$variant, levels = c('eft', 'default'), labels = c('EFT', 'Default'))

img1 <- ggplot(data = data, aes(x = n_wait, y = uncertainty, fill = variant, linetype = variant)) +
  stat_summary(geom = 'ribbon', fun.data = mean_se, alpha = 0.8, colour = NA) +
  stat_summary(geom = 'line', fun = mean) +
  theme_classic() +
  theme(text = element_text(size = 20)) +
  scale_x_continuous(breaks = 0:max(data$n_wait)) +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  labs(x = 'Delay', y = sprintf('\nUncertainty'),
       fill = 'Condition', linetype = 'Condition')

img2 <- ggplot(data = data, aes(x = n_wait, y = p_delayed, fill = variant, linetype = variant)) +
  stat_summary(geom = 'ribbon', fun.data = mean_se, alpha = 0.8, colour = NA) +
  stat_summary(geom = 'line', fun = mean) +
  theme_classic() +
  theme(text = element_text(size = 20)) +
  scale_x_continuous(breaks = 0:max(data$n_wait)) +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  # scale_y_log10(breaks = 10^seq(0, -10, by = -2), labels = function(x) {TeX(sprintf('$10^{%s}$', log10(x)))}) +
  labs(x = 'Delay', y = sprintf('Probability of waiting\nfor delayed reward'),
       fill = 'Condition', linetype = 'Condition')

img <- plot_grid(img2, img1, ncol = 1, labels = c('(a)', '(b)'), label_fontfamily = "serif",
                 hjust = -0.5)

ggsave(plot = img,
       filename = file.path(proj_path, 'figs', 'eft.png'),
       height = 20,
       width = 20,
       units = 'cm')
