
rm(list = ls())
library(ggplot2)

proj_path <- getwd()
proj_path <- 'C:/Users/isaac/Projects/bayesian-rl-chapter/github'
data <- c()
for (tag in c('default',
              'adjusted-params',
              'revaluation')) {
  sub_data <- read.csv(file.path(proj_path, 'data', sprintf('impulsivity-%s.csv', tag)))
  sub_data$tag <- tag
  data <- rbind(data, sub_data)
}
data$tag <- factor(data$tag,
                   levels = c('adjusted-params', 'default', 'revaluation'),
                   labels = c('Altered parameters', 'Default parameters', 'Revaluation'))
# data$tag <- factor(data$tag,
#                    levels = c('default', 'adjusted-params', 'revaluation'),
#                    labels = c('Default parameters', 'Altered parameters', 'Revaluation'))

data$agent_type <- factor(data$agent_type, levels = c('MB', 'MF'), labels = c('Model-based', 'Model-free'))
img <- ggplot(data = data, aes(x = n_wait, linetype = agent_type, y = p_delayed, fill = agent_type)) +
  stat_summary(geom = 'ribbon', fun.data = mean_se, alpha = 0.8, colour = NA) +
  stat_summary(geom = 'line', fun = mean) +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  theme_classic() +
  theme(text = element_text(size = 20)) +
  scale_x_continuous(breaks = 0:max(data$n_wait)) +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  facet_wrap(~ tag, ncol = 1) +
  # scale_y_log10(breaks = 10^seq(0, -10, by = -2), labels = function(x) {TeX(sprintf('$10^{%s}$', log10(x)))}) +
  labs(x = 'Delay', y = 'Probability of waiting for delayed reward', fill = 'Agent type', linetype = 'Agent type')
plot(img)
  
ggsave(plot = img,
       filename = file.path(proj_path, 'figs', 'impulsivity-2.png'),
       height = 20,
       width = 20,
       units = 'cm')
