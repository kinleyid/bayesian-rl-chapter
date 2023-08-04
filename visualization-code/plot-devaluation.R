
rm(list = ls())
library(ggplot2)

data <- read.csv('C:/Users/isaac/Projects/book-chapter/devaluation-results.csv')
data$agent_type <- factor(data$agent_type, levels = c('MB', 'MF'), labels = c('Model-based', 'Model-free'))
data$rel_rate <- data$prob_Lever_A / data$prob_Lever_B

img <- ggplot(data = data, aes(x = agent_type, y = rel_rate, fill = agent_type)) +
  stat_summary(geom = 'bar', fun = median) +
  # scale_y_log10(expand = c(0.01, NA)) +
  scale_fill_manual(values = c('darkgray', 'lightgray')) +
  labs(x = 'Agent type', y = 'Relative response rate for devalued option') +
  theme_classic()
plot(img)

ggsave(plot = img,
       filename = file.path('C:/Users/isaac/Projects/book-chapter/figs/devaluation.png'),
       height = 15,
       width = 20,
       units = 'cm')
