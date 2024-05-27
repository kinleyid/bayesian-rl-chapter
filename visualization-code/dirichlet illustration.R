
library(ggplot2)

# Figure of decreasing uncertainty with increasing numbers of observations
set.seed(123)
img <- ggplot()

x1 <- 1
x2 <- 1
true_p <- 0.75
n_iter <- 4

counts <- c(10, 20, 40, 80)

for (iter in 0:n_iter) {
  if (iter > 0) {
    # Update estimated mean
    obs <- runif(10*2**(iter-1)) < true_p
    x1 <- x1 + sum(obs)
    x2 <- x2 + sum(!obs)
  }
  # Add layer
  img <- img + 
    stat_function(
      data = data.frame(x = c(0, 1), iter = iter),
      mapping = aes(x = x, fill = iter),
      fun = dbeta, args = list(shape1 = x1, shape2 = x2),
      geom = "area",
      show.legend = F,
      alpha = 0.6,
      n = 10000)
}

img_2_plot <- img + 
  scale_fill_gradient(high = rgb(0.2, 0.2, 0.2), low = rgb(0.8, 0.8, 0.8)) +
  # facet_wrap(vars(s2), labeller = as_labeller(c('10' = 'High sensory noise', '2' = 'Low sensory noise'))) +
  theme_classic() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  scale_x_continuous(breaks = c(0, true_p, 1), labels = c(0, 'Truth', 1)) +
  labs(fill = 'N. obs.', x = 'Transition probability', y = 'Probability density') +
  theme(text = element_text(size = 20))
plot(img_2_plot)

ggsave(plot = img_2_plot,
       filename = file.path('C:/Users/isaac/Projects/bayesian-rl-chapter/github/figs/dirichlet-convergence.png'),
       height = 15,
       width = 20,
       units = 'cm')
