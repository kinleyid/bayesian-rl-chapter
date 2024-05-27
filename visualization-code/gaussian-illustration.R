
library(ggplot2)

# Figure of decreasing uncertainty with increasing numbers of observations

img <- ggplot()
for (sensory_s2 in c(10, 2)) {
  est_mu_initial <- 0
  est_mu <- est_mu_initial
  est_s2 <- 5
  true_mu <- 5
  n_iter <- 4
  
  for (iter in 0:n_iter) {
    if (iter > 0) {
      # Update estimated mean
      obs <- true_mu
      delta <- obs - est_mu
      alpha <- est_s2 / (sensory_s2 + est_s2)
      est_mu <- est_mu + alpha*delta
      # Update estimated s2
      est_s2 <- (1/est_s2 + 1/sensory_s2)^-1 
    }
    # Add layer
    img <- img + 
      stat_function(
        data = data.frame(x = c(-5, 8), iter = iter, s2 = sprintf('%s', sensory_s2)),
        mapping = aes(x = x, fill = iter),
        fun = dnorm, args = list(mean = est_mu, sd = sqrt(est_s2)),
        geom = "area",
        show.legend = F,
        alpha = 0.6,
        n = 10000)
  }
}

img_2_plot <- img + 
  scale_fill_gradient(high = rgb(0.2, 0.2, 0.2), low = rgb(0.8, 0.8, 0.8)) +
  facet_wrap(vars(s2), labeller = as_labeller(c('10' = 'High sensory noise', '2' = 'Low sensory noise'))) +
  theme_classic() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  scale_x_continuous(breaks = c(est_mu_initial, true_mu), labels = c('Initial estimate', 'Truth')) +
  labs(fill = 'N. obs.', x = 'Reward value', y = 'Probability density') +
  theme(
    text = element_text(size = 20),
    panel.spacing = unit(1, "cm"),
    panel.border = element_rect(fill = NA)
  )
  
plot(img_2_plot)

ggsave(plot = img_2_plot,
       filename = file.path('C:/Users/isaac/Projects/bayesian-rl-chapter/github/figs/est-convergence.png'),
       height = 15,
       width = 20,
       units = 'cm')
