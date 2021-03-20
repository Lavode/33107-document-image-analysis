library(ggplot2)
library(dplyr)
library(viridis)


df = read.csv("../data/handmade_classifiers.csv")
df$decision_mode = factor(df$decision_mode)
df$features = factor(df$features)
df$class = factor(df$class)

stats = df %>% group_by(decision_mode, features) %>% summarise(mean_accuracy = mean(accuracy))


ggplot(df, aes(x = class, y = accuracy, fill = features)) +
  geom_col(position = 'dodge', color = '#444444') +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::percent) +
  facet_wrap(vars(decision_mode), nrow = 3) +
  theme_minimal() +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Accuracy", x = "Class", fill = "Features") +
  ggsave("../features_overview.png", width = 20, units = 'cm', dpi = 'print')


df_single = df %>% filter(decision_mode == 'single')

ggplot(df_single, aes(x = class, y = accuracy, fill = features)) +
  geom_col(position = 'dodge', color = '#444444') +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Accuracy", x = "Class", fill = "Features") +
  ggsave("../features_single.png", width = 20, units = 'cm', dpi = 'print')



df_profiles = df %>% filter(features == 'Horizontal profile,Vertical profile' | features == 'Vertical profile' | features == 'Horizontal profile')

ggplot(df_profiles, aes(x = class, y = accuracy, fill = features:decision_mode)) +
  geom_col(position = 'dodge', color = "#444444") +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  scale_fill_viridis(discrete = TRUE) +
  labs(y = "Accuracy", x = "Class", fill = "Features") +
  ggsave("../features_profiles.png", width = 20, units = 'cm', dpi = 'print')

